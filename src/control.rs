use serde::{Deserialize, Serialize};

use crate::{config::DirectoryStore, db};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct StatusRoot {
    pub directory: String,
    pub watcher_live: bool,
    pub uptime_seconds: Option<u64>,
    pub documents: usize,
    pub chunks: usize,
    pub preparation_failures: usize,
}

struct IndexCounts {
    documents: usize,
    chunks: usize,
    preparation_failures: usize,
}

pub async fn status(stores: &[DirectoryStore]) -> anyhow::Result<Vec<StatusRoot>> {
    #[cfg(unix)]
    if let Some(status) = request_status().await? {
        return Ok(status);
    }

    Ok(local_status(stores).await)
}

async fn local_status(stores: &[DirectoryStore]) -> Vec<StatusRoot> {
    let mut status = Vec::with_capacity(stores.len());
    for store in stores {
        status.push(status_for_directory(&store.notes_dir, false, None, &store.db_path).await);
    }
    status
}

async fn status_for_directory(
    directory: &std::path::Path,
    watcher_live: bool,
    uptime_seconds: Option<u64>,
    db_path: &std::path::Path,
) -> StatusRoot {
    let counts = async {
        let (_db, conn) = db::connect(db_path, None).await?;
        index_counts(&conn).await
    }
    .await;
    match counts {
        Ok(counts) => StatusRoot {
            directory: directory.to_string_lossy().into_owned(),
            watcher_live,
            uptime_seconds,
            documents: counts.documents,
            chunks: counts.chunks,
            preparation_failures: counts.preparation_failures,
        },
        Err(error) => {
            tracing::warn!(directory = %directory.display(), error = %error, "failed to read index status");
            unavailable_status(directory, watcher_live, uptime_seconds)
        }
    }
}

fn unavailable_status(
    directory: &std::path::Path,
    watcher_live: bool,
    uptime_seconds: Option<u64>,
) -> StatusRoot {
    StatusRoot {
        directory: directory.to_string_lossy().into_owned(),
        watcher_live,
        uptime_seconds,
        documents: 0,
        chunks: 0,
        preparation_failures: 0,
    }
}

async fn index_counts(conn: &libsql::Connection) -> anyhow::Result<IndexCounts> {
    Ok(IndexCounts {
        documents: db::all_note_hashes(conn).await?.len(),
        chunks: db::chunk_index_state(conn).await?.0,
        preparation_failures: db::failed_files(conn).await?.len(),
    })
}

#[cfg(unix)]
mod socket {
    use std::{
        os::unix::fs::PermissionsExt as _,
        path::{Path, PathBuf},
        time::{Duration, Instant},
    };

    use serde::{Deserialize, Serialize};
    use tokio::{
        io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader},
        net::{UnixListener, UnixStream},
        time::timeout,
    };

    use super::{StatusRoot, index_counts, unavailable_status};
    use crate::{config, watch::OpenStore};

    const PROTOCOL_VERSION: u32 = 1;
    const CONNECTION_TIMEOUT: Duration = Duration::from_secs(1);

    #[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
    struct Request {
        version: u32,
        request: RequestType,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum RequestType {
        Status,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(tag = "response", rename_all = "snake_case")]
    enum Response {
        Status {
            version: u32,
            roots: Vec<StatusRoot>,
        },
        VersionMismatch {
            version: u32,
            expected_version: u32,
            received_version: u32,
        },
    }

    pub struct ControlSocket {
        listener: UnixListener,
        path: PathBuf,
    }

    impl ControlSocket {
        pub fn bind() -> anyhow::Result<Self> {
            Self::bind_at(&socket_path()?)
        }

        fn bind_at(path: &Path) -> anyhow::Result<Self> {
            let parent = path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("control socket path has no parent"))?;
            std::fs::create_dir_all(parent)?;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            let listener = UnixListener::bind(path)?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
            Ok(Self {
                listener,
                path: path.to_path_buf(),
            })
        }

        pub async fn serve_next(
            &self,
            stores: &[OpenStore],
            started_at: Instant,
        ) -> anyhow::Result<()> {
            let (stream, _) = self.listener.accept().await?;
            let uptime_seconds = Some(started_at.elapsed().as_secs());
            let mut roots = Vec::with_capacity(stores.len());
            for store in stores {
                let counts = index_counts(&store.conn).await;
                let root = match counts {
                    Ok(counts) => StatusRoot {
                        directory: store.notes_dir.to_string_lossy().into_owned(),
                        watcher_live: true,
                        uptime_seconds,
                        documents: counts.documents,
                        chunks: counts.chunks,
                        preparation_failures: counts.preparation_failures,
                    },
                    Err(error) => {
                        tracing::warn!(directory = %store.notes_dir.display(), error = %error, "failed to read index status");
                        unavailable_status(&store.notes_dir, true, uptime_seconds)
                    }
                };
                roots.push(root);
            }
            serve_connection(stream, roots).await
        }
    }

    impl Drop for ControlSocket {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_file(&self.path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(path = %self.path.display(), error = %error, "failed to remove control socket");
            }
        }
    }

    pub async fn request_status() -> anyhow::Result<Option<Vec<StatusRoot>>> {
        request_status_at(&socket_path()?).await
    }

    async fn request_status_at(path: &Path) -> anyhow::Result<Option<Vec<StatusRoot>>> {
        let Ok(Ok(stream)) = timeout(CONNECTION_TIMEOUT, UnixStream::connect(path)).await else {
            return Ok(None);
        };
        let request = Request {
            version: PROTOCOL_VERSION,
            request: RequestType::Status,
        };
        let Ok(request) = serde_json::to_string(&request) else {
            return Ok(None);
        };
        let (read, mut write) = stream.into_split();
        if write.write_all(request.as_bytes()).await.is_err()
            || write.write_all(b"\n").await.is_err()
        {
            return Ok(None);
        }

        let mut response = String::new();
        let mut reader = BufReader::new(read);
        let Ok(Ok(bytes_read)) = timeout(CONNECTION_TIMEOUT, reader.read_line(&mut response)).await
        else {
            return Ok(None);
        };
        if bytes_read == 0 {
            return Ok(None);
        }
        let response: Response = match serde_json::from_str(&response) {
            Ok(response) => response,
            Err(_) => return Ok(None),
        };
        match response {
            Response::Status { version, roots } => {
                check_version(version)?;
                Ok(Some(roots))
            }
            Response::VersionMismatch {
                expected_version,
                received_version,
                ..
            } => protocol_version_mismatch(expected_version, received_version),
        }
    }

    async fn serve_connection(stream: UnixStream, roots: Vec<StatusRoot>) -> anyhow::Result<()> {
        let (read, mut write) = stream.into_split();
        let mut request = String::new();
        let mut reader = BufReader::new(read);
        let bytes_read = timeout(CONNECTION_TIMEOUT, reader.read_line(&mut request)).await??;
        if bytes_read == 0 {
            return Ok(());
        }
        let request: Request = serde_json::from_str(&request)?;
        let response = if request.version == PROTOCOL_VERSION {
            Response::Status {
                version: PROTOCOL_VERSION,
                roots,
            }
        } else {
            Response::VersionMismatch {
                version: PROTOCOL_VERSION,
                expected_version: PROTOCOL_VERSION,
                received_version: request.version,
            }
        };
        let response = serde_json::to_string(&response)?;
        write.write_all(response.as_bytes()).await?;
        write.write_all(b"\n").await?;
        Ok(())
    }

    fn check_version(version: u32) -> anyhow::Result<()> {
        if version == PROTOCOL_VERSION {
            Ok(())
        } else {
            protocol_version_mismatch(PROTOCOL_VERSION, version)
        }
    }

    fn protocol_version_mismatch<T>(expected: u32, received: u32) -> anyhow::Result<T> {
        anyhow::bail!("control protocol version mismatch: expected {expected}, received {received}")
    }

    fn socket_path() -> anyhow::Result<PathBuf> {
        match std::env::var_os("XDG_RUNTIME_DIR").filter(|path| !path.is_empty()) {
            Some(runtime_dir) => Ok(PathBuf::from(runtime_dir).join("needle/watch.sock")),
            None => Ok(config::data_dir()?.join("watch.sock")),
        }
    }

    #[cfg(test)]
    mod tests {
        use std::os::unix::fs::FileTypeExt as _;

        use super::*;
        use crate::control::local_status;

        fn root() -> StatusRoot {
            StatusRoot {
                directory: "/notes".to_owned(),
                watcher_live: true,
                uptime_seconds: Some(42),
                documents: 3,
                chunks: 7,
                preparation_failures: 2,
            }
        }

        #[test]
        fn control_messages_round_trip_and_reject_version_mismatches() {
            let request = Request {
                version: PROTOCOL_VERSION,
                request: RequestType::Status,
            };
            let request_json = serde_json::to_string(&request).expect("serialize request");
            let parsed_request: Request =
                serde_json::from_str(&request_json).expect("parse request");
            assert_eq!(parsed_request, request);

            let response = Response::Status {
                version: PROTOCOL_VERSION,
                roots: vec![root()],
            };
            let response_json = serde_json::to_string(&response).expect("serialize response");
            let parsed_response: Response =
                serde_json::from_str(&response_json).expect("parse response");
            assert!(
                matches!(&parsed_response, Response::Status { .. }),
                "response must be status"
            );
            let Response::Status { version, roots } = parsed_response else {
                return;
            };
            check_version(version).expect("version matches");
            assert_eq!(roots, vec![root()]);

            let mismatch = r#"{"version":2,"request":"status"}"#;
            let mismatch: Request = serde_json::from_str(mismatch).expect("parse mismatch");
            assert!(check_version(mismatch.version).is_err());
        }

        #[tokio::test]
        async fn a_socket_server_returns_status_counts() {
            let tempdir = tempfile::tempdir().expect("create temp directory");
            let path = tempdir.path().join("watch.sock");
            let socket = ControlSocket::bind_at(&path).expect("bind socket");
            let expected = vec![root()];
            let server = async {
                let (stream, _) = socket.listener.accept().await?;
                serve_connection(stream, expected.clone()).await
            };
            let (server, client) = tokio::join!(server, request_status_at(&path));
            server.expect("serve status");
            assert_eq!(client.expect("request status"), Some(expected));
        }

        #[tokio::test]
        async fn binding_replaces_a_stale_socket_file() {
            let tempdir = tempfile::tempdir().expect("create temp directory");
            let path = tempdir.path().join("watch.sock");
            let stale = std::os::unix::net::UnixListener::bind(&path).expect("bind stale socket");
            drop(stale);

            let socket = ControlSocket::bind_at(&path).expect("replace stale socket");
            let metadata = std::fs::metadata(&path).expect("read socket metadata");
            assert!(metadata.file_type().is_socket());
            assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
            drop(socket);
        }

        #[tokio::test]
        async fn status_uses_the_local_index_when_no_socket_is_listening() {
            let tempdir = tempfile::tempdir().expect("create temp directory");
            let notes_dir = tempdir.path().join("notes");
            std::fs::create_dir(&notes_dir).expect("create notes directory");
            let store = crate::config::DirectoryStore {
                notes_dir: notes_dir.clone(),
                db_path: tempdir.path().join("needle.db"),
                tantivy_dir: tempdir.path().join("tantivy"),
            };
            let socket_path = tempdir.path().join("missing.sock");
            let status = status_from_socket_or_local_at(&[store], &socket_path)
                .await
                .expect("read local status");

            assert_eq!(status.len(), 1);
            assert_eq!(status[0].directory, notes_dir.to_string_lossy());
            assert!(!status[0].watcher_live);
            assert_eq!(status[0].uptime_seconds, None);
            assert_eq!(status[0].documents, 0);
            assert_eq!(status[0].chunks, 0);
            assert_eq!(status[0].preparation_failures, 0);
        }

        async fn status_from_socket_or_local_at(
            stores: &[crate::config::DirectoryStore],
            path: &Path,
        ) -> anyhow::Result<Vec<StatusRoot>> {
            match request_status_at(path).await? {
                Some(status) => Ok(status),
                None => Ok(local_status(stores).await),
            }
        }
    }
}

#[cfg(unix)]
pub use socket::{ControlSocket, request_status};

#[cfg(not(unix))]
pub struct ControlSocket;

#[cfg(not(unix))]
impl ControlSocket {
    pub const fn bind() -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub async fn serve_next(
        &self,
        _stores: &[crate::watch::OpenStore],
        _started_at: std::time::Instant,
    ) -> anyhow::Result<()> {
        std::future::pending().await
    }
}
