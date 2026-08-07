use serde::{Deserialize, Serialize};

use crate::{config::DirectoryStore, db, index};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct StatusRoot {
    pub directory: String,
    pub watcher_live: bool,
    pub uptime_seconds: Option<u64>,
    pub documents: usize,
    pub chunks: usize,
    pub preparation_failures: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ReindexRoot {
    pub directory: String,
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub failed: usize,
}

impl From<(String, index::IndexStats)> for ReindexRoot {
    fn from((directory, stats): (String, index::IndexStats)) -> Self {
        Self {
            directory,
            added: stats.added,
            updated: stats.updated,
            deleted: stats.deleted,
            unchanged: stats.unchanged,
            failed: stats.failed,
        }
    }
}

impl std::fmt::Display for ReindexRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "added={}, updated={}, deleted={}, unchanged={}, failed={}",
            self.added, self.updated, self.deleted, self.unchanged, self.failed
        )
    }
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

    use super::{ReindexRoot, StatusRoot, index_counts, unavailable_status};
    use crate::{
        config,
        document::DocumentPreparer,
        embed::Embedder,
        types,
        watch::{self, OpenStore},
    };

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
        Reindex {
            retry_failed: bool,
            profile: types::IndexProfile,
        },
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(tag = "response", rename_all = "snake_case")]
    enum Response {
        Status {
            version: u32,
            roots: Vec<StatusRoot>,
        },
        Reindex {
            version: u32,
            roots: Vec<ReindexRoot>,
        },
        Error {
            version: u32,
            message: String,
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
            stores: &mut [OpenStore],
            started_at: Instant,
            embedder: &Embedder,
            preparer: &dyn DocumentPreparer,
            profile: &types::IndexProfile,
        ) -> anyhow::Result<()> {
            let (stream, _) = self.listener.accept().await?;
            serve_connection(stream, stores, started_at, embedder, preparer, profile).await
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

    pub async fn request_reindex(
        retry_failed: bool,
        profile: types::IndexProfile,
    ) -> anyhow::Result<Option<Vec<ReindexRoot>>> {
        request_reindex_at(&socket_path()?, retry_failed, profile).await
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
            Response::Error { version, message } => {
                check_version(version)?;
                anyhow::bail!(message)
            }
            Response::Reindex { .. } => Ok(None),
        }
    }

    async fn request_reindex_at(
        path: &Path,
        retry_failed: bool,
        profile: types::IndexProfile,
    ) -> anyhow::Result<Option<Vec<ReindexRoot>>> {
        let Ok(Ok(stream)) = timeout(CONNECTION_TIMEOUT, UnixStream::connect(path)).await else {
            return Ok(None);
        };
        let request = Request {
            version: PROTOCOL_VERSION,
            request: RequestType::Reindex {
                retry_failed,
                profile,
            },
        };
        let Ok(request) = serde_json::to_string(&request) else {
            return Ok(None);
        };
        tracing::info!("delegating reindex to the running watcher");
        let (read, mut write) = stream.into_split();
        if write.write_all(request.as_bytes()).await.is_err()
            || write.write_all(b"\n").await.is_err()
        {
            return Ok(None);
        }

        let mut response = String::new();
        let mut reader = BufReader::new(read);
        let Ok(bytes_read) = reader.read_line(&mut response).await else {
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
            Response::Reindex { version, roots } => {
                check_version(version)?;
                Ok(Some(roots))
            }
            Response::VersionMismatch {
                expected_version,
                received_version,
                ..
            } => protocol_version_mismatch(expected_version, received_version),
            Response::Error { version, message } => {
                check_version(version)?;
                anyhow::bail!(message)
            }
            Response::Status { .. } => Ok(None),
        }
    }

    async fn serve_connection(
        stream: UnixStream,
        stores: &mut [OpenStore],
        started_at: Instant,
        embedder: &Embedder,
        preparer: &dyn DocumentPreparer,
        profile: &types::IndexProfile,
    ) -> anyhow::Result<()> {
        let (read, mut write) = stream.into_split();
        let mut request = String::new();
        let mut reader = BufReader::new(read);
        let bytes_read = timeout(CONNECTION_TIMEOUT, reader.read_line(&mut request)).await??;
        if bytes_read == 0 {
            return Ok(());
        }
        let request: Request = serde_json::from_str(&request)?;
        let response = if request.version == PROTOCOL_VERSION {
            match request.request {
                RequestType::Status => Response::Status {
                    version: PROTOCOL_VERSION,
                    roots: status_roots(stores, started_at).await,
                },
                RequestType::Reindex {
                    retry_failed,
                    profile: client_profile,
                } => {
                    if profile == &client_profile {
                        match watch::reindex_in_place(stores, embedder, preparer, retry_failed)
                            .await
                        {
                            Ok(roots) => Response::Reindex {
                                version: PROTOCOL_VERSION,
                                roots: roots.into_iter().map(ReindexRoot::from).collect(),
                            },
                            Err(error) => Response::Error {
                                version: PROTOCOL_VERSION,
                                message: format!("{error:#}"),
                            },
                        }
                    } else {
                        Response::Error {
                            version: PROTOCOL_VERSION,
                            message: format!(
                                "index profile differs from the running watcher (watcher: {profile:?}; client: {client_profile:?}); restart the watcher service and rerun needle reindex"
                            ),
                        }
                    }
                }
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

    async fn status_roots(stores: &[OpenStore], started_at: Instant) -> Vec<StatusRoot> {
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
        roots
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
        use std::{os::unix::fs::FileTypeExt as _, path::Path, sync::Arc, time::Instant};

        use super::*;
        use crate::{
            control::local_status, db, document::DefaultPreparer, embed::Embedder, fts::FtsIndex,
            vector::VectorIndex,
        };

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

        fn create_file(dir: &Path, relative: &str, content: &str) {
            std::fs::write(dir.join(relative), content).expect("write file");
        }

        async fn open_store(
            notes_dir: &Path,
            profile: &types::IndexProfile,
        ) -> (Vec<tempfile::TempDir>, OpenStore) {
            let db_dir = tempfile::tempdir().expect("create database directory");
            let fts_dir = tempfile::tempdir().expect("create FTS directory");
            let (_db, conn) = db::connect_with_profile(&db_dir.path().join("test.db"), profile)
                .await
                .expect("connect database");
            let fts = FtsIndex::open_or_create(fts_dir.path()).expect("open FTS index");
            let vector_path = db_dir.path().join("test.usearch");
            let vector = Arc::new(
                VectorIndex::rebuild(&conn, vector_path.clone())
                    .await
                    .expect("build vector index"),
            );
            let store = OpenStore {
                notes_dir: notes_dir.to_path_buf(),
                conn,
                fts,
                vector_path,
                vector,
            };
            (vec![db_dir, fts_dir], store)
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
            let notes_dir = tempdir.path().join("notes");
            std::fs::create_dir(&notes_dir).expect("create notes directory");
            let embedder = Embedder::create_null(4);
            let preparer = DefaultPreparer::default();
            let profile = crate::index_profile(&embedder, &preparer);
            let (_temps, store) = open_store(&notes_dir, &profile).await;
            let mut stores = vec![store];
            let path = tempdir.path().join("watch.sock");
            let socket = ControlSocket::bind_at(&path).expect("bind socket");
            let server =
                socket.serve_next(&mut stores, Instant::now(), &embedder, &preparer, &profile);
            let (server, client) = tokio::join!(server, request_status_at(&path));
            server.expect("serve status");
            let status = client.expect("request status").expect("status response");
            assert_eq!(status.len(), 1);
            assert_eq!(status[0].directory, notes_dir.to_string_lossy());
            assert!(status[0].watcher_live);
            assert_eq!(status[0].documents, 0);
            assert_eq!(status[0].chunks, 0);
            assert_eq!(status[0].preparation_failures, 0);
        }

        #[tokio::test]
        async fn delegated_reindex_returns_statistics_for_each_root() {
            let tempdir = tempfile::tempdir().expect("create temp directory");
            let notes_dir = tempdir.path().join("notes");
            std::fs::create_dir(&notes_dir).expect("create notes directory");
            create_file(&notes_dir, "note.md", "indexed content");
            let embedder = Embedder::create_null(4);
            let preparer = DefaultPreparer::default();
            let profile = crate::index_profile(&embedder, &preparer);
            let (_temps, store) = open_store(&notes_dir, &profile).await;
            let mut stores = vec![store];
            let path = tempdir.path().join("watch.sock");
            let socket = ControlSocket::bind_at(&path).expect("bind socket");
            let server =
                socket.serve_next(&mut stores, Instant::now(), &embedder, &preparer, &profile);
            let (server, client) =
                tokio::join!(server, request_reindex_at(&path, false, profile.clone()));
            server.expect("serve reindex");
            let roots = client.expect("request reindex").expect("reindex response");

            assert_eq!(roots.len(), 1);
            assert_eq!(roots[0].directory, notes_dir.to_string_lossy());
            assert_eq!(roots[0].added, 1);
            assert_eq!(roots[0].updated, 0);
            assert_eq!(roots[0].deleted, 0);
            assert_eq!(roots[0].failed, 0);
        }

        #[tokio::test]
        async fn a_mismatched_profile_refuses_delegated_reindexing() {
            let tempdir = tempfile::tempdir().expect("create temp directory");
            let notes_dir = tempdir.path().join("notes");
            std::fs::create_dir(&notes_dir).expect("create notes directory");
            create_file(&notes_dir, "note.md", "not indexed");
            let embedder = Embedder::create_null(4);
            let preparer = DefaultPreparer::default();
            let profile = crate::index_profile(&embedder, &preparer);
            let mismatched_profile = types::IndexProfile {
                embedder: profile.embedder.clone(),
                preparer: "different-preparer".to_owned(),
            };
            let (_temps, store) = open_store(&notes_dir, &profile).await;
            let mut stores = vec![store];
            let path = tempdir.path().join("watch.sock");
            let socket = ControlSocket::bind_at(&path).expect("bind socket");
            let server =
                socket.serve_next(&mut stores, Instant::now(), &embedder, &preparer, &profile);
            let (server, client) =
                tokio::join!(server, request_reindex_at(&path, false, mismatched_profile));
            server.expect("serve refusal");
            let error = client
                .expect_err("mismatched profile must fail")
                .to_string();

            assert!(error.contains("watcher:"), "error: {error}");
            assert!(error.contains("client:"), "error: {error}");
            assert!(
                error.contains("restart the watcher service and rerun"),
                "error: {error}"
            );
            assert!(
                db::all_note_hashes(&stores[0].conn)
                    .await
                    .expect("note hashes")
                    .is_empty()
            );
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
pub use socket::{ControlSocket, request_reindex, request_status};

#[cfg(not(unix))]
pub struct ControlSocket;

#[cfg(not(unix))]
impl ControlSocket {
    pub const fn bind() -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub async fn serve_next(
        &self,
        _stores: &mut [crate::watch::OpenStore],
        _started_at: std::time::Instant,
        _embedder: &crate::embed::Embedder,
        _preparer: &dyn crate::document::DocumentPreparer,
        _profile: &crate::types::IndexProfile,
    ) -> anyhow::Result<()> {
        std::future::pending().await
    }
}
