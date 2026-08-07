#![cfg(unix)]

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader},
        process::{Child, Command, ExitStatus, Stdio},
        sync::mpsc::{self, Receiver},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
    const EXIT_TIMEOUT: Duration = Duration::from_secs(5);
    const POLL_INTERVAL: Duration = Duration::from_millis(10);
    const VECTOR_REBUILD_LOG: &str = "rebuilding vector index";

    struct ServiceEnvironment {
        root: tempfile::TempDir,
    }

    impl ServiceEnvironment {
        fn new() -> Self {
            let root = tempfile::tempdir().expect("create temp directory");
            let config_dir = root.path().join("config");
            let notes_dir = root.path().join("notes");
            std::fs::create_dir_all(config_dir.join("needle")).expect("create config directory");
            std::fs::create_dir(&notes_dir).expect("create notes directory");
            let notes_path = notes_dir.to_str().expect("temp path is UTF-8");
            std::fs::write(
                config_dir.join("needle/config.toml"),
                format!(
                    r#"provider = "openai"
model = "test"
api_base = "http://127.0.0.1:9/v1"
dim = 4
openai_api_key = "dummy"
needle_api_key = "dummy"

[[namespaces]]
name = "empty"
paths = ["{notes_path}"]
"#
                ),
            )
            .expect("write config");
            Self { root }
        }

        fn config_dir(&self) -> std::path::PathBuf {
            self.root.path().join("config")
        }

        fn data_dir(&self) -> std::path::PathBuf {
            self.root.path().join("data")
        }
    }

    struct ServiceProcess {
        child: Child,
        stderr: Receiver<std::io::Result<String>>,
        logs: Vec<String>,
        reader: Option<JoinHandle<()>>,
    }

    impl ServiceProcess {
        fn spawn(environment: &ServiceEnvironment, args: &[&str]) -> Self {
            let mut child = Command::new(env!("CARGO_BIN_EXE_needle"))
                .args(args)
                .env("XDG_CONFIG_HOME", environment.config_dir())
                .env("XDG_DATA_HOME", environment.data_dir())
                .env_remove("RUST_LOG")
                .env_remove("NEEDLE_LOG")
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .expect("start needle");
            let stderr = child.stderr.take().expect("capture stderr");
            let (sender, receiver) = mpsc::channel();
            let reader = thread::spawn(move || {
                for line in BufReader::new(stderr).lines() {
                    if sender.send(line).is_err() {
                        return;
                    }
                }
            });

            Self {
                child,
                stderr: receiver,
                logs: Vec::new(),
                reader: Some(reader),
            }
        }

        fn wait_for_marker(&mut self, marker: &str) -> Result<(), String> {
            let deadline = Instant::now() + STARTUP_TIMEOUT;
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(format!(
                        "did not observe {marker:?} before the deadline; stderr:\n{}",
                        self.logs()
                    ));
                }
                match self.stderr.recv_timeout(remaining) {
                    Ok(Ok(line)) => {
                        let found = line.contains(marker);
                        self.logs.push(line);
                        if found {
                            return Ok(());
                        }
                    }
                    Ok(Err(error)) => {
                        self.logs.push(format!("error reading stderr: {error}"));
                        return Err(format!(
                            "failed while waiting for {marker:?}; stderr:\n{}",
                            self.logs()
                        ));
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        return Err(format!(
                            "did not observe {marker:?} before the deadline; stderr:\n{}",
                            self.logs()
                        ));
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(format!(
                            "stderr closed before {marker:?}; stderr:\n{}",
                            self.logs()
                        ));
                    }
                }
            }
        }

        fn send_signal(&self, signal: &str) {
            let status = Command::new("kill")
                .arg(signal)
                .arg(self.child.id().to_string())
                .status()
                .expect("run kill");
            assert!(status.success(), "kill {signal} failed with {status:?}");
        }

        fn wait_for_exit(&mut self) -> Result<ExitStatus, String> {
            let deadline = Instant::now() + EXIT_TIMEOUT;
            loop {
                self.collect_available_logs();
                match self.child.try_wait() {
                    Ok(Some(status)) => return Ok(status),
                    Ok(None) => {}
                    Err(error) => return Err(format!("check needle exit status: {error}")),
                }
                if Instant::now() >= deadline {
                    self.abort();
                    return Err(format!(
                        "needle did not exit before the deadline; stderr:\n{}",
                        self.logs()
                    ));
                }
                thread::sleep(POLL_INTERVAL);
            }
        }

        fn abort(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }

        fn contains_log(&self, marker: &str) -> bool {
            self.logs.iter().any(|line| line.contains(marker))
        }

        fn collect_available_logs(&mut self) {
            while let Ok(Ok(line)) = self.stderr.try_recv() {
                self.logs.push(line);
            }
        }

        fn logs(&self) -> String {
            self.logs.join("\n")
        }
    }

    impl Drop for ServiceProcess {
        fn drop(&mut self) {
            self.abort();
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
    }

    fn stop_and_assert_successfully(process: &mut ServiceProcess, signal: &str) {
        process.send_signal(signal);
        process
            .wait_for_marker("shutting down")
            .expect("observe shutdown log");
        let status = process.wait_for_exit().expect("wait for needle exit");
        assert!(
            status.success(),
            "needle exited unsuccessfully with {status:?}; stderr:\n{}",
            process.logs()
        );
    }

    #[test]
    fn watch_exits_successfully_on_sigterm() {
        let environment = ServiceEnvironment::new();
        let mut process = ServiceProcess::spawn(&environment, &["watch"]);
        process
            .wait_for_marker("watching for changes")
            .expect("observe watch readiness");
        stop_and_assert_successfully(&mut process, "-TERM");
    }

    #[test]
    fn watch_exits_successfully_on_sigint() {
        let environment = ServiceEnvironment::new();
        let mut process = ServiceProcess::spawn(&environment, &["watch"]);
        process
            .wait_for_marker("watching for changes")
            .expect("observe watch readiness");
        stop_and_assert_successfully(&mut process, "-INT");
    }

    #[test]
    fn serve_exits_successfully_on_sigterm() {
        let environment = ServiceEnvironment::new();
        let mut process = ServiceProcess::spawn(&environment, &["serve", "--port", "0"]);
        process
            .wait_for_marker("serving indexed documents at http://")
            .expect("observe server readiness");
        stop_and_assert_successfully(&mut process, "-TERM");
    }

    #[test]
    fn restarting_watch_does_not_rebuild_the_vector_index() {
        let environment = ServiceEnvironment::new();
        let mut first = ServiceProcess::spawn(&environment, &["watch"]);
        first
            .wait_for_marker("watching for changes")
            .expect("observe first watch readiness");
        assert!(
            first.contains_log(VECTOR_REBUILD_LOG),
            "first watch did not build the vector index; stderr:\n{}",
            first.logs()
        );
        stop_and_assert_successfully(&mut first, "-TERM");

        let mut second = ServiceProcess::spawn(&environment, &["watch"]);
        second
            .wait_for_marker("watching for changes")
            .expect("observe second watch readiness");
        assert!(
            !second.contains_log(VECTOR_REBUILD_LOG),
            "second watch rebuilt the vector index; stderr:\n{}",
            second.logs()
        );
        stop_and_assert_successfully(&mut second, "-TERM");
    }
}
