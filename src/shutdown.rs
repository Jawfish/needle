#[cfg(unix)]
pub struct Shutdown {
    terminate: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
}

#[cfg(not(unix))]
pub struct Shutdown;

impl Shutdown {
    #[cfg(unix)]
    pub fn install() -> anyhow::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};

        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }

    #[cfg(not(unix))]
    pub fn install() -> anyhow::Result<Self> {
        Ok(Self)
    }

    #[cfg(unix)]
    pub async fn requested(&mut self) {
        tokio::select! {
            _ = self.terminate.recv() => {}
            _ = self.interrupt.recv() => {}
        }
    }

    #[cfg(not(unix))]
    pub async fn requested(&mut self) {
        let _ = tokio::signal::ctrl_c().await;
    }
}
