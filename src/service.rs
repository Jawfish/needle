use std::{fmt, str::FromStr};

const RESTART_DELAY_SECONDS: u64 = 5;
const STOP_TIMEOUT_SECONDS: u64 = 120;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Role {
    Watch,
    Serve,
    Reindex,
    Timer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Backend {
    Systemd,
    Launchd,
}

impl Backend {
    #[cfg(target_os = "macos")]
    pub const fn host() -> Self {
        Self::Launchd
    }

    #[cfg(not(target_os = "macos"))]
    pub const fn host() -> Self {
        Self::Systemd
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Systemd => formatter.write_str("systemd"),
            Self::Launchd => formatter.write_str("launchd"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Interval {
    text: String,
    seconds: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalParseError;

impl fmt::Display for IntervalParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("interval must be a positive whole number followed by s, m, or h")
    }
}

impl std::error::Error for IntervalParseError {}

impl FromStr for Interval {
    type Err = IntervalParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() < 2 {
            return Err(IntervalParseError);
        }

        let unit = value.chars().last().ok_or(IntervalParseError)?;
        let digits = &value[..value.len() - unit.len_utf8()];
        if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
            return Err(IntervalParseError);
        }

        let quantity = digits.parse::<u64>().map_err(|_| IntervalParseError)?;
        if quantity == 0 {
            return Err(IntervalParseError);
        }

        let multiplier = match unit {
            's' => 1,
            'm' => 60,
            'h' => 60 * 60,
            _ => return Err(IntervalParseError),
        };
        let seconds = quantity.checked_mul(multiplier).ok_or(IntervalParseError)?;

        Ok(Self {
            text: value.to_owned(),
            seconds,
        })
    }
}

impl Interval {
    const fn seconds(&self) -> u64 {
        self.seconds
    }

    fn as_systemd(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Definition {
    pub exec_path: String,
    pub home: String,
    pub log_level: String,
    pub interval: Interval,
    pub host: String,
    pub port: u16,
}

pub fn render(role: Role, backend: Backend, definition: &Definition) -> String {
    match backend {
        Backend::Systemd => render_systemd(role, definition),
        Backend::Launchd => render_launchd(role, definition),
    }
}

fn render_systemd(role: Role, definition: &Definition) -> String {
    match role {
        Role::Watch => format!(
            "# needle-watch.service\n[Unit]\nDescription=Needle document watcher\n\n[Service]\nExecStart={} watch\nRestart=on-failure\nRestartSec={RESTART_DELAY_SECONDS}\nTimeoutStopSec={STOP_TIMEOUT_SECONDS}\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG={}\n\n[Install]\nWantedBy=default.target\n",
            definition.exec_path, definition.log_level
        ),
        Role::Serve => format!(
            "# needle-serve.service\n[Unit]\nDescription=Needle document server\n\n[Service]\nExecStart={} serve --host {} --port {}\nRestart=on-failure\nRestartSec={RESTART_DELAY_SECONDS}\nTimeoutStopSec={STOP_TIMEOUT_SECONDS}\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG={}\n\n[Install]\nWantedBy=default.target\n",
            definition.exec_path, definition.host, definition.port, definition.log_level
        ),
        Role::Reindex => format!(
            "# needle-reindex.service\n[Unit]\nDescription=Needle document reindex\n\n[Service]\nType=oneshot\nExecStart={} reindex\nTimeoutStopSec={STOP_TIMEOUT_SECONDS}\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG={}\n",
            definition.exec_path, definition.log_level
        ),
        Role::Timer => format!(
            "# needle-reindex.timer\n[Unit]\nDescription=Needle periodic reindex\n\n[Timer]\nOnBootSec={}\nOnUnitActiveSec={}\nUnit=needle-reindex.service\n\n[Install]\nWantedBy=timers.target\n",
            definition.interval.as_systemd(),
            definition.interval.as_systemd()
        ),
    }
}

fn render_launchd(role: Role, definition: &Definition) -> String {
    let role_name = match role {
        Role::Watch => "watch",
        Role::Serve => "serve",
        Role::Reindex => "reindex",
        Role::Timer => "timer",
    };
    let program_arguments = launchd_program_arguments(role, definition);
    let schedule = if role == Role::Timer {
        format!(
            "  <key>StartInterval</key>\n  <integer>{}</integer>\n",
            definition.interval.seconds()
        )
    } else {
        "  <key>RunAtLoad</key>\n  <true/>\n".to_owned()
    };
    let keep_alive = if matches!(role, Role::Watch | Role::Serve) {
        format!(
            "  <key>KeepAlive</key>\n  <dict>\n    <key>SuccessfulExit</key>\n    <false/>\n  </dict>\n  <key>ThrottleInterval</key>\n  <integer>{RESTART_DELAY_SECONDS}</integer>\n"
        )
    } else {
        String::new()
    };
    let exec_path = xml_escape(&definition.exec_path);
    let home = xml_escape(&definition.home);
    let log_level = xml_escape(&definition.log_level);

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- dev.needle.{role_name}.plist -->\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>dev.needle.{role_name}</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>{exec_path}</string>\n{program_arguments}  </array>\n{schedule}{keep_alive}  <key>ExitTimeOut</key>\n  <integer>{STOP_TIMEOUT_SECONDS}</integer>\n  <key>ProcessType</key>\n  <string>Background</string>\n  <key>EnvironmentVariables</key>\n  <dict>\n    <key>NEEDLE_LOG</key>\n    <string>{log_level}</string>\n  </dict>\n  <key>StandardErrorPath</key>\n  <string>{home}/Library/Logs/needle.log</string>\n</dict>\n</plist>\n"
    )
}

fn launchd_program_arguments(role: Role, definition: &Definition) -> String {
    match role {
        Role::Watch => "    <string>watch</string>\n".to_owned(),
        Role::Serve => format!(
            "    <string>serve</string>\n    <string>--host</string>\n    <string>{}</string>\n    <string>--port</string>\n    <string>{}</string>\n",
            xml_escape(&definition.host),
            definition.port
        ),
        Role::Reindex | Role::Timer => "    <string>reindex</string>\n".to_owned(),
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> Definition {
        Definition {
            exec_path: "/opt/needle/bin/needle".to_owned(),
            home: "/Users/alice".to_owned(),
            log_level: "info".to_owned(),
            interval: "15m".parse().expect("interval"),
            host: "127.0.0.1".to_owned(),
            port: 8080,
        }
    }

    #[test]
    fn systemd_watcher_definition_matches_the_golden_output() {
        let actual = render(Role::Watch, Backend::Systemd, &definition());
        let expected = "# needle-watch.service\n[Unit]\nDescription=Needle document watcher\n\n[Service]\nExecStart=/opt/needle/bin/needle watch\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=120\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG=info\n\n[Install]\nWantedBy=default.target\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn systemd_server_definition_matches_the_golden_output() {
        let actual = render(Role::Serve, Backend::Systemd, &definition());
        let expected = "# needle-serve.service\n[Unit]\nDescription=Needle document server\n\n[Service]\nExecStart=/opt/needle/bin/needle serve --host 127.0.0.1 --port 8080\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=120\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG=info\n\n[Install]\nWantedBy=default.target\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn systemd_reindex_definition_matches_the_golden_output() {
        let actual = render(Role::Reindex, Backend::Systemd, &definition());
        let expected = "# needle-reindex.service\n[Unit]\nDescription=Needle document reindex\n\n[Service]\nType=oneshot\nExecStart=/opt/needle/bin/needle reindex\nTimeoutStopSec=120\nNice=10\nIOSchedulingClass=idle\nEnvironment=NEEDLE_LOG=info\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn systemd_timer_definition_matches_the_golden_output() {
        let actual = render(Role::Timer, Backend::Systemd, &definition());
        let expected = "# needle-reindex.timer\n[Unit]\nDescription=Needle periodic reindex\n\n[Timer]\nOnBootSec=15m\nOnUnitActiveSec=15m\nUnit=needle-reindex.service\n\n[Install]\nWantedBy=timers.target\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn launchd_watcher_definition_matches_the_golden_output() {
        let actual = render(Role::Watch, Backend::Launchd, &definition());
        let expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- dev.needle.watch.plist -->\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>dev.needle.watch</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>/opt/needle/bin/needle</string>\n    <string>watch</string>\n  </array>\n  <key>RunAtLoad</key>\n  <true/>\n  <key>KeepAlive</key>\n  <dict>\n    <key>SuccessfulExit</key>\n    <false/>\n  </dict>\n  <key>ThrottleInterval</key>\n  <integer>5</integer>\n  <key>ExitTimeOut</key>\n  <integer>120</integer>\n  <key>ProcessType</key>\n  <string>Background</string>\n  <key>EnvironmentVariables</key>\n  <dict>\n    <key>NEEDLE_LOG</key>\n    <string>info</string>\n  </dict>\n  <key>StandardErrorPath</key>\n  <string>/Users/alice/Library/Logs/needle.log</string>\n</dict>\n</plist>\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn launchd_server_definition_matches_the_golden_output() {
        let actual = render(Role::Serve, Backend::Launchd, &definition());
        let expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- dev.needle.serve.plist -->\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>dev.needle.serve</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>/opt/needle/bin/needle</string>\n    <string>serve</string>\n    <string>--host</string>\n    <string>127.0.0.1</string>\n    <string>--port</string>\n    <string>8080</string>\n  </array>\n  <key>RunAtLoad</key>\n  <true/>\n  <key>KeepAlive</key>\n  <dict>\n    <key>SuccessfulExit</key>\n    <false/>\n  </dict>\n  <key>ThrottleInterval</key>\n  <integer>5</integer>\n  <key>ExitTimeOut</key>\n  <integer>120</integer>\n  <key>ProcessType</key>\n  <string>Background</string>\n  <key>EnvironmentVariables</key>\n  <dict>\n    <key>NEEDLE_LOG</key>\n    <string>info</string>\n  </dict>\n  <key>StandardErrorPath</key>\n  <string>/Users/alice/Library/Logs/needle.log</string>\n</dict>\n</plist>\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn launchd_reindex_definition_matches_the_golden_output() {
        let actual = render(Role::Reindex, Backend::Launchd, &definition());
        let expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- dev.needle.reindex.plist -->\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>dev.needle.reindex</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>/opt/needle/bin/needle</string>\n    <string>reindex</string>\n  </array>\n  <key>RunAtLoad</key>\n  <true/>\n  <key>ExitTimeOut</key>\n  <integer>120</integer>\n  <key>ProcessType</key>\n  <string>Background</string>\n  <key>EnvironmentVariables</key>\n  <dict>\n    <key>NEEDLE_LOG</key>\n    <string>info</string>\n  </dict>\n  <key>StandardErrorPath</key>\n  <string>/Users/alice/Library/Logs/needle.log</string>\n</dict>\n</plist>\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn launchd_timer_definition_matches_the_golden_output() {
        let actual = render(Role::Timer, Backend::Launchd, &definition());
        let expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- dev.needle.timer.plist -->\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>dev.needle.timer</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>/opt/needle/bin/needle</string>\n    <string>reindex</string>\n  </array>\n  <key>StartInterval</key>\n  <integer>900</integer>\n  <key>ExitTimeOut</key>\n  <integer>120</integer>\n  <key>ProcessType</key>\n  <string>Background</string>\n  <key>EnvironmentVariables</key>\n  <dict>\n    <key>NEEDLE_LOG</key>\n    <string>info</string>\n  </dict>\n  <key>StandardErrorPath</key>\n  <string>/Users/alice/Library/Logs/needle.log</string>\n</dict>\n</plist>\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn intervals_parse_supported_units() {
        let seconds = "30s".parse::<Interval>().expect("seconds interval");
        let minutes = "15m".parse::<Interval>().expect("minutes interval");
        let hours = "2h".parse::<Interval>().expect("hours interval");

        assert_eq!(seconds.seconds(), 30);
        assert_eq!(minutes.seconds(), 900);
        assert_eq!(hours.seconds(), 7200);
        assert_eq!(minutes.as_systemd(), "15m");
    }

    #[test]
    fn intervals_reject_invalid_values() {
        for value in ["", "30", "0s", "5d", "s", "five minutes", "1.5h"] {
            assert!(value.parse::<Interval>().is_err(), "accepted {value}");
        }
    }

    #[test]
    fn launchd_paths_escape_xml_metacharacters() {
        let mut definition = definition();
        definition.exec_path = "/opt/needle&more<bin>".to_owned();
        definition.home = "/Users/a&b<test>".to_owned();

        let actual = render(Role::Watch, Backend::Launchd, &definition);

        assert!(actual.contains("<string>/opt/needle&amp;more&lt;bin&gt;</string>"));
        assert!(
            actual.contains("<string>/Users/a&amp;b&lt;test&gt;/Library/Logs/needle.log</string>")
        );
    }
}
