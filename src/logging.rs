const DEFAULT_DIRECTIVES: &str = "needle=info";

pub fn filter_directives(
    rust_log: Option<&str>,
    needle_log: Option<&str>,
    verbosity: u8,
) -> String {
    if let Some(directives) = non_empty(rust_log) {
        return directives.to_owned();
    }
    if let Some(level) = verbosity_level(verbosity) {
        return format!("needle={level}");
    }
    if let Some(value) = non_empty(needle_log) {
        return scope_to_needle(value);
    }
    DEFAULT_DIRECTIVES.to_owned()
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

const fn verbosity_level(verbosity: u8) -> Option<&'static str> {
    match verbosity {
        0 => None,
        1 => Some("debug"),
        _ => Some("trace"),
    }
}

fn scope_to_needle(value: &str) -> String {
    if value.contains('=') {
        value.to_owned()
    } else {
        format!("needle={value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_needle_only_info() {
        assert_eq!(filter_directives(None, None, 0), "needle=info");
    }

    #[test]
    fn one_verbose_flag_raises_needle_to_debug_without_naming_dependencies() {
        assert_eq!(filter_directives(None, None, 1), "needle=debug");
    }

    #[test]
    fn repeated_verbose_flags_raise_needle_to_trace() {
        assert_eq!(filter_directives(None, None, 2), "needle=trace");
        assert_eq!(filter_directives(None, None, 5), "needle=trace");
    }

    #[test]
    fn needle_variable_scopes_a_bare_level_to_needle() {
        assert_eq!(filter_directives(None, Some("debug"), 0), "needle=debug");
    }

    #[test]
    fn needle_variable_passes_through_explicit_directives() {
        assert_eq!(
            filter_directives(None, Some("needle::db=trace"), 0),
            "needle::db=trace"
        );
    }

    #[test]
    fn verbose_flag_wins_over_the_needle_variable() {
        assert_eq!(filter_directives(None, Some("warn"), 1), "needle=debug");
    }

    #[test]
    fn rust_log_is_honoured_verbatim() {
        assert_eq!(filter_directives(Some("info"), None, 0), "info");
    }

    #[test]
    fn rust_log_keeps_directives_naming_a_dependency() {
        assert_eq!(
            filter_directives(Some("warn,pdf_oxide=trace"), None, 0),
            "warn,pdf_oxide=trace"
        );
    }

    #[test]
    fn rust_log_outranks_both_the_flag_and_the_needle_variable() {
        assert_eq!(filter_directives(Some("error"), Some("trace"), 3), "error");
    }

    #[test]
    fn blank_values_fall_through_to_the_next_source() {
        assert_eq!(filter_directives(Some("   "), None, 0), "needle=info");
        assert_eq!(filter_directives(None, Some(""), 0), "needle=info");
    }
}
