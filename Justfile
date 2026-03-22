# Justfile for common tasks

# List available commands
default:
    @just --list

# Check code for syntax errors
check:
    cargo check

# Run strict linting
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
test:
    cargo test --all-targets --all-features

# Format code
fmt:
    cargo fmt

# Check formatting (for CI)
fmt-check:
    cargo fmt --check

# Build the project
build:
    cargo build

# Run the application
run:
    cargo run

# Fix auto-fixable lint issues
fix:
    cargo clippy --fix --allow-dirty --allow-staged
