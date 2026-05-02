#!/bin/bash
# Generate rust-toolchain.toml for the POLARIS project based on kernel config.
#
# Usage: ./scripts/gen-rust-toolchain.sh [KDIR=/path/to/kernel]
#
# Reads the kernel's minimum Rust version from scripts/min-tool-version.sh and,
# if available, the actual built-with version from .config.

set -euo pipefail

# Resolve kernel directory
KDIR="${KDIR:-}"
for candidate in \
    "$KDIR" \
    "/lib/modules/$(uname -r)/build" \
    "/lib/modules/$(uname -r)/source"; do
    if [ -d "$candidate" ] && { [ -f "$candidate/Makefile" ] || [ -f "$candidate/Kbuild" ]; }; then
        KDIR="$candidate"
        break
    fi
done

if [ -z "$KDIR" ] || [ ! -d "$KDIR" ]; then
    echo "Error: Cannot find kernel source tree." >&2
    echo "Set KDIR env var or pass as first argument." >&2
    exit 1
fi

KDIR="$(realpath "$KDIR")"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT="$PROJECT_ROOT/rust-toolchain.toml"

# Determine Rust version: prefer .config built-with version, fall back to min-tool-version.sh
RUST_VERSION=""
if [ -f "$KDIR/.config" ]; then
    RUST_VERSION=$(grep -oP 'CONFIG_RUSTC_VERSION_TEXT="rustc \K[0-9]+\.[0-9]+\.[0-9]+' "$KDIR/.config" 2>/dev/null || true)
fi

if [ -z "$RUST_VERSION" ] && [ -f "$KDIR/scripts/min-tool-version.sh" ]; then
    RUST_VERSION=$(bash "$KDIR/scripts/min-tool-version.sh" rustc)
fi

if [ -z "$RUST_VERSION" ]; then
    echo "Error: Cannot determine required Rust version from KDIR." >&2
    echo "Ensure .config or scripts/min-tool-version.sh exists in $KDIR." >&2
    exit 1
fi

# Determine host target
HOST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"

echo "POLARIS rust-toolchain.toml generator"
echo "  Kernel  : $KDIR"
echo "  Version : $RUST_VERSION"
echo "  Target  : $HOST_TARGET"

cat > "$OUTPUT" <<EOF
[toolchain]
channel = "$RUST_VERSION"
components = ["rust-src", "rustfmt", "clippy"]
targets = ["$HOST_TARGET"]
EOF

echo "Written: $OUTPUT"
echo "Run 'rustup show' to verify the toolchain is installed."
