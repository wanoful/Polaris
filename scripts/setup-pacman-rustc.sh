#!/bin/bash
# setup-pacman-rustc.sh — Download and extract Arch Linux's rustc package
# locally for Polaris kernel module builds.
#
# Usage:
#   ./scripts/setup-pacman-rustc.sh                  # auto-detect version from kernel
#   ./scripts/setup-pacman-rustc.sh 1.93.0            # specify exact version
#   ./scripts/setup-pacman-rustc.sh --rustup-link     # also register as a rustup toolchain
#
# The pacman rustc is needed because the kernel's Rust-for-Linux support
# requires the exact compiler build that compiled the kernel.  Even the same
# version number from rustup can produce binary-incompatible code due to
# different LLVM / linker / patch sets.
#
# This script downloads the package WITHOUT installing it system-wide and
# extracts it to .tools/pacman-rustc/ in the project root.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACMAN_DIR="$PROJECT_ROOT/.tools/pacman-rustc"
LINK_NAME="${POLARIS_RUSTUP_LINK_NAME:-polaris-kernel}"
DO_RUSTUP_LINK=false

# Arch's rustc is dynamically linked against librustc_driver-*.so which
# lives in usr/lib/.  We need to set LD_LIBRARY_PATH whenever we invoke it.
run_rustc() {
    LD_LIBRARY_PATH="$PACMAN_DIR/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PACMAN_DIR/usr/bin/rustc" "$@"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [VERSION]

Download and extract Arch's pacman rustc package for local use with Polaris.

Options:
  --rustup-link    Register the extracted rustc as a custom rustup toolchain
                   named '$LINK_NAME' (override via POLARIS_RUSTUP_LINK_NAME env).
  -h, --help       Show this message.

Examples:
  $(basename "$0")                       # auto-detect kernel's rustc version
  $(basename "$0") 1.93.0                # download a specific version from archive
  $(basename "$0") --rustup-link         # also make it available to cargo via rustup
EOF
    exit 0
}

# --- argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rustup-link) DO_RUSTUP_LINK=true; shift ;;
        -h|--help) usage ;;
        --*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) REQUESTED_VERSION="$1"; shift ;;
    esac
done

# --- determine target version -------------------------------------------------
if [[ -n "${REQUESTED_VERSION:-}" ]]; then
    TARGET_VERSION="$REQUESTED_VERSION"
else
    # Auto-detect from running kernel config
    if KDIR="${KDIR:-/lib/modules/$(uname -r)/build}"; [ -f "$KDIR/.config" ]; then
        # Try CONFIG_RUSTC_VERSION_TEXT (human-readable, e.g. "rustc 1.93.0")
        TARGET_VERSION=$(grep -oP 'CONFIG_RUSTC_VERSION_TEXT="rustc \K[0-9]+\.[0-9]+\.[0-9]+' "$KDIR/.config" 2>/dev/null || true)
        # Fall back to CONFIG_RUSTC_VERSION (integer, e.g. 109300)
        if [[ -z "$TARGET_VERSION" ]]; then
            raw_ver=$(grep -oP 'CONFIG_RUSTC_VERSION=\K[0-9]+' "$KDIR/.config" 2>/dev/null || true)
            if [[ -n "$raw_ver" ]]; then
                MAJOR=$(( raw_ver / 100000 ))
                MINOR=$(( (raw_ver % 100000) / 1000 ))
                PATCH=$(( raw_ver % 1000 ))
                TARGET_VERSION="${MAJOR}.${MINOR}.${PATCH}"
            fi
        fi
    fi
    if [[ -z "${TARGET_VERSION:-}" ]]; then
        echo "Error: Cannot auto-detect kernel rustc version." >&2
        echo "Specify a version explicitly:  $(basename "$0") VER" >&2
        exit 1
    fi
fi

echo "==> Target rustc version: $TARGET_VERSION"
echo "==> Extract directory:    $PACMAN_DIR"

# --- check if already set up --------------------------------------------------
if [[ -x "$PACMAN_DIR/usr/bin/rustc" ]]; then
    INSTALLED_VER=$(run_rustc --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    if [[ "$INSTALLED_VER" == "$TARGET_VERSION" ]]; then
        echo "==> Already set up (version $INSTALLED_VER matches)."
        if $DO_RUSTUP_LINK; then link_rustup; fi
        exit 0
    fi
    echo "==> Existing version $INSTALLED_VER differs from target $TARGET_VERSION."
    echo "==> Removing old extraction..."
    rm -rf "$PACMAN_DIR"
fi

mkdir -p "$(dirname "$PACMAN_DIR")"

# --- try to download the package ----------------------------------------------
download_pkg() {
    local ver="$1"
    local pkgname="rust-${ver}-x86_64"
    local cache_dir="/var/cache/pacman/pkg"
    local pkg_file="${pkgname}.pkg.tar.zst"
    local found=false
    local url

    # Check local pacman cache first
    for f in "$cache_dir/rust-${ver}"*.pkg.tar* "$cache_dir/rust-1%3A${ver}"*.pkg.tar*; do
        if [[ -f "$f" ]]; then
            echo "==> Found in pacman cache: $f"
            echo "==> Extracting..."
            mkdir -p "$PACMAN_DIR"
            bsdtar -xf "$f" -C "$PACMAN_DIR"
            found=true
            break
        fi
    done
    if $found; then return 0; fi

    # Try main repos via pacman -Sw
    echo "==> Downloading rust package (download only, no install)..."
    if sudo pacman -Sw --noconfirm "rust=${ver}" 2>/dev/null; then
        for f in "$cache_dir/rust-${ver}"*.pkg.tar* "$cache_dir/rust-1%3A${ver}"*.pkg.tar*; do
            if [[ -f "$f" ]]; then
                mkdir -p "$PACMAN_DIR"
                bsdtar -xf "$f" -C "$PACMAN_DIR"
                found=true
                break
            fi
        done
        if $found; then return 0; fi
    fi

    # Fall back to Arch Linux Archive
    local archive_url="https://archive.archlinux.org/packages/r/rust"
    echo "==> Trying Arch Linux Archive..."
    # Fetch the package list page, find the right pkg
    local html; html=$(curl -sL "$archive_url/" 2>/dev/null || true)
    # Extract the filename matching our version
    local filename; filename=$(echo "$html" | grep -oP "rust-${ver}(-[^-]*)?-x86_64\\.pkg\\.tar\\.(xz|zst)" | head -1)
    if [[ -z "$filename" ]]; then
        filename=$(echo "$html" | grep -oP "rust-1%3A${ver}(-[^-]*)?-x86_64\\.pkg\\.tar\\.(xz|zst)" | head -1)
    fi
    if [[ -z "$filename" ]]; then
        echo "Error: Cannot find rust ${ver} in Arch repos or archive." >&2
        echo "Check https://archive.archlinux.org/packages/r/rust/ manually." >&2
        return 1
    fi

    url="${archive_url}/${filename}"
    echo "==> Downloading $url ..."
    local tmpfile; tmpfile="$(mktemp)"
    curl -sL "$url" -o "$tmpfile"
    mkdir -p "$PACMAN_DIR"
    bsdtar -xf "$tmpfile" -C "$PACMAN_DIR"
    rm -f "$tmpfile"
    return 0
}

if ! download_pkg "$TARGET_VERSION"; then
    echo >&2
    echo "Manual alternative:" >&2
    echo "  1. Download rust-<version>-x86_64.pkg.tar.zst from" >&2
    echo "     https://archive.archlinux.org/packages/r/rust/" >&2
    echo "  2. Extract it to $PACMAN_DIR" >&2
    echo "     bsdtar -xf rust-*.pkg.tar.zst -C $PACMAN_DIR" >&2
    exit 1
fi

# --- verify rustc and its sysroot -----------------------------------------
if [[ ! -x "$PACMAN_DIR/usr/bin/rustc" ]]; then
    echo "Error: Extraction failed — rustc binary not found." >&2
    exit 1
fi

echo "==> rustc: $(run_rustc --version)"

# Ensure rustc can find its own sysroot (lib/rustlib/ relative to binary).
# rustc resolves this automatically by looking at ../lib/rustlib/ from the
# binary path, which matches the Arch pacman package layout:
#   .tools/pacman-rustc/usr/{bin/rustc, lib/rustlib/...}
SYSROOT="$(run_rustc --print sysroot 2>/dev/null || true)"
EXPECTED_SYSROOT="$PACMAN_DIR/usr"
if [[ -z "$SYSROOT" ]]; then
    echo "Warning: rustc could not print sysroot." >&2
elif [[ "$SYSROOT" != "$EXPECTED_SYSROOT" ]]; then
    echo "Warning: rustc reports sysroot as '$SYSROOT' but expected '$EXPECTED_SYSROOT'." >&2
    echo "This may indicate the package layout doesn't match. Kernel build may fail." >&2
    echo "If so, set RUSTC_SYSROOT (or POLARIS_RUSTC) directly: make kernel POLARIS_RUSTC=..." >&2
else
    echo "==> Sysroot verified: $SYSROOT"
fi

# Verify lib/rustlib exists
if [[ ! -d "$PACMAN_DIR/usr/lib/rustlib" ]]; then
    echo "Error: No lib/rustlib/ found in extracted package. The pacman rust package may have changed layout." >&2
    exit 1
fi

echo "==> Done. Local rustc at: $PACMAN_DIR/usr/bin/rustc"
echo "==> The Makefile 'kernel' target will auto-detect this."

# --- optional rustup link --------------------------------------------------
link_rustup() {
    if ! command -v rustup &>/dev/null; then
        echo "==> rustup not found, skipping toolchain link."
        return
    fi
    echo "==> Registering rustup custom toolchain: $LINK_NAME"
    rustup toolchain link "$LINK_NAME" "$PACMAN_DIR/usr"
    echo "==> Usage: cargo +$LINK_NAME build"
    echo "==> To override for this project:"
    echo "    rustup override set $LINK_NAME"
}

if $DO_RUSTUP_LINK; then
    link_rustup
fi
