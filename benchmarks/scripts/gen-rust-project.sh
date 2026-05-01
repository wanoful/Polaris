#!/bin/bash
# Generate rust-project.json for the POLARIS out-of-tree kernel module.
#
# Usage: ./scripts/gen-rust-project.sh [KDIR=/path/to/kernel]
#
# This invokes the kernel's generate_rust_analyzer.py with exttree pointing
# to our module source, then patches the output to add OBJTREE and source
# include directories for proper rust-analyzer resolution.

set -euo pipefail

# Resolve kernel directory
KDIR="${KDIR:-}"
for candidate in \
    "/lib/modules/$(uname -r)/build" \
    "/lib/modules/$(uname -r)/source"; do
    if [ -f "$candidate/Kbuild" ] || [ -f "$candidate/Makefile" ]; then
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
EXTTREE="$PROJECT_ROOT/kernel"

echo "POLARIS rust-analyzer config generator"
echo "  Kernel  : $KDIR"
echo "  Module  : $EXTTREE"

export RUSTC="${RUSTC:-$(which rustc)}"
RUSTC_SYSROOT="$($RUSTC --print sysroot)"
RUST_LIB_SRC="$RUSTC_SYSROOT/lib/rustlib/src/rust"
CORE_CFGS="--cfg no_fp_fmt_parse"
CORE_EDITION="2024"

# Generate base rust-project.json via kernel script
python3 "$KDIR/scripts/generate_rust_analyzer.py" \
    --cfgs "core=$CORE_CFGS" \
    --cfgs 'proc_macro2=feature="proc-macro" wrap_proc_macro proc_macro_span_file proc_macro_span_location' \
    --cfgs 'quote=feature="proc-macro"' \
    --cfgs 'syn=feature="clone-impls" feature="derive" feature="full" feature="parsing" feature="printing" feature="proc-macro" feature="visit-mut"' \
    "$CORE_EDITION" \
    "$KDIR" \
    "$KDIR" \
    "$RUSTC_SYSROOT" \
    "$RUST_LIB_SRC" \
    "$EXTTREE" \
    > "$PROJECT_ROOT/rust-project.json"

# Patch the polaris crate entry with OBJTREE and source include dirs
python3 -c "
import json, sys

with open('$PROJECT_ROOT/rust-project.json') as f:
    data = json.load(f)

for c in data['crates']:
    if c['display_name'] == 'polaris':
        c.setdefault('source', {})
        c['source'].setdefault('include_dirs', [])
        c['source'].setdefault('exclude_dirs', [])
        c['source']['include_dirs'].extend([
            '$EXTTREE',
            '$KDIR/rust',
        ])
        c['env']['OBJTREE'] = '$KDIR'
        # Remove duplicates
        c['source']['include_dirs'] = list(set(c['source']['include_dirs']))
        c['source']['exclude_dirs'] = list(set(c['source']['exclude_dirs']))
        break

with open('$PROJECT_ROOT/rust-project.json', 'w') as f:
    json.dump(data, f, sort_keys=True, indent=4)

print('  Patched polaris crate with OBJTREE and source includes')
"

echo "Written: $PROJECT_ROOT/rust-project.json"
echo "Done. rust-analyzer will auto-detect this file."
