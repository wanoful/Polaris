# POLARIS Top-Level Makefile
#
# Variables (override on command line or via environment):
#   KDIR          - Kernel source/build tree
#   CC            - C compiler (default: cc)
#   POLARIS_RUSTC - Path to rustc for kernel builds (auto-detected on Arch)
#
# Examples:
#   make KDIR=/path/to/kernel          # specify kernel tree
#   make kernel CC=clang               # build kernel module with clang
#   make setup-pacman-rustc            # one-time setup on Arch

KDIR          ?= /lib/modules/$(shell uname -r)/build
CC            ?= cc

# --- auto-detect pacman rustc for kernel builds (Arch Linux) ------------------
# The Linux kernel's Rust-for-Linux requires the *exact* rustc build that
# compiled the kernel.  On Arch this is the pacman package, not rustup's
# build (even when version numbers match).  Run 'make setup-pacman-rustc'
# once to download and extract the pacman rustc locally.
PACMAN_RUSTC_LOCAL := $(CURDIR)/.tools/pacman-rustc/usr/bin/rustc
PACMAN_RUSTC_LIB   := $(CURDIR)/.tools/pacman-rustc/usr/lib
ifneq (,$(wildcard $(PACMAN_RUSTC_LOCAL)))
    # Arch's rustc is dynamically linked against librustc_driver-*.so,
    # which lives in usr/lib/ alongside the binary.  Prepend that path
    # to LD_LIBRARY_PATH so the linker can find it at runtime.
    POLARIS_RUSTC ?= env LD_LIBRARY_PATH="$(PACMAN_RUSTC_LIB):$$LD_LIBRARY_PATH" $(PACMAN_RUSTC_LOCAL)
endif

# Fallback: search PATH
POLARIS_RUSTC ?= $(shell command -v rustc 2>/dev/null || echo rustc)

.PHONY: all kernel userspace clean help rust-analyzer rust-toolchain setup-pacman-rustc

help:
	@echo "POLARIS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make kernel              Build the kernel module (polaris.ko)"
	@echo "  make userspace           Build all Rust userspace programs"
	@echo "  make all                 Build everything"
	@echo "  make clean               Clean all build artifacts"
	@echo "  make setup-pacman-rustc  Download pacman rustc locally (Arch Linux)"
	@echo "  make rust-analyzer       Generate rust-project.json for IDE support"
	@echo "  make rust-toolchain      Generate rust-toolchain.toml from kernel config"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=<path>              Kernel source/build tree"
	@echo "  POLARIS_RUSTC=<path>     rustc for kernel builds (auto-detected)"
	@echo "  CC=<compiler>            C compiler (default: cc)"

all: kernel userspace

kernel:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel modules CC=$(CC) RUSTC="$(POLARIS_RUSTC)"

userspace:
	cargo build --release

clean:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel clean 2>/dev/null || true
	cargo clean

setup-pacman-rustc:
	@bash ./scripts/setup-pacman-rustc.sh --rustup-link

rust-analyzer:
	@bash ./benchmarks/scripts/gen-rust-project.sh

rust-toolchain:
	@bash ./benchmarks/scripts/gen-rust-toolchain.sh $(KDIR)
