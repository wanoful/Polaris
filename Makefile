# POLARIS Top-Level Makefile
#
# Variables (override on command line or via environment):
#   KDIR          - Kernel source/build tree
#   NVIDIA_KO_DIR - Patched open-gpu-kernel-modules tree or kernel-open dir
#   CC            - C compiler (default: cc)
#   POLARIS_RUSTC - Path to rustc for kernel builds (auto-detected on Arch)
#
# Examples:
#   make KDIR=/path/to/kernel          # specify kernel tree
#   make kernel NVIDIA_KO_DIR=third_party/open-gpu-kernel-modules
#   make kernel CC=clang               # build kernel module with clang
#   make setup-pacman-rustc            # one-time setup on Arch

KDIR          ?= /lib/modules/$(shell uname -r)/build
CC            ?= cc
EXTERNAL_NVIDIA_KO_DIR := /home/wano/workspace/open-gpu-kernel-modules
ifneq (,$(wildcard $(EXTERNAL_NVIDIA_KO_DIR)/kernel-open/Module.symvers))
    DEFAULT_NVIDIA_KO_DIR := $(EXTERNAL_NVIDIA_KO_DIR)
else
    DEFAULT_NVIDIA_KO_DIR := $(abspath $(CURDIR)/third_party/open-gpu-kernel-modules)
endif

NVIDIA_KO_DIR ?= $(DEFAULT_NVIDIA_KO_DIR)

ifneq (,$(wildcard $(NVIDIA_KO_DIR)/kernel-open/Module.symvers))
    POLARIS_EXTRA_SYMBOLS ?= $(abspath $(NVIDIA_KO_DIR)/kernel-open/Module.symvers)
else ifneq (,$(wildcard $(NVIDIA_KO_DIR)/Module.symvers))
    POLARIS_EXTRA_SYMBOLS ?= $(abspath $(NVIDIA_KO_DIR)/Module.symvers)
endif

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

.PHONY: all kernel userspace clean help rust-analyzer rust-toolchain setup-pacman-rustc llama-e2e m6-daemon-rm-soak m6-module-unload-stress

help:
	@echo "POLARIS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make kernel              Build the kernel module (polaris.ko)"
	@echo "  make userspace           Build all Rust userspace programs"
	@echo "  make llama-e2e           Run the strict root/GPU llama.cpp shim regression"
	@echo "  make m6-daemon-rm-soak   Run the daemon-backed RM soak with patched UVM"
	@echo "  make m6-module-unload-stress"
	@echo "                            Run module unload/reload stress with patched UVM"
	@echo "  make all                 Build everything"
	@echo "  make clean               Clean all build artifacts"
	@echo "  make setup-pacman-rustc  Download pacman rustc locally (Arch Linux)"
	@echo "  make rust-analyzer       Generate rust-project.json for IDE support"
	@echo "  make rust-toolchain      Generate rust-toolchain.toml from kernel config"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=<path>              Kernel source/build tree"
	@echo "  NVIDIA_KO_DIR=<path>     Patched open-gpu-kernel-modules tree for UVM symbols"
	@echo "                            (default: third_party/open-gpu-kernel-modules)"
	@echo "  POLARIS_RUSTC=<path>     rustc for kernel builds (auto-detected)"
	@echo "  CC=<compiler>            C compiler (default: cc)"

all: kernel userspace

kernel:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel modules CC=$(CC) RUSTC="$(POLARIS_RUSTC)" KBUILD_EXTRA_SYMBOLS="$(POLARIS_EXTRA_SYMBOLS)"

userspace:
	cargo build --release

llama-e2e:
	$(MAKE) kernel NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	$(MAKE) -C libpolaris-shim all tests NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	cargo build -p polarisd
	sudo env \
		POLARIS_LLAMA_LOAD_MODULE=1 \
		POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1 \
		POLARIS_LLAMA_RUN_DYNAMIC_WINDOW_PROBE=1 \
		NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)" \
		LLAMA_CPP_DIR="$${LLAMA_CPP_DIR:-/home/wano/workspace/llama.cpp}" \
		bash tests/llama_cpp/run_llama_shim_e2e.sh

m6-daemon-rm-soak:
	$(MAKE) kernel NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	cargo build -p polarisd
	$(MAKE) -C tests/m2 NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	sudo env NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)" tests/m2/run_daemon_rm_soak.sh

m6-module-unload-stress:
	$(MAKE) kernel NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	cargo build -p polarisd
	$(MAKE) -C tests/m2 NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)"
	sudo env NVIDIA_KO_DIR="$(NVIDIA_KO_DIR)" tests/m2/run_module_unload_stress.sh

clean:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel clean 2>/dev/null || true
	cargo clean

setup-pacman-rustc:
	@bash ./scripts/setup-pacman-rustc.sh --rustup-link

rust-analyzer:
	@bash ./scripts/gen-rust-project.sh

rust-toolchain:
	@bash ./scripts/gen-rust-toolchain.sh $(KDIR)
