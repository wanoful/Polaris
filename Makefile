# POLARIS Top-Level Makefile
#
# Variables (override on command line or via environment):
#   KDIR     - Kernel source/build tree
#   CC       - C compiler (default: clang, since WSL2 kernel is clang-built)
#
# Examples:
#   make KDIR=/path/to/kernel          # specify kernel tree
#   make kernel CC=clang               # build kernel module with clang

KDIR ?= /lib/modules/$(shell uname -r)/build
CC   ?= cc

.PHONY: all kernel userspace clean help rust-analyzer

help:
	@echo "POLARIS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make kernel           Build the kernel module (polaris.ko)"
	@echo "  make userspace        Build all Rust userspace programs"
	@echo "  make all              Build everything"
	@echo "  make clean            Clean all build artifacts"
	@echo "  make rust-analyzer    Generate rust-project.json for IDE support"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=<path>           Kernel source/build tree (default: /lib/modules/\$$(uname -r)/build)"
	@echo "  CC=<compiler>         C compiler (default: clang)"

all: kernel userspace

kernel:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel modules CC=$(CC)

userspace:
	cargo build --release

clean:
	$(MAKE) -C $(KDIR) M=$(PWD)/kernel clean 2>/dev/null || true
	cargo clean

rust-analyzer:
	@bash ./benchmarks/scripts/gen-rust-project.sh
