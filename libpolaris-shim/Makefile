# libpolaris-shim Makefile
#
# Produces libpolaris-shim.so: a small LD_PRELOAD that intercepts the CUDA
# driver-API entry points POLARIS cares about and steers worker allocations
# into a fault-capable VA-space owned by UVM and serviced by polaris.ko.
#
# v4-M2 scaffolding: cuInit is intercepted and can register a harness-created
# VA-space through environment variables. Allocator interception is still
# pending.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden -std=c11 \
           -D_GNU_SOURCE
LDFLAGS ?= -shared -Wl,--no-undefined -ldl

DEFAULT_NVIDIA_KO_DIR := $(abspath $(CURDIR)/../third_party/open-gpu-kernel-modules)
NVIDIA_KO_DIR ?= $(DEFAULT_NVIDIA_KO_DIR)
NVIDIA_KERNEL_OPEN := $(NVIDIA_KO_DIR)/kernel-open
NVIDIA_COMMON_INC := $(NVIDIA_KO_DIR)/src/common/sdk/nvidia/inc
NVIDIA_UNIX_INC := $(NVIDIA_KO_DIR)/src/nvidia/arch/nvalloc/unix/include

SRCDIR   := src
INCDIR   := include
BUILDDIR := build

SOURCES := $(SRCDIR)/shim.c $(SRCDIR)/cuda_loader.c $(SRCDIR)/device.c \
           $(SRCDIR)/uvm_external.c $(SRCDIR)/rm_uvm_bootstrap.c
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TARGET  := libpolaris-shim.so
TESTS   := $(BUILDDIR)/smoke $(BUILDDIR)/managed_alloc

.PHONY: all tests clean
all: $(TARGET)
tests: $(TESTS)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) \
		-I$(NVIDIA_KERNEL_OPEN)/common/inc \
		-I$(NVIDIA_KERNEL_OPEN)/nvidia-uvm \
		-I$(NVIDIA_COMMON_INC) \
		-I$(NVIDIA_UNIX_INC) \
		-c -o $@ $<

$(BUILDDIR)/%: tests/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -o $@ $< -ldl

clean:
	rm -rf $(BUILDDIR) $(TARGET)
