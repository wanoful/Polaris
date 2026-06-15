# SPDX-License-Identifier: GPL-2.0

CC ?= cc
DEFAULT_NVIDIA_KO_DIR := $(abspath $(CURDIR)/../../third_party/open-gpu-kernel-modules)

NVIDIA_KO_DIR ?= $(DEFAULT_NVIDIA_KO_DIR)
CUDA_HOME ?= /usr/local/cuda

NVIDIA_KERNEL_OPEN := $(NVIDIA_KO_DIR)/kernel-open
NVIDIA_COMMON_INC := $(NVIDIA_KO_DIR)/src/common/sdk/nvidia/inc
NVIDIA_UNIX_INC := $(NVIDIA_KO_DIR)/src/nvidia/arch/nvalloc/unix/include

CFLAGS ?= -O2 -g -Wall -Wextra -Wpedantic -std=c11
CPPFLAGS += \
	-I$(CUDA_HOME)/include \
	-I$(NVIDIA_KERNEL_OPEN)/common/inc \
	-I$(NVIDIA_KERNEL_OPEN)/nvidia-uvm \
	-I$(NVIDIA_COMMON_INC) \
	-I$(NVIDIA_UNIX_INC)
LDFLAGS += -pthread
LDLIBS += -lcuda

.PHONY: all clean module-unload-stress daemon-rm-soak

all: m2_static_block_setup

m2_static_block_setup: m2_static_block_setup.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

module-unload-stress: m2_static_block_setup
	./run_module_unload_stress.sh

daemon-rm-soak: m2_static_block_setup
	./run_daemon_rm_soak.sh

clean:
	rm -f m2_static_block_setup
