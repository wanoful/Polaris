# libpolaris-shim Makefile
#
# Produces libpolaris-shim.so: a small LD_PRELOAD that intercepts the CUDA
# driver-API entry points POLARIS cares about and steers worker allocations
# into a fault-capable VA-space owned by UVM and serviced by polaris.ko.
#
# v4-M0 scaffolding: only cuInit is intercepted, and the interceptor is a
# pass-through that proves the LD_PRELOAD shape works end-to-end.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden -std=c11 \
           -D_GNU_SOURCE
LDFLAGS ?= -shared -Wl,--no-undefined -ldl

SRCDIR   := src
INCDIR   := include
BUILDDIR := build

SOURCES := $(SRCDIR)/shim.c $(SRCDIR)/cuda_loader.c $(SRCDIR)/device.c
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TARGET  := libpolaris-shim.so

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c -o $@ $<

clean:
	rm -rf $(BUILDDIR) $(TARGET)
