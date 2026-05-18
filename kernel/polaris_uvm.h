/* SPDX-License-Identifier: GPL-2.0 */
#ifndef POLARIS_UVM_H
#define POLARIS_UVM_H

#include <linux/types.h>

enum polaris_uvm_fault_result {
    POLARIS_UVM_FAULT_NOT_MINE = 0,
    POLARIS_UVM_FAULT_HANDLED = 1,
    POLARIS_UVM_FAULT_ERROR = -1,
};

/*
 * Called from the NVIDIA UVM replayable fault bottom half after UVM has parsed
 * a virtual fault packet, and before stock UVM managed-memory servicing.
 *
 * POLARIS_UVM_FAULT_NOT_MINE:
 *   The GPU is unknown to POLARIS, no POLARIS VA range is registered for it, or
 *   fault_address is outside that range. UVM must continue stock handling.
 *
 * POLARIS_UVM_FAULT_HANDLED:
 *   POLARIS queued the fault decision, waited for polarisd, and the block is
 *   resident. UVM may advance its fault buffer and replay the GPU work.
 *
 * POLARIS_UVM_FAULT_ERROR:
 *   The fault is inside a POLARIS VA range but could not be resolved, timed out,
 *   or hit an internal error. UVM should route this through its fatal/cancel
 *   replayable-fault path rather than falling through to stock handling.
 */
int polaris_uvm_handle_gpu_fault(u32 gpu_id, u64 fault_address, u32 access_type);

#endif /* POLARIS_UVM_H */
