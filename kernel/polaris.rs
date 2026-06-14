// SPDX-License-Identifier: GPL-2.0

// POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems
//
// This kernel module provides OS-level paged KV Cache management for LLM inference.
// It maintains the authoritative block table, manages sessions, and issues GPU memory
// management decisions to the userspace daemon (polarisd) via an ioctl-based
// decision protocol.
//
// All state is global: every open("/dev/polaris") shares the same block table,
// session table, GPU registry, and decision queue. This is the fundamental
// value of the kernel module — cross-process visibility.

mod polaris_types;
mod polaris_policy;

use core::pin::Pin;
use core::ffi::c_int;

use kernel::{
    bindings,
    c_str,
    device::Device,
    fs::File,
    ioctl::_IOC_SIZE,
    miscdevice::{MiscDevice, MiscDeviceOptions, MiscDeviceRegistration},
    prelude::*,
    sync::{
        aref::ARef,
        atomic::{Acquire, Atomic, Relaxed, Release},
    },
    uaccess::UserSlice,
};

use polaris_types::*;

#[derive(PartialEq)]
enum PolarisUvmFaultResult {
    NotMine,
    Handled,
    Error,
}

#[repr(C)]
struct UvmPolarisOps {
    owner: *mut bindings::module,
    handle_gpu_fault: unsafe extern "C" fn(u32, u64, u64, u64, u64, u32) -> c_int,
}

const UVM_POLARIS_FAULT_NOT_MINE: c_int = 0;
const UVM_POLARIS_FAULT_HANDLED: c_int = 1;
const UVM_POLARIS_FAULT_ERROR: c_int = -1;
const POLARIS_MAX_FAST_VASPACES: usize = 16;
const POLARIS_MAX_FAST_STATIC_BLOCKS: usize = 16;

static UVM_POLARIS_OPS: UvmPolarisOps = UvmPolarisOps {
    owner: core::ptr::addr_of_mut!(bindings::__this_module),
    handle_gpu_fault: polaris_uvm_handle_gpu_fault,
};

unsafe impl Sync for UvmPolarisOps {}

struct PolarisFastVaSpaceSlot {
    gpu_id: Atomic<u32>,
    rm_client_token: Atomic<u64>,
    va_space_token: Atomic<u64>,
    managed_base: Atomic<u64>,
    managed_length: Atomic<u64>,
}

impl PolarisFastVaSpaceSlot {
    const fn new() -> Self {
        Self {
            gpu_id: Atomic::new(0),
            rm_client_token: Atomic::new(0),
            va_space_token: Atomic::new(0),
            managed_base: Atomic::new(0),
            managed_length: Atomic::new(0),
        }
    }

    fn clear(&self) {
        self.va_space_token.store(0, Release);
        self.gpu_id.store(0, Relaxed);
        self.rm_client_token.store(0, Relaxed);
        self.managed_base.store(0, Relaxed);
        self.managed_length.store(0, Relaxed);
    }
}

static POLARIS_FAST_VASPACES: [PolarisFastVaSpaceSlot; POLARIS_MAX_FAST_VASPACES] =
    [const { PolarisFastVaSpaceSlot::new() }; POLARIS_MAX_FAST_VASPACES];
static POLARIS_UVM_FAULT_HOOK_CALLS: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_FAULT_FAST_MATCHES: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_FAULT_FAST_MISSES: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_FAULT_UNSERVICEABLE_MATCHES: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_FAULT_HANDLED: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_FAULT_ERRORS: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_LAST_GPU_ID: Atomic<u32> = Atomic::new(0);
static POLARIS_UVM_LAST_RM_CLIENT_TOKEN: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_LAST_VA_SPACE_TOKEN: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_LAST_GPU_VA_SPACE_PTR: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_LAST_FAULT_ADDRESS: Atomic<u64> = Atomic::new(0);
static POLARIS_UVM_LAST_ACCESS_TYPE: Atomic<u32> = Atomic::new(0);
static POLARIS_UVM_LAST_MAP_RET: Atomic<i32> = Atomic::new(0);
static POLARIS_UVM_LAST_ENSURE_RET: Atomic<i32> = Atomic::new(0);
static POLARIS_UVM_LAST_RESULT: Atomic<i32> = Atomic::new(0);

struct PolarisFastStaticBlockSlot {
    gpu_id: Atomic<u32>,
    rm_client_token: Atomic<u64>,
    rm_control_fd: Atomic<i32>,
    h_client: Atomic<u32>,
    h_memory: Atomic<u32>,
    va_space_token: Atomic<u64>,
    base: Atomic<u64>,
    length: Atomic<u64>,
    offset: Atomic<u64>,
    last_gpu_va_space_ptr: Atomic<u64>,
}

impl PolarisFastStaticBlockSlot {
    const fn new() -> Self {
        Self {
            gpu_id: Atomic::new(0),
            rm_client_token: Atomic::new(0),
            rm_control_fd: Atomic::new(0),
            h_client: Atomic::new(0),
            h_memory: Atomic::new(0),
            va_space_token: Atomic::new(0),
            base: Atomic::new(0),
            length: Atomic::new(0),
            offset: Atomic::new(0),
            last_gpu_va_space_ptr: Atomic::new(0),
        }
    }

    fn clear(&self) {
        self.va_space_token.store(0, Release);
        self.gpu_id.store(0, Relaxed);
        self.rm_client_token.store(0, Relaxed);
        self.rm_control_fd.store(0, Relaxed);
        self.h_client.store(0, Relaxed);
        self.h_memory.store(0, Relaxed);
        self.base.store(0, Relaxed);
        self.length.store(0, Relaxed);
        self.offset.store(0, Relaxed);
        self.last_gpu_va_space_ptr.store(0, Relaxed);
    }
}

static POLARIS_FAST_STATIC_BLOCKS: [PolarisFastStaticBlockSlot; POLARIS_MAX_FAST_STATIC_BLOCKS] =
    [const { PolarisFastStaticBlockSlot::new() }; POLARIS_MAX_FAST_STATIC_BLOCKS];

fn polaris_fast_vaspace_register(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    managed_base: u64,
    managed_length: u64,
) -> Result<()> {
    if va_space_token == 0 || managed_length == 0 {
        return Err(EINVAL);
    }

    for slot in &POLARIS_FAST_VASPACES {
        let token = slot.va_space_token.load(Acquire);
        if token == va_space_token
            && slot.gpu_id.load(Relaxed) == gpu_id
            && slot.rm_client_token.load(Relaxed) == rm_client_token
        {
            slot.managed_base.store(managed_base, Relaxed);
            slot.managed_length.store(managed_length, Relaxed);
            return Ok(());
        }
    }

    for slot in &POLARIS_FAST_VASPACES {
        if slot.va_space_token.load(Acquire) == 0 {
            slot.gpu_id.store(gpu_id, Relaxed);
            slot.rm_client_token.store(rm_client_token, Relaxed);
            slot.managed_base.store(managed_base, Relaxed);
            slot.managed_length.store(managed_length, Relaxed);
            slot.va_space_token.store(va_space_token, Release);
            return Ok(());
        }
    }

    Err(ENOMEM)
}

fn polaris_fast_vaspace_unregister(gpu_id: u32, rm_client_token: u64, va_space_token: u64) {
    if va_space_token == 0 {
        return;
    }

    for slot in &POLARIS_FAST_VASPACES {
        if slot.va_space_token.load(Acquire) == va_space_token
            && slot.gpu_id.load(Relaxed) == gpu_id
            && slot.rm_client_token.load(Relaxed) == rm_client_token
        {
            slot.clear();
            return;
        }
    }
}

fn polaris_fast_static_block_register(block: &PolarisStaticBlock) -> Result<()> {
    if block.va_space_token == 0 || block.length == 0 || block.h_memory == 0 {
        return Err(EINVAL);
    }

    for slot in &POLARIS_FAST_STATIC_BLOCKS {
        let token = slot.va_space_token.load(Acquire);
        if token == block.va_space_token
            && slot.gpu_id.load(Relaxed) == block.gpu_id
            && slot.rm_client_token.load(Relaxed) == block.rm_client_token
            && slot.base.load(Relaxed) == block.base
        {
            slot.length.store(block.length, Relaxed);
            slot.offset.store(block.offset, Relaxed);
            slot.rm_control_fd.store(block.rm_control_fd, Relaxed);
            slot.h_client.store(block.h_client, Relaxed);
            slot.h_memory.store(block.h_memory, Relaxed);
            return Ok(());
        }
    }

    for slot in &POLARIS_FAST_STATIC_BLOCKS {
        if slot.va_space_token.load(Acquire) == 0 {
            slot.gpu_id.store(block.gpu_id, Relaxed);
            slot.rm_client_token.store(block.rm_client_token, Relaxed);
            slot.base.store(block.base, Relaxed);
            slot.length.store(block.length, Relaxed);
            slot.offset.store(block.offset, Relaxed);
            slot.rm_control_fd.store(block.rm_control_fd, Relaxed);
            slot.h_client.store(block.h_client, Relaxed);
            slot.h_memory.store(block.h_memory, Relaxed);
            slot.va_space_token.store(block.va_space_token, Release);
            return Ok(());
        }
    }

    Err(ENOMEM)
}

fn polaris_fast_static_blocks_unregister_va_space(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
) {
    if va_space_token == 0 {
        return;
    }

    for slot in &POLARIS_FAST_STATIC_BLOCKS {
        if slot.va_space_token.load(Acquire) == va_space_token
            && slot.gpu_id.load(Relaxed) == gpu_id
            && slot.rm_client_token.load(Relaxed) == rm_client_token
        {
            slot.clear();
        }
    }
}

fn polaris_fast_static_block_unregister(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    base: u64,
) {
    if va_space_token == 0 {
        return;
    }

    for slot in &POLARIS_FAST_STATIC_BLOCKS {
        if slot.va_space_token.load(Acquire) == va_space_token
            && slot.gpu_id.load(Relaxed) == gpu_id
            && slot.rm_client_token.load(Relaxed) == rm_client_token
            && slot.base.load(Relaxed) == base
        {
            slot.clear();
            return;
        }
    }
}

fn polaris_fast_vaspace_clear_all() {
    for slot in &POLARIS_FAST_VASPACES {
        slot.clear();
    }
    for slot in &POLARIS_FAST_STATIC_BLOCKS {
        slot.clear();
    }
}

unsafe extern "C" {
    fn uvm_polaris_register_hook(ops: *const UvmPolarisOps) -> c_int;
    fn uvm_polaris_unregister_hook(ops: *const UvmPolarisOps);
    fn uvm_polaris_map_external_allocation(
        gpu_va_space_ptr: u64,
        base: u64,
        length: u64,
        offset: u64,
        rm_control_fd: i32,
        h_client: u32,
        h_memory: u32,
    ) -> c_int;
    fn uvm_polaris_ensure_external_range(
        gpu_va_space_ptr: u64,
        base: u64,
        length: u64,
    ) -> c_int;
    fn uvm_polaris_unmap_external_allocation(
        gpu_va_space_ptr: u64,
        base: u64,
        length: u64,
    ) -> c_int;
    fn uvm_polaris_probe_external_allocation(
        gpu_va_space_ptr: u64,
        offset: u64,
        length: u64,
        rm_control_fd: i32,
        h_client: u32,
        h_memory: u32,
        page_size_out: *mut u64,
        phys_addr_count_out: *mut u64,
        first_phys_addr_out: *mut u64,
        last_phys_addr_out: *mut u64,
        contiguous_out: *mut u64,
        sysmem_out: *mut u64,
        egm_out: *mut u64,
        fabricmem_out: *mut u64,
    ) -> c_int;
    fn uvm_polaris_probe_external_copy(
        gpu_va_space_ptr: u64,
        offset: u64,
        length: u64,
        rm_control_fd: i32,
        h_client: u32,
        h_memory: u32,
        pattern_seed: u64,
        page_size_out: *mut u64,
        phys_addr_count_out: *mut u64,
        first_phys_addr_out: *mut u64,
        last_phys_addr_out: *mut u64,
        flags_out: *mut u64,
        bytes_checked_out: *mut u64,
        first_mismatch_offset_out: *mut u64,
        expected_byte_out: *mut u64,
        actual_byte_out: *mut u64,
    ) -> c_int;
    fn uvm_polaris_copy_external_allocation(
        gpu_va_space_ptr: u64,
        offset: u64,
        length: u64,
        rm_control_fd: i32,
        h_client: u32,
        h_memory: u32,
        user_cpu_addr: u64,
        direction: u32,
        page_size_out: *mut u64,
        phys_addr_count_out: *mut u64,
        first_phys_addr_out: *mut u64,
        last_phys_addr_out: *mut u64,
        flags_out: *mut u64,
        bytes_copied_out: *mut u64,
    ) -> c_int;
}

unsafe extern "C" fn polaris_uvm_handle_gpu_fault(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    gpu_va_space_ptr: u64,
    fault_address: u64,
    access_type: u32,
) -> c_int {
    POLARIS_UVM_FAULT_HOOK_CALLS.fetch_add(1, Relaxed);
    POLARIS_UVM_LAST_GPU_ID.store(gpu_id, Relaxed);
    POLARIS_UVM_LAST_RM_CLIENT_TOKEN.store(rm_client_token, Relaxed);
    POLARIS_UVM_LAST_VA_SPACE_TOKEN.store(va_space_token, Relaxed);
    POLARIS_UVM_LAST_GPU_VA_SPACE_PTR.store(gpu_va_space_ptr, Relaxed);
    POLARIS_UVM_LAST_FAULT_ADDRESS.store(fault_address, Relaxed);
    POLARIS_UVM_LAST_ACCESS_TYPE.store(access_type, Relaxed);

    for slot in &POLARIS_FAST_VASPACES {
        let token = slot.va_space_token.load(Acquire);
        if token == 0 {
            continue;
        }
        if token != va_space_token {
            continue;
        }
        if slot.gpu_id.load(Relaxed) != gpu_id {
            continue;
        }
        if slot.rm_client_token.load(Relaxed) != rm_client_token {
            continue;
        }
        let base = slot.managed_base.load(Relaxed);
        let length = slot.managed_length.load(Relaxed);
        let end = base.saturating_add(length);
        if fault_address < base || fault_address >= end {
            continue;
        }

        POLARIS_UVM_FAULT_FAST_MATCHES.fetch_add(1, Relaxed);

        for block in &POLARIS_FAST_STATIC_BLOCKS {
            let block_token = block.va_space_token.load(Acquire);
            if block_token == 0 {
                continue;
            }
            if block_token != va_space_token {
                continue;
            }
            if block.gpu_id.load(Relaxed) != gpu_id {
                continue;
            }
            if block.rm_client_token.load(Relaxed) != rm_client_token {
                continue;
            }
            let block_base = block.base.load(Relaxed);
            let block_length = block.length.load(Relaxed);
            let block_end = block_base.saturating_add(block_length);
            if fault_address < block_base || fault_address >= block_end {
                continue;
            }

            let mapping = PolarisFaultMapping {
                base: block_base,
                length: block_length,
                offset: block.offset.load(Relaxed),
                rm_control_fd: block.rm_control_fd.load(Relaxed),
                h_client: block.h_client.load(Relaxed),
                h_memory: block.h_memory.load(Relaxed),
            };
            match polaris_map_fault_mapping(gpu_va_space_ptr, &mapping) {
                Ok(()) => {
                    block.last_gpu_va_space_ptr.store(gpu_va_space_ptr, Release);
                    polaris_note_block_mapping_fault(
                        gpu_id,
                        fault_address,
                        gpu_va_space_ptr,
                    );
                    POLARIS_UVM_FAULT_HANDLED.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_HANDLED, Relaxed);
                    return UVM_POLARIS_FAULT_HANDLED;
                }
                Err(()) => {
                    POLARIS_UVM_FAULT_ERRORS.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_ERROR, Relaxed);
                    return UVM_POLARIS_FAULT_ERROR;
                }
            }
        }

        if let Some(mapping) = polaris_find_logical_fault_mapping(
            gpu_id,
            rm_client_token,
            va_space_token,
            fault_address,
        ) {
            match polaris_map_fault_mapping(gpu_va_space_ptr, &mapping) {
                Ok(()) => {
                    polaris_note_block_mapping_fault(
                        gpu_id,
                        fault_address,
                        gpu_va_space_ptr,
                    );
                    POLARIS_UVM_FAULT_HANDLED.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_HANDLED, Relaxed);
                    return UVM_POLARIS_FAULT_HANDLED;
                }
                Err(()) => {
                    POLARIS_UVM_FAULT_ERRORS.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_ERROR, Relaxed);
                    return UVM_POLARIS_FAULT_ERROR;
                }
            }
        }

        if polaris_can_materialize_logical_fault_mapping(
            gpu_id,
            rm_client_token,
            va_space_token,
            fault_address,
        ) {
            match polaris_resolve_gpu_fault(
                gpu_id,
                rm_client_token,
                va_space_token,
                fault_address,
                access_type,
            ) {
                Ok(PolarisUvmFaultResult::Handled) => {
                    if let Some(mapping) = polaris_find_logical_fault_mapping(
                        gpu_id,
                        rm_client_token,
                        va_space_token,
                        fault_address,
                    ) {
                        match polaris_map_fault_mapping(gpu_va_space_ptr, &mapping) {
                            Ok(()) => {
                                polaris_note_block_mapping_fault(
                                    gpu_id,
                                    fault_address,
                                    gpu_va_space_ptr,
                                );
                                POLARIS_UVM_FAULT_HANDLED.fetch_add(1, Relaxed);
                                POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_HANDLED, Relaxed);
                                return UVM_POLARIS_FAULT_HANDLED;
                            }
                            Err(()) => {
                                POLARIS_UVM_FAULT_ERRORS.fetch_add(1, Relaxed);
                                POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_ERROR, Relaxed);
                                return UVM_POLARIS_FAULT_ERROR;
                            }
                        }
                    }
                    POLARIS_UVM_FAULT_UNSERVICEABLE_MATCHES.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_NOT_MINE, Relaxed);
                    return UVM_POLARIS_FAULT_NOT_MINE;
                }
                Ok(PolarisUvmFaultResult::NotMine) => {
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_NOT_MINE, Relaxed);
                    return UVM_POLARIS_FAULT_NOT_MINE;
                }
                Ok(PolarisUvmFaultResult::Error) | Err(_) => {
                    POLARIS_UVM_FAULT_ERRORS.fetch_add(1, Relaxed);
                    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_ERROR, Relaxed);
                    return UVM_POLARIS_FAULT_ERROR;
                }
            }
        }

        POLARIS_UVM_FAULT_UNSERVICEABLE_MATCHES.fetch_add(1, Relaxed);
        POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_NOT_MINE, Relaxed);
        return UVM_POLARIS_FAULT_NOT_MINE;
    }

    if let Some(mapping) = polaris_find_single_observed_fault_mapping(
        gpu_id,
        fault_address,
    ) {
        match polaris_map_fault_mapping(gpu_va_space_ptr, &mapping) {
            Ok(()) => {
                polaris_note_block_mapping_fault(
                    gpu_id,
                    fault_address,
                    gpu_va_space_ptr,
                );
                POLARIS_UVM_FAULT_HANDLED.fetch_add(1, Relaxed);
                POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_HANDLED, Relaxed);
                return UVM_POLARIS_FAULT_HANDLED;
            }
            Err(()) => {
                POLARIS_UVM_FAULT_ERRORS.fetch_add(1, Relaxed);
                POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_ERROR, Relaxed);
                return UVM_POLARIS_FAULT_ERROR;
            }
        }
    }

    POLARIS_UVM_FAULT_FAST_MISSES.fetch_add(1, Relaxed);
    POLARIS_UVM_LAST_RESULT.store(UVM_POLARIS_FAULT_NOT_MINE, Relaxed);
    UVM_POLARIS_FAULT_NOT_MINE
}

fn polaris_map_fault_mapping(gpu_va_space_ptr: u64, mapping: &PolarisFaultMapping) -> Result<(), ()> {
    let ret = unsafe {
        uvm_polaris_map_external_allocation(
            gpu_va_space_ptr,
            mapping.base,
            mapping.length,
            mapping.offset,
            mapping.rm_control_fd,
            mapping.h_client,
            mapping.h_memory,
        )
    };
    POLARIS_UVM_LAST_MAP_RET.store(ret, Relaxed);
    if ret == 0 {
        return Ok(());
    }

    if ret == bindings::EFAULT as c_int {
        let range_ret = unsafe {
            uvm_polaris_ensure_external_range(
                gpu_va_space_ptr,
                mapping.base,
                mapping.length,
            )
        };
        POLARIS_UVM_LAST_ENSURE_RET.store(range_ret, Relaxed);
        if range_ret == 0 {
            let retry_ret = unsafe {
                uvm_polaris_map_external_allocation(
                    gpu_va_space_ptr,
                    mapping.base,
                    mapping.length,
                    mapping.offset,
                    mapping.rm_control_fd,
                    mapping.h_client,
                    mapping.h_memory,
                )
            };
            POLARIS_UVM_LAST_MAP_RET.store(retry_ret, Relaxed);
            if retry_ret == 0 {
                return Ok(());
            }
        }
    }

    Err(())
}

#[allow(dead_code)]
fn polaris_uvm_map_external_allocation(
    gpu_va_space_ptr: u64,
    base: u64,
    length: u64,
    offset: u64,
    rm_control_fd: i32,
    h_client: u32,
    h_memory: u32,
) -> Result<()> {
    let ret = unsafe {
        uvm_polaris_map_external_allocation(
            gpu_va_space_ptr,
            base,
            length,
            offset,
            rm_control_fd,
            h_client,
            h_memory,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(Error::from_errno(-ret))
    }
}

fn polaris_uvm_unmap_external_allocation(
    gpu_va_space_ptr: u64,
    base: u64,
    length: u64,
) -> Result<()> {
    let ret = unsafe { uvm_polaris_unmap_external_allocation(gpu_va_space_ptr, base, length) };
    if ret == 0 {
        Ok(())
    } else {
        Err(Error::from_errno(-ret))
    }
}

fn polaris_note_block_mapping_fault(
    gpu_id: u32,
    fault_address: u64,
    gpu_va_space_ptr: u64,
) {
    if gpu_va_space_ptr == 0 {
        return;
    }

    let mut guard = POLARIS_STATE.lock();
    let Some(inner) = guard.as_mut() else {
        return;
    };

    for mapping in inner.block_mappings.iter_mut() {
        if mapping.gpu_id == gpu_id
            && fault_address >= mapping.base
            && fault_address < mapping.base.saturating_add(mapping.length)
        {
            mapping.last_gpu_va_space_ptr = gpu_va_space_ptr;
        }
    }
}

struct PolarisMappingUnmap {
    block_id: u64,
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    gpu_va_space_ptr: u64,
    base: u64,
    length: u64,
}

#[derive(Clone, Copy)]
struct PolarisRmPhysProbeTarget {
    block_id: u64,
    gpu_id: u32,
    gpu_va_space_ptr: u64,
    rm_control_fd: i32,
    h_client: u32,
    h_memory: u32,
    length: u64,
}

#[derive(Clone, Copy)]
struct PolarisFaultMapping {
    base: u64,
    length: u64,
    offset: u64,
    rm_control_fd: i32,
    h_client: u32,
    h_memory: u32,
}

#[derive(Clone, Copy)]
struct PolarisCompletedRmBacking {
    rm_control_fd: i32,
    h_client: u32,
    h_memory: u32,
    length: u64,
}

struct PolarisStaticBlockKey {
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    base: u64,
}

fn polaris_find_logical_fault_mapping(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    fault_address: u64,
) -> Option<PolarisFaultMapping> {
    let guard = POLARIS_STATE.lock();
    let inner = guard.as_ref()?;

    for mapping in &inner.block_mappings {
        if mapping.gpu_id != gpu_id
            || mapping.rm_client_token != rm_client_token
            || mapping.va_space_token != va_space_token
            || fault_address < mapping.base
            || fault_address >= mapping.base.saturating_add(mapping.length)
        {
            continue;
        }

        let block = inner.blocks.iter().find(|b| b.block_id == mapping.block_id)?;
        if block.home_gpu != gpu_id
            || block.rm_h_memory == 0
            || block.rm_h_client == 0
            || block.rm_backing_length < mapping.length
        {
            continue;
        }

        return Some(PolarisFaultMapping {
            base: mapping.base,
            length: mapping.length,
            offset: block.rm_backing_offset,
            rm_control_fd: block.rm_control_fd,
            h_client: block.rm_h_client,
            h_memory: block.rm_h_memory,
        });
    }

    None
}

fn polaris_can_materialize_logical_fault_mapping(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    fault_address: u64,
) -> bool {
    let guard = POLARIS_STATE.lock();
    let Some(inner) = guard.as_ref() else {
        return false;
    };

    if inner.daemon_attached == 0 {
        return false;
    }

    inner.block_mappings.iter().any(|mapping| {
        if mapping.gpu_id != gpu_id
            || mapping.rm_client_token != rm_client_token
            || mapping.va_space_token != va_space_token
            || fault_address < mapping.base
            || fault_address >= mapping.base.saturating_add(mapping.length)
        {
            return false;
        }

        inner.blocks.iter().any(|block| {
            block.block_id == mapping.block_id
                && matches!(
                    block.state,
                    PolarisBlockState::Unmapped
                        | PolarisBlockState::CpuOffloaded
                        | PolarisBlockState::CowPending
                )
        })
    })
}

fn polaris_find_single_observed_fault_mapping(
    gpu_id: u32,
    fault_address: u64,
) -> Option<PolarisFaultMapping> {
    let guard = POLARIS_STATE.lock();
    let inner = guard.as_ref()?;
    let mut found: Option<PolarisFaultMapping> = None;

    for mapping in &inner.block_mappings {
        if mapping.gpu_id != gpu_id
            || fault_address < mapping.base
            || fault_address >= mapping.base.saturating_add(mapping.length)
        {
            continue;
        }

        let Some(block) = inner.blocks.iter().find(|b| b.block_id == mapping.block_id) else {
            continue;
        };
        if block.home_gpu != gpu_id
            || block.rm_h_memory == 0
            || block.rm_h_client == 0
            || block.rm_backing_length < mapping.length
        {
            continue;
        }

        let candidate = PolarisFaultMapping {
            base: mapping.base,
            length: mapping.length,
            offset: block.rm_backing_offset,
            rm_control_fd: block.rm_control_fd,
            h_client: block.rm_h_client,
            h_memory: block.rm_h_memory,
        };

        if found.is_some() {
            return None;
        }
        found = Some(candidate);
    }

    found
}

fn polaris_forget_static_blocks_for_block(inner: &mut PolarisInner, block_id: u64) -> Result<()> {
    let mut to_forget: KVec<PolarisStaticBlockKey> = KVec::new();

    for mapping in &inner.block_mappings {
        if mapping.block_id == block_id {
            to_forget.push(
                PolarisStaticBlockKey {
                    gpu_id: mapping.gpu_id,
                    rm_client_token: mapping.rm_client_token,
                    va_space_token: mapping.va_space_token,
                    base: mapping.base,
                },
                GFP_KERNEL,
            )?;
        }
    }

    for key in &to_forget {
        inner.static_blocks.retain(|b| {
            !(b.gpu_id == key.gpu_id
                && b.rm_client_token == key.rm_client_token
                && b.va_space_token == key.va_space_token
                && b.base == key.base)
        });
        polaris_fast_static_block_unregister(
            key.gpu_id,
            key.rm_client_token,
            key.va_space_token,
            key.base,
        );
    }

    Ok(())
}

fn polaris_clear_block_rm_backing(block: &mut PolarisBlock) {
    block.rm_control_fd = 0;
    block.rm_h_client = 0;
    block.rm_h_memory = 0;
    block.rm_backing_length = 0;
    block.rm_backing_offset = 0;
}

fn polaris_block_has_rm_backing(block: &PolarisBlock) -> bool {
    block.rm_h_client != 0 && block.rm_h_memory != 0 && block.rm_backing_length >= block.size_bytes
}

fn polaris_block_needs_free_decision(block: &PolarisBlock) -> bool {
    match block.state {
        PolarisBlockState::Resident => block.gpu_phys_handle != 0 || polaris_block_has_rm_backing(block),
        PolarisBlockState::CpuOffloaded => block.cpu_buf_addr != 0,
        _ => false,
    }
}

fn polaris_account_direct_block_removal(
    inner: &mut PolarisInner,
    gpu_id: u32,
    state: PolarisBlockState,
    size_bytes: u64,
    had_cpu_buf: bool,
) {
    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
        if state == PolarisBlockState::Resident {
            gpu.used_bytes = gpu.used_bytes.saturating_sub(size_bytes);
        } else if had_cpu_buf {
            gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(size_bytes);
        }
    }
}

fn polaris_unsupported() -> Error {
    Error::from_errno(-(bindings::EOPNOTSUPP as i32))
}

fn polaris_complete_rm_backing(
    arg: &PolarisCompleteOperationArg,
) -> Result<Option<PolarisCompletedRmBacking>> {
    let has_backing = arg.rm_control_fd != 0
        || arg.rm_h_client != 0
        || arg.rm_h_memory != 0
        || arg.rm_backing_length != 0;
    if !has_backing {
        return Ok(None);
    }
    if arg.rm_h_client == 0 || arg.rm_h_memory == 0 || arg.rm_backing_length == 0 {
        return Err(EINVAL);
    }
    Ok(Some(PolarisCompletedRmBacking {
        rm_control_fd: arg.rm_control_fd,
        h_client: arg.rm_h_client,
        h_memory: arg.rm_h_memory,
        length: arg.rm_backing_length,
    }))
}

fn polaris_apply_completed_rm_backing(
    block: &mut PolarisBlock,
    backing: Option<PolarisCompletedRmBacking>,
) {
    if let Some(backing) = backing {
        block.rm_control_fd = backing.rm_control_fd;
        block.rm_h_client = backing.h_client;
        block.rm_h_memory = backing.h_memory;
        block.rm_backing_length = backing.length;
        block.rm_backing_offset = 0;
    } else {
        polaris_clear_block_rm_backing(block);
    }
}

fn polaris_snapshot_block_mappings(inner: &PolarisInner, block_id: u64) -> Result<KVec<PolarisMappingUnmap>> {
    let mut to_unmap: KVec<PolarisMappingUnmap> = KVec::new();
    for mapping in &inner.block_mappings {
        if mapping.block_id == block_id && mapping.last_gpu_va_space_ptr != 0 {
            to_unmap.push(
                PolarisMappingUnmap {
                    block_id: mapping.block_id,
                    gpu_id: mapping.gpu_id,
                    rm_client_token: mapping.rm_client_token,
                    va_space_token: mapping.va_space_token,
                    gpu_va_space_ptr: mapping.last_gpu_va_space_ptr,
                    base: mapping.base,
                    length: mapping.length,
                },
                GFP_KERNEL,
            )?;
        }
    }
    Ok(to_unmap)
}

fn polaris_snapshot_rm_phys_probe_target(
    inner: &PolarisInner,
    block_id: u64,
) -> Result<PolarisRmPhysProbeTarget> {
    let block = inner
        .blocks
        .iter()
        .find(|b| b.block_id == block_id)
        .ok_or(ENOENT)?;

    if block.state != PolarisBlockState::Resident || !polaris_block_has_rm_backing(block) {
        return Err(ENOENT);
    }

    let mut found_gpu_va_space_ptr = 0u64;
    for mapping in &inner.block_mappings {
        if mapping.block_id != block_id || mapping.last_gpu_va_space_ptr == 0 {
            continue;
        }
        if found_gpu_va_space_ptr != 0 && found_gpu_va_space_ptr != mapping.last_gpu_va_space_ptr {
            return Err(EBUSY);
        }
        found_gpu_va_space_ptr = mapping.last_gpu_va_space_ptr;
    }

    if found_gpu_va_space_ptr == 0 {
        return Err(ENOENT);
    }

    Ok(PolarisRmPhysProbeTarget {
        block_id: block.block_id,
        gpu_id: block.home_gpu,
        gpu_va_space_ptr: found_gpu_va_space_ptr,
        rm_control_fd: block.rm_control_fd,
        h_client: block.rm_h_client,
        h_memory: block.rm_h_memory,
        length: block.rm_backing_length,
    })
}

fn polaris_unmap_observed_block_mappings(block_id: u64) -> Result<u32> {
    if block_id == 0 {
        return Err(EINVAL);
    }

    let to_unmap = {
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        if !inner.blocks.iter().any(|b| b.block_id == block_id) {
            return Err(ENOENT);
        }

        polaris_snapshot_block_mappings(inner, block_id)?
    };

    if to_unmap.is_empty() {
        return Ok(0);
    }

    let mut unmapped = 0u32;
    for mapping in &to_unmap {
        polaris_uvm_unmap_external_allocation(
            mapping.gpu_va_space_ptr,
            mapping.base,
            mapping.length,
        )?;
        unmapped = unmapped.saturating_add(1);
    }

    {
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        for unmapped_mapping in &to_unmap {
            if let Some(mapping) = inner.block_mappings.iter_mut().find(|m| {
                m.block_id == unmapped_mapping.block_id
                    && m.gpu_id == unmapped_mapping.gpu_id
                    && m.rm_client_token == unmapped_mapping.rm_client_token
                    && m.va_space_token == unmapped_mapping.va_space_token
                    && m.base == unmapped_mapping.base
                    && m.length == unmapped_mapping.length
                    && m.last_gpu_va_space_ptr == unmapped_mapping.gpu_va_space_ptr
            }) {
                mapping.last_gpu_va_space_ptr = 0;
            }
        }
    }

    Ok(unmapped)
}

fn polaris_queue_offload_decision(inner: &mut PolarisInner, block_idx: usize) -> Result<u64> {
    if block_idx >= inner.blocks.len() {
        return Err(ENOENT);
    }
    if inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS {
        return Err(ENOMEM);
    }

    let decision_id = inner.next_decision_id;
    inner.next_decision_id += 1;

    let block = &inner.blocks[block_idx];
    if block.state != PolarisBlockState::Resident {
        return Err(ENOENT);
    }
    if polaris_block_has_rm_backing(block) && block.gpu_phys_handle == 0 {
        return Err(polaris_unsupported());
    }
    if block.gpu_phys_handle == 0 {
        return Err(ENOENT);
    }

    let decision = PolarisDecision {
        decision_id,
        fault_id: 0,
        generation: 0,
        op: PolarisDecisionOp::Offload as u32,
        gpu_id: block.home_gpu,
        block_id: block.block_id,
        session_id: block.session_id,
        src_handle: block.gpu_phys_handle,
        dst_handle: 0,
        src_vaddr: block.gpu_vaddr,
        dst_vaddr: 0,
        size_bytes: block.size_bytes,
        cpu_addr: 0,
        access_flags: 0,
        timeout_ms: 0,
        _reserved: [0u64; 4],
    };

    inner.pending_decisions.push(decision, GFP_KERNEL)?;

    let block = &mut inner.blocks[block_idx];
    block.state = PolarisBlockState::OffloadPending;
    block.pending_decision_id = decision_id;
    block.pending_fault_id = 0;
    block.pending_generation = 0;

    Ok(decision_id)
}

fn polaris_queue_free_decision(inner: &mut PolarisInner, block_idx: usize) -> Result<u64> {
    if block_idx >= inner.blocks.len() {
        return Err(ENOENT);
    }
    if inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS {
        return Err(ENOMEM);
    }

    let block = &inner.blocks[block_idx];
    if !polaris_block_needs_free_decision(block) {
        return Err(ENOENT);
    }

    let decision_id = inner.next_decision_id;
    inner.next_decision_id += 1;

    let decision = PolarisDecision {
        decision_id,
        fault_id: 0,
        generation: 0,
        op: PolarisDecisionOp::Free as u32,
        gpu_id: block.home_gpu,
        block_id: block.block_id,
        session_id: block.session_id,
        src_handle: block.gpu_phys_handle,
        dst_handle: 0,
        src_vaddr: block.gpu_vaddr,
        dst_vaddr: 0,
        size_bytes: block.size_bytes,
        cpu_addr: block.cpu_buf_addr,
        access_flags: 0,
        timeout_ms: 0,
        _reserved: [0u64; 4],
    };

    inner.pending_decisions.push(decision, GFP_KERNEL)?;

    let block = &mut inner.blocks[block_idx];
    block.state = PolarisBlockState::FreePending;
    block.pending_decision_id = decision_id;
    block.pending_fault_id = 0;
    block.pending_generation = 0;

    Ok(decision_id)
}

fn polaris_schedule_budget_victim(
    inner: &mut PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    needed_bytes: u64,
    protected_phys_handle: u64,
) -> Result<()> {
    if needed_bytes == 0 {
        return Ok(());
    }

    let (used_bytes, budget_bytes, cpu_pool_used, cpu_pool_total) = {
        let gpu = inner.gpus.iter().find(|g| g.gpu_id == target_gpu).ok_or(ENODEV)?;
        (
            gpu.used_bytes,
            gpu.budget_bytes,
            gpu.cpu_pool_used_bytes,
            gpu.cpu_pool_total_bytes,
        )
    };

    if budget_bytes == 0 || used_bytes.saturating_add(needed_bytes) <= budget_bytes {
        return Ok(());
    }

    let deficit = used_bytes.saturating_add(needed_bytes).saturating_sub(budget_bytes);
    let Some((victim_idx, _victim_id, victim_size, _victim_gpu)) =
        polaris_policy::select_victim(
            inner,
            requesting_session_id,
            target_gpu,
            protected_phys_handle,
        )
    else {
        return Err(ENOMEM);
    };

    // Keep the first production scheduler slice deliberately simple: one
    // policy-selected victim must satisfy the deficit. Multi-victim planning
    // belongs in the resident-set scheduler.
    if victim_size < deficit {
        return Err(ENOMEM);
    }
    if cpu_pool_used.saturating_add(victim_size) > cpu_pool_total {
        return Err(ENOMEM);
    }
    if inner.pending_decisions.len().saturating_add(2) > POLARIS_MAX_PENDING_DECISIONS {
        return Err(ENOMEM);
    }

    polaris_queue_offload_decision(inner, victim_idx)?;
    Ok(())
}

fn polaris_current_pid() -> i32 {
    // SAFETY: get_current() returns the current task for this CPU. Passing a
    // null namespace asks for the pid in the caller's active pid namespace.
    unsafe {
        bindings::__task_pid_nr_ns(
            bindings::get_current(),
            bindings::pid_type_PIDTYPE_PID,
            core::ptr::null_mut(),
        ) as i32
    }
}

// ─── Global shared state ────────────────────────────────────────────────────

pub(crate) struct PolarisInner {
    next_block_id: u64,
    next_session_id: u64,
    next_decision_id: u64,
    next_fault_id: u64,
    next_range_id: u64,
    fault_generation: u64,
    daemon_attached: u32,
    gpus: KVec<PolarisGpu>,
    blocks: KVec<PolarisBlock>,
    sessions: KVec<PolarisSession>,
    va_spaces: KVec<PolarisVaSpace>,
    static_blocks: KVec<PolarisStaticBlock>,
    block_mappings: KVec<PolarisBlockMapping>,
    pending_decisions: KVec<PolarisDecision>,
    pending_faults: KVec<PolarisFault>,
    eviction_policy: PolarisEvictionPolicy,
    offload_count: u64,
    reload_count: u64,
    total_evictions: u64,
    cow_break_count: u64,
    cow_copy_bytes: u64,
}

// Global state protected by a kernel mutex.  Wrapped in Option because
// KVec cannot be const-constructed; the real state is installed at module
// init time and all handlers unwrap it.
kernel::sync::global_lock! {
    // SAFETY: Initialized in module init before any /dev/polaris open.
    unsafe(uninit) static POLARIS_STATE: Mutex<Option<PolarisInner>> = None;
}

// Set to 1 when the module begins its exit path.  Used by PolarisDevice's
// PinnedDrop to skip module_put during forced unload (rmmod -f) — the
// kernel has already zeroed the refcount, so calling module_put again
// would trigger BUG().
static MODULE_EXITING: Atomic<u32> = Atomic::new(0);

fn polaris_resolve_gpu_fault(
    gpu_id: u32,
    rm_client_token: u64,
    va_space_token: u64,
    fault_address: u64,
    access_type: u32,
) -> Result<PolarisUvmFaultResult> {
    let mut guard = POLARIS_STATE.lock();
    let inner = guard.as_mut().ok_or(ENODEV)?;

    if va_space_token != 0 {
        let va_space = match inner.va_spaces.iter().find(|v| {
            v.gpu_id == gpu_id
                && v.va_space_token == va_space_token
                && (v.rm_client_token == 0 || v.rm_client_token == rm_client_token)
        }) {
            Some(va_space) => va_space,
            None => return Ok(PolarisUvmFaultResult::NotMine),
        };
        let managed_end = va_space.managed_base.saturating_add(va_space.managed_length);
        if fault_address < va_space.managed_base || fault_address >= managed_end {
            return Ok(PolarisUvmFaultResult::NotMine);
        }
    } else {
        let gpu = match inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
            Some(gpu) => gpu,
            None => return Ok(PolarisUvmFaultResult::NotMine),
        };
        if !gpu.va_range_registered {
            return Ok(PolarisUvmFaultResult::NotMine);
        }
        let va_range_end = gpu.va_range_base.saturating_add(gpu.va_range_length);
        if fault_address < gpu.va_range_base || fault_address >= va_range_end {
            return Ok(PolarisUvmFaultResult::NotMine);
        }
    }

    // Search from the end so that newly-created pending blocks are
    // matched before older resident blocks that share the same VA
    // (COW break creates a CowPending block at the same VA as the
    // original shared block).
    let block_idx = match inner.blocks.iter()
        .enumerate()
        .rev()
        .find(|(_, b)| {
            b.home_gpu == gpu_id
                && fault_address >= b.gpu_vaddr
                && fault_address < b.gpu_vaddr.saturating_add(b.size_bytes)
        })
    {
        Some((idx, _)) => Some(idx),
        None => {
            // COW path: the fault is in a child session's VA range.
            // Find the owning session.
            let owning_sid = inner.sessions.iter()
                .find(|s| {
                    fault_address >= s.gpu_vas_base
                        && fault_address < s.gpu_vas_base.saturating_add(s.gpu_vas_size)
                })
                .map(|s| s.session_id);

            // Look up the COW-shared block via the owner session's block_ids.
            let mut found_idx = None;
            if let Some(sid) = owning_sid {
                // Collect COW-shared block data before pushing new block entries.
                let shared_info: Option<(u64, u64, u64, u32, u32, PolarisPhase)> =
                    if let Some(session) = inner.sessions.iter().find(|s| s.session_id == sid) {
                        let mut info = None;
                        for &bid in &session.block_ids {
                            if let Some((_bi, b)) = inner.blocks.iter()
                                .enumerate()
                                .find(|(_, b)| b.block_id == bid && b.session_id != sid)
                            {
                                if b.state == PolarisBlockState::Resident && b.gpu_phys_handle != 0 {
                                    info = Some((
                                        b.gpu_phys_handle,
                                        b.size_bytes,
                                        b.gpu_vaddr,
                                        b.token_start,
                                        b.token_count,
                                        b.phase,
                                    ));
                                    found_idx = Some(_bi);
                                    break;
                                } else {
                                    found_idx = Some(_bi);
                                    break;
                                }
                            }
                        }
                        info
                    } else {
                        None
                    };

                if let Some((phys_handle, sz, src_va, tok_start, tok_count, phase)) = shared_info {
                    let new_block_id = inner.next_block_id;
                    inner.next_block_id += 1;
                    let new_fault_id = inner.next_fault_id;
                    inner.next_fault_id += 1;
                    let generation = inner.fault_generation;
                    let decision_id = inner.next_decision_id;
                    inner.next_decision_id += 1;

                    inner.blocks.push(
                        PolarisBlock {
                            block_id: new_block_id,
                            session_id: sid,
                            token_start: tok_start,
                            token_count: tok_count,
                            home_gpu: gpu_id,
                            gpu_vaddr: fault_address,
                            gpu_phys_handle: 0,
                            rm_control_fd: 0,
                            rm_h_client: 0,
                            rm_h_memory: 0,
                            rm_backing_length: 0,
                            rm_backing_offset: 0,
                            cpu_buf_addr: 0,
                            size_bytes: sz,
                            refcount: 1,
                            state: PolarisBlockState::AllocPending,
                            flags: PolarisBlockFlags::empty(),
                            phase,
                            last_touch_ns: 0,
                            map_time_ns: 0,
                            cow_src_handle: 0,
                            retry_count: 0,
                            pending_decision_id: decision_id,
                            pending_fault_id: new_fault_id,
                            pending_generation: generation,
                            fault_timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                            completion_ptr: core::ptr::null_mut(),
                        },
                        GFP_KERNEL,
                    )?;

                    inner.pending_faults.push(
                        PolarisFault {
                            fault_id: new_fault_id,
                            generation,
                            gpu_id,
                            fault_address,
                            block_id: new_block_id,
                            access_type,
                            state: 0,
                            enqueue_ns: unsafe { bindings::ktime_get_mono_fast_ns() },
                            deadline_ns: 0,
                            resolved_ns: 0,
                        },
                        GFP_KERNEL,
                    )?;
                    inner.pending_decisions.push(
                        PolarisDecision {
                            decision_id,
                            fault_id: new_fault_id,
                            generation,
                            op: PolarisDecisionOp::MapExisting as u32,
                            gpu_id,
                            block_id: new_block_id,
                            session_id: sid,
                            src_handle: phys_handle,
                            dst_handle: 0,
                            src_vaddr: src_va,
                            dst_vaddr: fault_address,
                            size_bytes: sz,
                            cpu_addr: 0,
                            access_flags: access_type,
                            timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                            _reserved: [0u64; 4],
                        },
                        GFP_KERNEL,
                    )?;

                    let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
                    unsafe { bindings::init_completion(&raw mut comp); }
                    inner.blocks.last_mut().unwrap().completion_ptr = &raw mut comp;
                    drop(guard);

                    let wait_ret = unsafe {
                        bindings::wait_for_completion_interruptible_timeout(
                            &raw mut comp,
                            bindings::__msecs_to_jiffies(POLARIS_DEFAULT_FAULT_TIMEOUT_MS),
                        )
                    };
                    let mut guard = POLARIS_STATE.lock();
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    if let Some(blk) = inner.blocks.iter_mut().find(|b| b.block_id == new_block_id) {
                        blk.completion_ptr = core::ptr::null_mut();
                        if wait_ret == 0 && blk.state != PolarisBlockState::Resident {
                            blk.state = PolarisBlockState::Evicted;
                            polaris_clear_block_rm_backing(blk);
                            return Err(ETIMEDOUT);
                        }
                    }
                    return Ok(PolarisUvmFaultResult::Handled);
                }
            }
            found_idx
        }
    };

    let block_idx = match block_idx {
        Some(idx) => idx,
        None => return Ok(PolarisUvmFaultResult::Error),
    };

    let fault_id = inner.next_fault_id;
    inner.next_fault_id += 1;
    let generation = inner.fault_generation;
    let op = match inner.blocks[block_idx].state {
        PolarisBlockState::Unmapped => PolarisDecisionOp::Alloc,
        PolarisBlockState::CpuOffloaded => PolarisDecisionOp::Reload,
        PolarisBlockState::CowPending => PolarisDecisionOp::CowBreak,
        PolarisBlockState::Resident => PolarisDecisionOp::MapExisting,
        _ => PolarisDecisionOp::Alloc,
    };
    if matches!(
        op,
        PolarisDecisionOp::Alloc | PolarisDecisionOp::Reload | PolarisDecisionOp::CowBreak
    ) {
        let requesting_session_id = inner.blocks[block_idx].session_id;
        let needed_bytes = inner.blocks[block_idx].size_bytes;
        let protected_phys_handle = if op == PolarisDecisionOp::CowBreak {
            inner.blocks[block_idx].cow_src_handle
        } else {
            0
        };
        polaris_schedule_budget_victim(
            inner,
            requesting_session_id,
            gpu_id,
            needed_bytes,
            protected_phys_handle,
        )?;
    }
    let decision_id = inner.next_decision_id;
    inner.next_decision_id += 1;
    let (
        block_id,
        session_id,
        src_handle,
        dst_vaddr,
        size_bytes,
        cpu_addr,
        timeout_ms,
    );
    {
        let block = &mut inner.blocks[block_idx];
        block.pending_fault_id = fault_id;
        block.pending_generation = generation;
        block.pending_decision_id = decision_id;
        block.state = match op {
            PolarisDecisionOp::Reload => PolarisBlockState::ReloadPending,
            PolarisDecisionOp::CowBreak => PolarisBlockState::CowPending,
            PolarisDecisionOp::MapExisting => PolarisBlockState::Resident,
            _ => PolarisBlockState::AllocPending,
        };
        block.fault_timeout_ms = POLARIS_DEFAULT_FAULT_TIMEOUT_MS;
        block_id = block.block_id;
        session_id = block.session_id;
        src_handle = if block.state == PolarisBlockState::CowPending {
            block.cow_src_handle
        } else {
            block.gpu_phys_handle
        };
        dst_vaddr = block.gpu_vaddr;
        size_bytes = block.size_bytes;
        cpu_addr = block.cpu_buf_addr;
        timeout_ms = block.fault_timeout_ms;
    }
    inner.pending_faults.push(
        PolarisFault {
            fault_id,
            generation,
            gpu_id,
            fault_address,
            block_id,
            access_type,
            state: 0,
            enqueue_ns: unsafe { bindings::ktime_get_mono_fast_ns() },
            deadline_ns: 0,
            resolved_ns: 0,
        },
        GFP_KERNEL,
    )?;
    inner.pending_decisions.push(
        PolarisDecision {
            decision_id,
            fault_id,
            generation,
            op: op as u32,
            gpu_id,
            block_id,
            session_id,
            src_handle,
            dst_handle: 0,
            src_vaddr: fault_address,
            dst_vaddr,
            size_bytes,
            cpu_addr,
            access_flags: access_type,
            timeout_ms,
            _reserved: [0u64; 4],
        },
        GFP_KERNEL,
    )?;
    let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
    unsafe { bindings::init_completion(&raw mut comp); }
    inner.blocks[block_idx].completion_ptr = &raw mut comp;
    drop(guard);

    let wait_ret = unsafe {
        bindings::wait_for_completion_interruptible_timeout(
            &raw mut comp,
            bindings::__msecs_to_jiffies(POLARIS_DEFAULT_FAULT_TIMEOUT_MS),
        )
    };
    let mut guard = POLARIS_STATE.lock();
    let inner = guard.as_mut().ok_or(ENODEV)?;
    if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
        block.completion_ptr = core::ptr::null_mut();
        if wait_ret == 0 && block.state != PolarisBlockState::Resident {
            block.state = PolarisBlockState::Evicted;
            polaris_clear_block_rm_backing(block);
            return Err(ETIMEDOUT);
        }
    }
    Ok(PolarisUvmFaultResult::Handled)
}

// ─── sysfs buffer writer ────────────────────────────────────────────────────

/// Minimal `core::fmt::Write` impl over a fixed u8 buffer, for use in
/// sysfs show functions where we have a `char *buf` from the kernel.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl core::fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let room = self.buf.len().saturating_sub(1).saturating_sub(self.pos);
        let n = bytes.len().min(room);
        self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
        self.pos += n;
        Ok(())
    }
}

// ─── sysfs stats show function ──────────────────────────────────────────────

// Null-terminated "stats" as a byte array so the pointer can be used
// in a static. Using c_str!("stats").as_char_ptr() directly in a static
// may not work if as_char_ptr is not const fn.
const STATS_NAME_BYTES: &[u8] = b"stats\0";

// Newtype wrapper so we can safely mark the kobj_attribute as Sync.
// The attribute is write-once (at module init) then read-only forever;
// all its fields (function pointers, name pointer) are immutable.
// `unsafe impl Sync` is the standard kernel-Rust pattern for C structs
// that are only accessed under the kernel's concurrency guarantees.
struct PolarisStatsAttr(bindings::kobj_attribute);
unsafe impl Sync for PolarisStatsAttr {}

static POLARIS_STATS_ATTR: PolarisStatsAttr = PolarisStatsAttr(bindings::kobj_attribute {
    attr: bindings::attribute {
        name: STATS_NAME_BYTES.as_ptr() as *const kernel::ffi::c_char,
        mode: 0o444,
    },
    show: Some(polaris_stats_show),
    store: None,
});

unsafe extern "C" fn polaris_stats_show(
    _kobj: *mut bindings::kobject,
    _attr: *mut bindings::kobj_attribute,
    buf: *mut kernel::ffi::c_char,
) -> isize {
    let guard = POLARIS_STATE.lock();
    let inner = match guard.as_ref() {
        Some(i) => i,
        None => {
            // Module not fully initialized yet.
            return 0;
        }
    };

    // Count block states.
    let (mut resident, mut offloaded, mut evicted, mut pending) = (0u32, 0u32, 0u32, 0u32);
    let (mut shared, mut private) = (0u64, 0u64);
    let mut memory_saved: u64 = 0;
    for b in &inner.blocks {
        match b.state {
            PolarisBlockState::Resident => resident += 1,
            PolarisBlockState::CpuOffloaded => offloaded += 1,
            PolarisBlockState::Evicted => evicted += 1,
            _ => pending += 1,
        }
        if b.flags.contains(PolarisBlockFlag::Shared) {
            shared += b.size_bytes;
            memory_saved = memory_saved.saturating_add(
                b.refcount.saturating_sub(1).saturating_mul(b.size_bytes),
            );
        } else {
            private += b.size_bytes;
        }
    }

    let (mut total_gpu, mut used_gpu, mut cpu_total, mut cpu_used, mut unhealthy_gpus) =
        (0u64, 0u64, 0u64, 0u64, 0u32);
    for g in &inner.gpus {
        total_gpu += g.total_bytes;
        used_gpu += g.used_bytes;
        cpu_total += g.cpu_pool_total_bytes;
        cpu_used += g.cpu_pool_used_bytes;
        if !g.healthy {
            unhealthy_gpus += 1;
        }
    }

    let daemon = inner.daemon_attached;
    let sessions = inner.sessions.len();
    let blocks = inner.blocks.len();
    let gpus = inner.gpus.len();
    let decisions = inner.pending_decisions.len();
    let static_blocks = inner.static_blocks.len();
    let block_mappings = inner.block_mappings.len();
    let v4_worker_pids = inner.va_spaces.iter().filter(|v| v.pid != 0).count();
    let fast_va_spaces = POLARIS_FAST_VASPACES
        .iter()
        .filter(|slot| slot.va_space_token.load(Acquire) != 0)
        .count();
    let policy: u32 = inner.eviction_policy as u32;
    let offload_cnt = inner.offload_count;
    let reload_cnt = inner.reload_count;
    let evictions = inner.total_evictions;
    let cow_cnt = inner.cow_break_count;
    let cow_bytes = inner.cow_copy_bytes;
    let memory_saved_naive = memory_saved.saturating_sub(cow_bytes);
    let uvm_hook_calls = POLARIS_UVM_FAULT_HOOK_CALLS.load(Relaxed);
    let uvm_fast_matches = POLARIS_UVM_FAULT_FAST_MATCHES.load(Relaxed);
    let uvm_fast_misses = POLARIS_UVM_FAULT_FAST_MISSES.load(Relaxed);
    let uvm_unserviceable = POLARIS_UVM_FAULT_UNSERVICEABLE_MATCHES.load(Relaxed);
    let uvm_handled = POLARIS_UVM_FAULT_HANDLED.load(Relaxed);
    let uvm_errors = POLARIS_UVM_FAULT_ERRORS.load(Relaxed);
    let uvm_last_gpu = POLARIS_UVM_LAST_GPU_ID.load(Relaxed);
    let uvm_last_client = POLARIS_UVM_LAST_RM_CLIENT_TOKEN.load(Relaxed);
    let uvm_last_token = POLARIS_UVM_LAST_VA_SPACE_TOKEN.load(Relaxed);
    let uvm_last_gpu_va_space = POLARIS_UVM_LAST_GPU_VA_SPACE_PTR.load(Relaxed);
    let uvm_last_fault = POLARIS_UVM_LAST_FAULT_ADDRESS.load(Relaxed);
    let uvm_last_access = POLARIS_UVM_LAST_ACCESS_TYPE.load(Relaxed);
    let uvm_last_map_ret = POLARIS_UVM_LAST_MAP_RET.load(Relaxed);
    let uvm_last_ensure_ret = POLARIS_UVM_LAST_ENSURE_RET.load(Relaxed);
    let uvm_last_result = POLARIS_UVM_LAST_RESULT.load(Relaxed);
    drop(guard);

    // Write into the kernel-provided buffer (typically PAGE_SIZE = 4096).
    // SAFETY: buf points to a valid kernel buffer of at least PAGE_SIZE bytes.
    let buf_slice = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, 4096) };
    let len = {
        let mut w = BufWriter::new(buf_slice);
        let _ = core::fmt::write(
            &mut w,
            format_args!(
                "\
sessions:       {sessions}
blocks:         {blocks}
  resident:     {resident}
  offloaded:    {offloaded}
  evicted:      {evicted}
  pending:      {pending}
gpus:           {gpus}
  unhealthy:    {unhealthy}
daemon:         {daemon}
policy:         {policy} (0=fifo,1=lru,2=phase_aware)
offloads:       {offload_cnt}
reloads:        {reload_cnt}
evictions:      {evictions}
cow_breaks:     {cow_cnt}
cow_copy_mib:   {cow_copy_mib}
cow_saved_mib:  {saved_mib}
gpu_total_mib:  {gpu_total_mib}
gpu_used_mib:   {gpu_used_mib}
cpu_pool_mib:   {cpu_pool_mib}
cpu_used_mib:   {cpu_used_mib}
shared_mib:     {shared_mib}
private_mib:    {private_mib}
pending_decs:   {decisions}
v4_va_spaces:   {v4_va_spaces}
v4_worker_pids: {v4_worker_pids}
static_blocks:  {static_blocks}
block_mappings: {block_mappings}
uvm_hook_calls: {uvm_hook_calls}
uvm_fast_hits:  {uvm_fast_matches}
uvm_fast_miss:  {uvm_fast_misses}
uvm_no_pte:     {uvm_unserviceable}
uvm_handled:    {uvm_handled}
uvm_errors:     {uvm_errors}
uvm_last_gpu:   {uvm_last_gpu}
uvm_last_client:0x{uvm_last_client:x}
uvm_last_token: 0x{uvm_last_token:x}
uvm_last_va:    0x{uvm_last_gpu_va_space:x}
uvm_last_fault: 0x{uvm_last_fault:x}
uvm_last_access:{uvm_last_access}
uvm_last_map_ret:{uvm_last_map_ret}
uvm_last_ensure_ret:{uvm_last_ensure_ret}
uvm_last_result:{uvm_last_result}
",
                sessions = sessions,
                blocks = blocks,
                resident = resident,
                offloaded = offloaded,
                evicted = evicted,
                pending = pending,
                gpus = gpus,
                unhealthy = unhealthy_gpus,
                daemon = daemon,
                policy = policy,
                offload_cnt = offload_cnt,
                reload_cnt = reload_cnt,
                evictions = evictions,
                cow_cnt = cow_cnt,
                cow_copy_mib = cow_bytes / (1024 * 1024),
                saved_mib = memory_saved_naive / (1024 * 1024),
                gpu_total_mib = total_gpu / (1024 * 1024),
                gpu_used_mib = used_gpu / (1024 * 1024),
                cpu_pool_mib = cpu_total / (1024 * 1024),
                cpu_used_mib = cpu_used / (1024 * 1024),
                shared_mib = shared / (1024 * 1024),
                private_mib = private / (1024 * 1024),
                decisions = decisions,
                v4_va_spaces = fast_va_spaces,
                v4_worker_pids = v4_worker_pids,
                static_blocks = static_blocks,
                block_mappings = block_mappings,
                uvm_hook_calls = uvm_hook_calls,
                uvm_fast_matches = uvm_fast_matches,
                uvm_fast_misses = uvm_fast_misses,
                uvm_unserviceable = uvm_unserviceable,
                uvm_handled = uvm_handled,
                uvm_errors = uvm_errors,
                uvm_last_gpu = uvm_last_gpu,
                uvm_last_client = uvm_last_client,
                uvm_last_token = uvm_last_token,
                uvm_last_gpu_va_space = uvm_last_gpu_va_space,
                uvm_last_fault = uvm_last_fault,
                uvm_last_access = uvm_last_access,
                uvm_last_map_ret = uvm_last_map_ret,
                uvm_last_ensure_ret = uvm_last_ensure_ret,
                uvm_last_result = uvm_last_result,
            ),
        );
        w.pos
    };
    buf_slice[len] = 0; // null-terminate
    len as isize
}

// ─── Module declaration ─────────────────────────────────────────────────────

module! {
    type: PolarisModule,
    name: "polaris",
    authors: ["POLARIS Team"],
    description: "OS-level Paged KV Cache Management for LLM Inference",
    license: "GPL",
}

/// Create `/sys/kernel/polaris/stats` and return the kobject pointer.
fn init_polaris_sysfs() -> Result<*mut bindings::kobject> {
    // SAFETY: kernel_kobj is always valid. This is called only during module init.
    let polaris_kobj = unsafe {
        bindings::kobject_create_and_add(
            c_str!("polaris").as_char_ptr(),
            bindings::kernel_kobj,
        )
    };
    if polaris_kobj.is_null() {
        pr_err!("POLARIS: failed to create /sys/kernel/polaris kobject\n");
        return Err(ENOMEM);
    }

    // SAFETY: polaris_kobj is valid; POLARIS_STATS_ATTR is a static with
    // 'static lifetime (matches the kobject's lifetime).
    let ret = unsafe {
        bindings::sysfs_create_file_ns(
            polaris_kobj,
            &raw const POLARIS_STATS_ATTR.0.attr as *const bindings::attribute,
            core::ptr::null(),
        )
    };
    if ret != 0 {
        pr_err!("POLARIS: failed to create sysfs stats attribute (err {ret})\n");
        // SAFETY: polaris_kobj was just created above.
        unsafe { bindings::kobject_put(polaris_kobj) };
        return Err(ENOMEM);
    }

    pr_info!("POLARIS: /sys/kernel/polaris/stats created\n");
    Ok(polaris_kobj)
}

struct PolarisUvmHookRegistration;

impl PolarisUvmHookRegistration {
    fn register() -> Result<Self> {
        let hook_ret = unsafe { uvm_polaris_register_hook(&raw const UVM_POLARIS_OPS) };
        if hook_ret != 0 {
            pr_err!("POLARIS: failed to register UVM v4 fault hook (err {hook_ret})\n");
            return if hook_ret == -(bindings::EBUSY as c_int) {
                Err(EBUSY)
            } else if hook_ret == -(bindings::EINVAL as c_int) {
                Err(EINVAL)
            } else {
                Err(EIO)
            };
        }

        pr_info!("POLARIS: registered UVM v4 fault hook\n");
        Ok(Self)
    }
}

impl Drop for PolarisUvmHookRegistration {
    fn drop(&mut self) {
        unsafe { uvm_polaris_unregister_hook(&raw const UVM_POLARIS_OPS) };
        pr_info!("POLARIS: unregistered UVM v4 fault hook\n");
    }
}

#[pin_data(PinnedDrop)]
struct PolarisModule {
    _uvm_hook: PolarisUvmHookRegistration,
    #[pin]
    _miscdev: MiscDeviceRegistration<PolarisDevice>,
    /// sysfs kobject created under /sys/kernel/polaris
    polaris_kobj: *mut bindings::kobject,
}

// The raw pointer `polaris_kobj` is only touched during module init/exit
// (single-threaded, before/after the module is live).  After init it is
// read-only until exit.
unsafe impl Send for PolarisModule {}
unsafe impl Sync for PolarisModule {}

impl kernel::InPlaceModule for PolarisModule {
    fn init(_module: &'static ThisModule) -> impl PinInit<Self, Error> {
        pr_info!("POLARIS: initializing kernel module\n");

        // Initialize the C mutex backing the global lock.
        // SAFETY: Called exactly once at module init, before any use.
        unsafe { POLARIS_STATE.init() };

        // Install the real shared state.
        {
            let mut guard = POLARIS_STATE.lock();
            *guard = Some(PolarisInner {
                next_block_id: 1,
                next_session_id: 1,
                next_decision_id: 1,
                next_fault_id: 1,
                next_range_id: 1,
                fault_generation: 1,
                daemon_attached: 0,
                gpus: KVec::new(),
                blocks: KVec::new(),
                sessions: KVec::new(),
                va_spaces: KVec::new(),
                static_blocks: KVec::new(),
                block_mappings: KVec::new(),
                pending_decisions: KVec::new(),
                pending_faults: KVec::new(),
                eviction_policy: PolarisEvictionPolicy::Fifo,
                offload_count: 0,
                reload_count: 0,
                total_evictions: 0,
                cow_break_count: 0,
                cow_copy_bytes: 0,
            });
        }

        let options = MiscDeviceOptions {
            name: c_str!("polaris"),
        };

        try_pin_init!(Self {
            _uvm_hook: PolarisUvmHookRegistration::register()?,
            _miscdev <- MiscDeviceRegistration::<PolarisDevice>::register(options),
            polaris_kobj: init_polaris_sysfs()?,
        })
    }
}

#[pinned_drop]
impl PinnedDrop for PolarisModule {
    fn drop(self: Pin<&mut Self>) {
        // Mark that we're in the module exit path.  PolarisDevice drops
        // that run as a side-effect of _miscdev being dropped during
        // forced unload (rmmod -f) will see this and skip module_put.
        MODULE_EXITING.store(1, Relaxed);

        if !self.polaris_kobj.is_null() {
            // SAFETY: The kobject was created in init and is valid.
            // Order matters: remove files → del kobject from sysfs tree
            // → put reference. kobject_del synchronously waits for all
            // in-flight readers, so after it returns no one can call
            // polaris_stats_show anymore.
            unsafe {
                bindings::sysfs_remove_file_ns(
                    self.polaris_kobj,
                    &raw const POLARIS_STATS_ATTR.0.attr as *const bindings::attribute,
                    core::ptr::null(),
                );
                bindings::kobject_del(self.polaris_kobj);
                bindings::kobject_put(self.polaris_kobj);
            }
        }

        polaris_fast_vaspace_clear_all();
    }
}

// ─── Device (per open fd, but all share POLARIS_STATE) ──────────────────────

#[pin_data(PinnedDrop)]
struct PolarisDevice {
    dev: ARef<Device>,
    /// Whether this fd registered as an executor.
    registered_gpu: Atomic<u32>,
    registered_transient_gpu: Atomic<u32>,
    registered_transient_gpu_id: Atomic<u32>,
    registered_va_gpu: Atomic<u32>,
    registered_va_range_id: Atomic<u64>,
    registered_v4_gpu: Atomic<u32>,
    registered_v4_client: Atomic<u64>,
    registered_v4_token: Atomic<u64>,
}

#[vtable]
impl MiscDevice for PolarisDevice {
    type Ptr = Pin<KBox<Self>>;

    fn open(_file: &File, misc: &MiscDeviceRegistration<Self>) -> Result<Pin<KBox<Self>>> {
        let dev = ARef::from(misc.device());
        dev_info!(dev, "POLARIS: device opened\n");

        let ptr = KBox::try_pin_init(
            try_pin_init! {
                PolarisDevice {
                    dev: dev,
                    registered_gpu: Atomic::new(0),
                    registered_transient_gpu: Atomic::new(0),
                    registered_transient_gpu_id: Atomic::new(0),
                    registered_va_gpu: Atomic::new(0),
                    registered_va_range_id: Atomic::new(0),
                    registered_v4_gpu: Atomic::new(0),
                    registered_v4_client: Atomic::new(0),
                    registered_v4_token: Atomic::new(0),
                }
            },
            GFP_KERNEL,
        )?;

        // Keep the module loaded while this fd is open.  The kernel Rust
        // miscdevice vtable cannot yet set fops->owner = THIS_MODULE (the
        // const-eval limitation documented in gen_disk.rs), so we do it
        // by hand.
        // SAFETY: __this_module is valid for our lifetime; we are inside
        // the module's own open handler, so the module is definitely live.
        unsafe {
            bindings::__module_get(
                core::ptr::addr_of!(bindings::__this_module) as *mut bindings::module,
            );
        }

        Ok(ptr)
    }

    fn ioctl(me: Pin<&PolarisDevice>, _file: &File, cmd: u32, arg: usize) -> Result<isize> {
        let user_ptr = UserPtr::from_addr(arg);
        let size = _IOC_SIZE(cmd);

        // All handlers share the same global state via POLARIS_STATE.
        match cmd {
            POLARIS_REGISTER_GPU => me.handle_register_gpu(user_ptr, size),
            POLARIS_REGISTER_VA_RANGE => me.handle_register_va_range(user_ptr, size),
            POLARIS_SESSION_CREATE => me.handle_session_create(user_ptr, size),
            POLARIS_SESSION_DESTROY => me.handle_session_destroy(user_ptr, size),
            POLARIS_SESSION_GET_STATS => me.handle_session_get_stats(user_ptr, size),
            POLARIS_SESSION_BRANCH => me.handle_session_branch(user_ptr, size),
            POLARIS_BLOCK_RESERVE => me.handle_block_reserve(user_ptr, size),
            POLARIS_BLOCK_RELEASE => me.handle_block_release(user_ptr, size),
            POLARIS_BLOCK_TOUCH => me.handle_block_touch(user_ptr, size),
            POLARIS_BLOCK_GET_STATE => me.handle_block_get_state(user_ptr, size),
            POLARIS_GET_DECISION => me.handle_get_decision(user_ptr, size),
            POLARIS_COMPLETE_OPERATION => me.handle_complete_operation(user_ptr, size),
            POLARIS_GET_GLOBAL_STATS => me.handle_get_global_stats(user_ptr, size),
            POLARIS_LIST_SESSIONS => me.handle_list_sessions(user_ptr, size),
            POLARIS_SET_POLICY => me.handle_set_policy(user_ptr, size),
            POLARIS_REGISTER_VASPACE => me.handle_register_va_space(user_ptr, size),
            POLARIS_UNREGISTER_VASPACE => me.handle_unregister_va_space(user_ptr, size),
            POLARIS_REGISTER_STATIC_BLOCK => me.handle_register_static_block(user_ptr, size),
            POLARIS_UNMAP_STATIC_BLOCK => me.handle_unmap_static_block(user_ptr, size),
            POLARIS_REGISTER_BLOCK_MAPPING => me.handle_register_block_mapping(user_ptr, size),
            POLARIS_UNMAP_BLOCK_MAPPINGS => me.handle_unmap_block_mappings(user_ptr, size),
            POLARIS_SPILL_BLOCK => me.handle_spill_block(user_ptr, size),
            POLARIS_REGISTER_BLOCK_BACKING => me.handle_register_block_backing(user_ptr, size),
            POLARIS_PROBE_RM_PHYS => me.handle_probe_rm_phys(user_ptr, size),
            POLARIS_PROBE_RM_COPY => me.handle_probe_rm_copy(user_ptr, size),
            POLARIS_RM_COPY => me.handle_rm_copy(user_ptr, size),
            _ => {
                dev_err!(me.dev, "POLARIS: unknown ioctl 0x{:x}\n", cmd);
                Err(ENOTTY)
            }
        }
    }
}

#[pinned_drop]
impl PinnedDrop for PolarisDevice {
    fn drop(self: Pin<&mut Self>) {
        if self.registered_gpu.load(Relaxed) != 0 {
            let mut guard = POLARIS_STATE.lock();
            if let Some(inner) = guard.as_mut() {
                let range_id = self.registered_va_range_id.load(Relaxed);
                if range_id != 0 {
                    let range_gpu = self.registered_va_gpu.load(Relaxed);
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == range_gpu) {
                        if gpu.va_range_id == range_id {
                            gpu.va_range_id = 0;
                            gpu.va_range_base = 0;
                            gpu.va_range_length = 0;
                            gpu.va_block_size = 0;
                            gpu.va_range_flags = 0;
                            gpu.va_range_registered = false;
                            gpu.next_va_offset = 0;
                            dev_info!(
                                self.dev,
                                "POLARIS: VA range {} on GPU {} unregistered\n",
                                range_id,
                                range_gpu
                            );
                        }
                    }
                }

                dev_info!(self.dev, "POLARIS: executor disconnected, evicting pending blocks\n");
                inner.daemon_attached = inner.daemon_attached.saturating_sub(1);
                if self.registered_transient_gpu.load(Relaxed) != 0 {
                    let transient_gpu_id = self.registered_transient_gpu_id.load(Relaxed);
                    let still_referenced = inner.va_spaces.iter().any(|v| v.gpu_id == transient_gpu_id)
                        || inner.blocks.iter().any(|b| b.home_gpu == transient_gpu_id)
                        || inner.sessions.iter().any(|s| s.home_gpu == transient_gpu_id);
                    if !still_referenced {
                        inner.gpus.retain(|g| g.gpu_id != transient_gpu_id);
                    }
                }
                for block in inner.blocks.iter_mut() {
                    match block.state {
                        PolarisBlockState::AllocPending
                        | PolarisBlockState::OffloadPending
                        | PolarisBlockState::ReloadPending
                        | PolarisBlockState::CowPending
                        | PolarisBlockState::FreePending => {
                            block.state = PolarisBlockState::Evicted;
                            block.pending_decision_id = 0;
                            polaris_clear_block_rm_backing(block);
                            // Wake any bounded fault waiter.
                            let comp_ptr = block.completion_ptr;
                            block.completion_ptr = core::ptr::null_mut();
                            if !comp_ptr.is_null() {
                                unsafe { bindings::complete(comp_ptr); }
                            }
                        }
                        _ => {}
                    }
                }
                inner.pending_decisions.clear();
            }
        }

        let v4_token = self.registered_v4_token.load(Relaxed);
        if v4_token != 0 {
            let v4_gpu = self.registered_v4_gpu.load(Relaxed);
            let v4_client = self.registered_v4_client.load(Relaxed);
            let mut guard = POLARIS_STATE.lock();
            if let Some(inner) = guard.as_mut() {
                inner.va_spaces.retain(|v| {
                    !(v.gpu_id == v4_gpu
                        && v.rm_client_token == v4_client
                        && v.va_space_token == v4_token)
                });
                inner.static_blocks.retain(|b| {
                    !(b.gpu_id == v4_gpu
                        && b.rm_client_token == v4_client
                        && b.va_space_token == v4_token)
                });
                inner.block_mappings.retain(|m| {
                    !(m.gpu_id == v4_gpu
                        && m.rm_client_token == v4_client
                        && m.va_space_token == v4_token)
                });
            }
            polaris_fast_vaspace_unregister(v4_gpu, v4_client, v4_token);
            polaris_fast_static_blocks_unregister_va_space(v4_gpu, v4_client, v4_token);
        }

        dev_info!(self.dev, "POLARIS: device closed\n");

        // Release the module reference taken in open().  Skip during
        // force-unload (MODULE_EXITING is set before _miscdev is dropped
        // and triggers this path) — the kernel already zeroed the refcount
        // and calling module_put would BUG().
        if MODULE_EXITING.load(Relaxed) == 0 {
            // SAFETY: __this_module is valid; we took a reference in open().
            unsafe {
                bindings::module_put(
                    core::ptr::addr_of!(bindings::__this_module) as *mut bindings::module,
                );
            }
        }
    }
}

// ─── IOCTL handler implementations ──────────────────────────────────────────

impl PolarisDevice {
    fn handle_register_gpu(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterGpuArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let transient = (arg._reserved & POLARIS_REGISTER_GPU_FLAG_TRANSIENT) != 0;

        // Idempotent: if this GPU ID is already registered, update only
        // non-zero parameters. In-process runtimes attach with zero capacity
        // fields so they do not overwrite the control-plane daemon's global
        // GPU accounting.
        if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == arg.gpu_id) {
            if arg.total_bytes != 0 {
                gpu.total_bytes = arg.total_bytes;
            }
            if arg.budget_bytes != 0 {
                gpu.budget_bytes = arg.budget_bytes;
            }
            if arg.cpu_pool_bytes != 0 {
                gpu.cpu_pool_total_bytes = arg.cpu_pool_bytes;
            }
            gpu.healthy = true;
            dev_info!(self.dev, "POLARIS: GPU {} re-registered\n", arg.gpu_id);
        } else {
            if arg.total_bytes == 0 || arg.budget_bytes == 0 {
                return Err(ENOENT);
            }
            inner.gpus.push(
                PolarisGpu {
                    gpu_id: arg.gpu_id,
                    gpu_uuid: [0u8; 16],
                    total_bytes: arg.total_bytes,
                    used_bytes: 0,
                    budget_bytes: arg.budget_bytes,
                    pressure_score: 0,
                    cpu_pool_total_bytes: arg.cpu_pool_bytes,
                    cpu_pool_used_bytes: 0,
                    va_range_id: 0,
                    va_range_base: 0,
                    va_range_length: 0,
                    va_block_size: 0,
                    va_range_flags: 0,
                    va_range_registered: false,
                    healthy: true,
                    next_va_offset: 0,
                },
                GFP_KERNEL,
            )?;
            dev_info!(self.dev, "POLARIS: GPU {} registered\n", arg.gpu_id);
        }

        if self.registered_gpu.load(Relaxed) == 0 {
            self.registered_gpu.store(1, Relaxed);
            if transient {
                self.registered_transient_gpu.store(1, Relaxed);
                self.registered_transient_gpu_id.store(arg.gpu_id, Relaxed);
            }
            inner.daemon_attached += 1;
        }
        Ok(0)
    }

    fn handle_register_va_range(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisRegisterVaRangeArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        if self.registered_gpu.load(Relaxed) == 0 {
            return Err(EPERM);
        }
        if arg.length == 0 || arg.block_size == 0 {
            return Err(EINVAL);
        }

        if arg.range_id == 0 {
            arg.range_id = ((arg.gpu_id as u64) << 32) | inner.next_range_id;
            inner.next_range_id += 1;
        }
        let gpu = inner.gpus.iter_mut().find(|g| g.gpu_id == arg.gpu_id).ok_or(ENOENT)?;
        gpu.va_range_id = arg.range_id;
        gpu.va_range_base = arg.base;
        gpu.va_range_length = arg.length;
        gpu.va_block_size = arg.block_size;
        gpu.va_range_flags = arg.flags;
        gpu.va_range_registered = true;
        self.registered_va_gpu.store(arg.gpu_id, Relaxed);
        self.registered_va_range_id.store(arg.range_id, Relaxed);

        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_session_create(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionCreateArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        let (gpu_vas_base, gpu_vas_size) =
            if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == arg.home_gpu) {
                if !gpu.healthy {
                dev_err!(self.dev, "POLARIS: GPU {} is unhealthy, rejecting session\n", arg.home_gpu);
                return Err(ENODEV);
            }
                if gpu.va_range_registered {
                    let session_va = if arg.gpu_vas_bytes > 0 {
                        arg.gpu_vas_bytes
                    } else {
                        POLARIS_DEFAULT_SESSION_VA_BYTES
                    };
                    let aligned_va = (session_va + POLARIS_VA_ALIGNMENT - 1) & !(POLARIS_VA_ALIGNMENT - 1);
                    if gpu.next_va_offset + aligned_va > gpu.va_range_length {
                        dev_err!(self.dev, "POLARIS: VA pool exhausted for GPU {}\n", arg.home_gpu);
                        return Err(ENOMEM);
                    }
                    let base = gpu.va_range_base + gpu.next_va_offset;
                    gpu.next_va_offset += aligned_va;
                    (base, session_va)
                } else {
                    (0, arg.gpu_vas_bytes)
                }
            } else {
                (0, arg.gpu_vas_bytes)
            };

        let session_id = inner.next_session_id;
        inner.next_session_id += 1;

        let bpt = if arg.bytes_per_token > 0 {
            arg.bytes_per_token
        } else {
            POLARIS_DEFAULT_BYTES_PER_TOKEN
        };

        let priority = if arg.priority > 0 { arg.priority } else { 5 };
        inner.sessions.push(
            PolarisSession {
                session_id,
                home_gpu: arg.home_gpu,
                gpu_vas_base,
                gpu_vas_size,
                beam_width: arg.beam_width,
                bytes_per_token: bpt,
                parent_session_id: 0,
                priority,
                block_ids: KVec::new(),
            },
            GFP_KERNEL,
        )?;

        arg.session_id = session_id;
        drop(guard); // release lock before writing back to userspace
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} created\n", session_id);
        Ok(0)
    }

    fn handle_session_destroy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisSessionDestroyArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let sid = arg.session_id;

        if !inner.sessions.iter().any(|s| s.session_id == sid) {
            return Err(ENOENT);
        }

        let gpu_id = inner
            .sessions
            .iter()
            .find(|s| s.session_id == sid)
            .map(|s| s.home_gpu)
            .unwrap_or(0);
        let (session_va_base, session_va_end) = inner
            .sessions
            .iter()
            .find(|s| s.session_id == sid)
            .map(|s| {
                (
                    s.gpu_vas_base,
                    s.gpu_vas_base.saturating_add(s.gpu_vas_size),
                )
            })
            .unwrap_or((0, 0));

        // Collect info before any mutation (avoids double-borrow with
        // pending_decisions.push inside the loop).
        struct ToFree {
            idx: usize,
            block_id: u64,
            home_gpu: u32,
            size_bytes: u64,
            state: PolarisBlockState,
            refcount: u64,
            had_cpu_buf: bool,
            needs_free_decision: bool,
        }
        let mut to_free: KVec<ToFree> = KVec::new();
        for idx in 0..inner.blocks.len() {
            let b = &inner.blocks[idx];
            if b.session_id == sid {
                to_free.push(
                    ToFree {
                        idx,
                        block_id: b.block_id,
                        home_gpu: b.home_gpu,
                        size_bytes: b.size_bytes,
                        state: b.state,
                        refcount: b.refcount,
                        had_cpu_buf: b.cpu_buf_addr != 0,
                        needs_free_decision: polaris_block_needs_free_decision(b),
                    },
                    GFP_KERNEL,
                )?;
            }
        }

        // Second pass: for COW child sessions, the block table is scanned
        // by session_id which misses shared blocks (their session_id is the
        // parent's).  Walk the session's block_ids list to find these
        // COW-shared blocks and decrement their refcounts.
        {
            let session = inner.sessions.iter().find(|s| s.session_id == sid);
            if let Some(sess) = session {
                for &bid in &sess.block_ids {
                    // Skip blocks already collected in the first pass.
                    if to_free.iter().any(|tf| tf.block_id == bid) {
                        continue;
                    }
                    if let Some(idx) = inner.blocks.iter().position(|b| b.block_id == bid) {
                        let b = &inner.blocks[idx];
                        to_free.push(
                            ToFree {
                                idx,
                                block_id: b.block_id,
                                home_gpu: b.home_gpu,
                                size_bytes: b.size_bytes,
                                state: b.state,
                                refcount: b.refcount,
                                had_cpu_buf: b.cpu_buf_addr != 0,
                                needs_free_decision: polaris_block_needs_free_decision(b),
                            },
                            GFP_KERNEL,
                        )?;
                    }
                }
            }
        }

        // Now mutate: queue FREE or clean up directly.
        if inner.daemon_attached > 0 {
            let needed_free_decisions = to_free
                .iter()
                .filter(|tf| {
                    tf.refcount <= 1
                        && tf.state != PolarisBlockState::FreePending
                        && tf.needs_free_decision
                })
                .count();
            if inner.pending_decisions.len().saturating_add(needed_free_decisions)
                > POLARIS_MAX_PENDING_DECISIONS
            {
                dev_warn!(
                    self.dev,
                    "POLARIS: pending decision queue has {} free slot(s), refusing to drop {} live backing object(s) for session {}\n",
                    POLARIS_MAX_PENDING_DECISIONS.saturating_sub(inner.pending_decisions.len()),
                    needed_free_decisions,
                    sid
                );
                return Err(ENOMEM);
            }
        }

        for tf in &mut to_free {
            let block_id = tf.block_id;
            let sz = tf.size_bytes;

            if tf.refcount > 1 {
                let block = &mut inner.blocks[tf.idx];
                block.refcount -= 1;
                if block.refcount == 1 {
                    block.flags = block.flags & !PolarisBlockFlag::Shared;
                }
                inner.block_mappings.retain(|m| {
                    !(m.block_id == block_id
                        && m.gpu_id == gpu_id
                        && m.base < session_va_end
                        && m.base.saturating_add(m.length) > session_va_base)
                });
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} refcount decremented to {}\n",
                    sid, block_id, block.refcount
                );
                continue;
            }

            if tf.state == PolarisBlockState::FreePending {
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} already has FREE pending\n",
                    sid, block_id
                );
                continue;
            }

            if inner.daemon_attached > 0 && tf.needs_free_decision {
                let dec_id = polaris_queue_free_decision(inner, tf.idx)?;
                polaris_forget_static_blocks_for_block(inner, block_id)?;
                inner.block_mappings.retain(|m| m.block_id != block_id);
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} free queued (FREE {})\n",
                    sid, block_id, dec_id
                );
            } else {
                // No daemon or never mapped — remove directly.
                polaris_account_direct_block_removal(inner, tf.home_gpu, tf.state, sz, tf.had_cpu_buf);
                inner.blocks[tf.idx].refcount = 0;
                polaris_forget_static_blocks_for_block(inner, block_id)?;
                inner.block_mappings.retain(|m| m.block_id != block_id);
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} freed directly\n",
                    sid, block_id
                );
            }
        }

        // Remove session entry. FreePending blocks stay in the table for
        // the daemon to complete; non-FreePending blocks for this session
        // are removed.
        inner.sessions.retain(|s| s.session_id != sid);
        // Only remove blocks whose refcount has dropped to 0.  COW-shared
        // blocks (refcount > 0 after decrement) must stay in the table for
        // child sessions that still reference them.
        inner.blocks.retain(|b| {
            b.state == PolarisBlockState::FreePending
                || b.refcount != 0
                || !to_free.iter().any(|tf| tf.block_id == b.block_id)
        });
        inner.block_mappings.retain(|m| {
            inner.blocks.iter().any(|b| b.block_id == m.block_id)
        });

        dev_info!(self.dev, "POLARIS: session {} destroyed\n", sid);
        Ok(0)
    }

    fn handle_session_get_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionGetStatsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        match inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
            Some(session) => {
                arg.home_gpu = session.home_gpu;
                arg.beam_width = session.beam_width;
                arg.num_blocks = session.block_ids.len() as u32;
                arg.total_bytes = session.gpu_vas_size;
            }
            None => return Err(ENOENT),
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_session_branch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionBranchArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Extract parent info (immutable borrow) before mutating.
        let parent = inner
            .sessions
            .iter()
            .find(|s| s.session_id == arg.parent_session_id)
            .ok_or(ENOENT)?;
        let parent_gpu = parent.home_gpu;
        let parent_vas_base = parent.gpu_vas_base;
        let parent_vas_size = parent.gpu_vas_size;
        let parent_beam = parent.beam_width;
        let parent_bpt = parent.bytes_per_token;
        let parent_priority = parent.priority;
        let parent_block_ids: KVec<u64> = {
            let mut ids = KVec::new();
            for &bid in &parent.block_ids {
                ids.push(bid, GFP_KERNEL)?;
            }
            ids
        };
        let parent_id = arg.parent_session_id;

        // G4: reject branch if the parent's GPU is unhealthy.
        if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == parent_gpu) {
            if !gpu.healthy {
                dev_err!(self.dev, "POLARIS: GPU {} unhealthy, rejecting SESSION_BRANCH\n", parent_gpu);
                return Err(ENODEV);
            }
        }
        let _ = parent;

        // COW: increment refcount on all parent blocks.
        for &bid in &parent_block_ids {
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                block.refcount += 1;
                block.flags |= PolarisBlockFlag::Shared;
            }
        }

        let child_id = inner.next_session_id;
        inner.next_session_id += 1;

        let (child_vas_base, child_vas_size) =
            if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == parent_gpu) {
                if gpu.va_range_registered {
                    let va_needed = parent_vas_size;
                    let aligned = (va_needed + POLARIS_VA_ALIGNMENT - 1) & !(POLARIS_VA_ALIGNMENT - 1);
                    if gpu.next_va_offset + aligned > gpu.va_range_length {
                        dev_err!(self.dev, "POLARIS: VA pool exhausted for GPU {}, rejecting branch\n", parent_gpu);
                        return Err(ENOMEM);
                    }
                    let base = gpu.va_range_base + gpu.next_va_offset;
                    gpu.next_va_offset += aligned;
                    (base, parent_vas_size)
                } else {
                    (parent_vas_base, parent_vas_size)
                }
            } else {
                (parent_vas_base, parent_vas_size)
            };

        inner.sessions.push(
            PolarisSession {
                session_id: child_id,
                home_gpu: parent_gpu,
                gpu_vas_base: child_vas_base,
                gpu_vas_size: child_vas_size,
                beam_width: parent_beam,
                bytes_per_token: parent_bpt,
                parent_session_id: parent_id,
                priority: parent_priority,
                block_ids: parent_block_ids,
            },
            GFP_KERNEL,
        )?;

        arg.child_session_id = child_id;
        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} branched from {}\n", child_id, parent_id);
        Ok(0)
    }

    fn handle_block_reserve(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockReserveArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        if inner.daemon_attached == 0 {
            return Err(ENODEV);
        }
        if !inner.sessions.iter().any(|s| s.session_id == arg.session_id) {
            return Err(ENOENT);
        }
        // Extract all fields from the session reference before any
        // mutable borrow of inner.sessions below.
        let sess = inner.sessions.iter().find(|s| s.session_id == arg.session_id).ok_or(ENOENT)?;
        let gpu_vas_base = sess.gpu_vas_base;
        let bytes_per_token = sess.bytes_per_token;
        let home_gpu = sess.home_gpu;
        let gpu_vaddr = gpu_vas_base + ((arg.token_start as u64) * bytes_per_token);
        let size_bytes = (arg.token_count as u64).saturating_mul(bytes_per_token);
        let req_start = arg.token_start as u64;
        let req_end = req_start + arg.token_count as u64;
        let overwrite = arg.flags & POLARIS_RESERVE_FLAG_OVERWRITE != 0;
        let mut existing_block_idx: Option<usize> = None;

        // Collect all block IDs accessible to this session (directly
        // owned + COW-shared from parent).
        let mut session_bids: KVec<u64> = KVec::new();
        for &bid in &sess.block_ids {
            session_bids.push(bid, GFP_KERNEL)?;
        }

        // ── Phase-2 COW logic ──────────────────────────────────────────
        // Two-phase overlap detection:
        //   Pass 1: directly-owned blocks (session_id matches)
        //   Pass 2: COW-shared blocks (in session block_ids but
        //            different session_id)
        // This ordering ensures that a post-COW-break private block
        // (refcount==1) is found before the original shared block.

        // Pass 1: directly-owned blocks.
        for (i, b) in inner.blocks.iter().enumerate() {
            if b.session_id == arg.session_id {
                let b_start = b.token_start as u64;
                let b_end = b_start + b.token_count as u64;
                if req_start < b_end && req_end > b_start {
                    if !overwrite {
                        return Err(EINVAL);
                    }
                    existing_block_idx = Some(i);
                    break;
                }
            }
        }
        if existing_block_idx.is_none() {
            // Pass 2: COW-shared blocks (different session_id, but in this
            // session's block_ids list).
            for (i, b) in inner.blocks.iter().enumerate() {
                if session_bids.iter().any(|&bid| bid == b.block_id) {
                    let b_start = b.token_start as u64;
                    let b_end = b_start + b.token_count as u64;
                    if req_start < b_end && req_end > b_start {
                        if !overwrite {
                            return Err(EINVAL);
                        }
                        existing_block_idx = Some(i);
                        break;
                    }
                }
            }
        }

        if let Some(eb_idx) = existing_block_idx {
            let eb = &mut inner.blocks[eb_idx];
            if eb.refcount > 1 {
                // COW break: create a private copy.
                let block_id = inner.next_block_id;
                inner.next_block_id += 1;
                let old_block_id = eb.block_id;
                let cow_src = eb.gpu_phys_handle;
                eb.refcount -= 1;
                if eb.refcount == 1 {
                    eb.flags = eb.flags & !PolarisBlockFlag::Shared;
                }
                inner.blocks.push(
                    PolarisBlock {
                        block_id,
                        session_id: arg.session_id,
                        token_start: arg.token_start,
                        token_count: arg.token_count,
                        home_gpu,
                        gpu_vaddr,
                        gpu_phys_handle: 0,
                        rm_control_fd: 0,
                        rm_h_client: 0,
                        rm_h_memory: 0,
                        rm_backing_length: 0,
                        rm_backing_offset: 0,
                        cpu_buf_addr: 0,
                        size_bytes,
                        refcount: 1,
                        state: PolarisBlockState::CowPending,
                        flags: PolarisBlockFlags::empty(),
                        phase: if arg.phase == PolarisPhase::Decode as u32 {
                            PolarisPhase::Decode
                        } else {
                            PolarisPhase::Prefill
                        },
                        last_touch_ns: 0,
                        map_time_ns: 0,
                        cow_src_handle: cow_src,
                        retry_count: 0,
                        pending_decision_id: 0,
                        pending_fault_id: 0,
                        pending_generation: 0,
                        fault_timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                        completion_ptr: core::ptr::null_mut(),
                    },
                    GFP_KERNEL,
                )?;
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                    let mut replaced = false;
                    for bid in &mut session.block_ids {
                        if *bid == old_block_id {
                            *bid = block_id;
                            replaced = true;
                            break;
                        }
                    }
                    if !replaced {
                        session.block_ids.push(block_id, GFP_KERNEL)?;
                    }
                }
                arg.block_id = block_id;
                arg.gpu_vaddr = gpu_vaddr;
                drop(guard);
                let mut writer = UserSlice::new(user_ptr, size).writer();
                writer.write(&arg)?;

                let fault_result = polaris_resolve_gpu_fault(
                    home_gpu,
                    0,
                    0,
                    gpu_vaddr,
                    1, // write fault for COW break
                )?;
                if fault_result != PolarisUvmFaultResult::Handled {
                    return Err(EIO);
                }
                return Ok(0);
            }
            // refcount == 1: in-place overwrite — return existing block.
            eb.token_start = arg.token_start;
            eb.token_count = arg.token_count;
            eb.size_bytes = size_bytes;
            eb.gpu_vaddr = gpu_vaddr;
            polaris_clear_block_rm_backing(eb);
            arg.block_id = eb.block_id;
            arg.gpu_vaddr = gpu_vaddr;
            let should_resolve = !matches!(eb.state, PolarisBlockState::Resident)
                && (arg.flags & POLARIS_RESERVE_FLAG_DEFER_FAULT) == 0;
            let resolve_gpu = home_gpu;
            let resolve_vaddr = gpu_vaddr;
            drop(guard);
            let mut writer = UserSlice::new(user_ptr, size).writer();
            writer.write(&arg)?;
            if should_resolve {
                let fault_result = polaris_resolve_gpu_fault(
                    resolve_gpu,
                    0,
                    0,
                    resolve_vaddr,
                    0,
                )?;
                if fault_result != PolarisUvmFaultResult::Handled {
                    return Err(EIO);
                }
            }
            return Ok(0);
        }

        // No overlap: create a fresh block (normal path).
        let block_id = inner.next_block_id;
        inner.next_block_id += 1;
        inner.blocks.push(
            PolarisBlock {
                block_id,
                session_id: arg.session_id,
                token_start: arg.token_start,
                token_count: arg.token_count,
                home_gpu,
                gpu_vaddr,
                gpu_phys_handle: 0,
                rm_control_fd: 0,
                rm_h_client: 0,
                rm_h_memory: 0,
                rm_backing_length: 0,
                rm_backing_offset: 0,
                cpu_buf_addr: 0,
                size_bytes,
                refcount: 1,
                state: PolarisBlockState::Unmapped,
                flags: PolarisBlockFlags::empty(),
                phase: if arg.phase == PolarisPhase::Decode as u32 {
                    PolarisPhase::Decode
                } else {
                    PolarisPhase::Prefill
                },
                last_touch_ns: 0,
                map_time_ns: 0,
                cow_src_handle: 0,
                retry_count: 0,
                pending_decision_id: 0,
                pending_fault_id: 0,
                pending_generation: 0,
                fault_timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                completion_ptr: core::ptr::null_mut(),
            },
            GFP_KERNEL,
        )?;
        if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
            session.block_ids.push(block_id, GFP_KERNEL)?;
        }
        arg.block_id = block_id;
        arg.gpu_vaddr = gpu_vaddr;
        let defer_fault = arg.flags & POLARIS_RESERVE_FLAG_DEFER_FAULT != 0;
        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        if defer_fault {
            return Ok(0);
        }

        // Trigger the GPU page-fault pipeline through the same entry point
        // that the patched nvidia-uvm.ko replayable-fault ISR uses.  This
        // exercises the full kernel→daemon→kernel decision protocol
        // (ALLOC + cuMemMap) while still being driven from BLOCK_RESERVE
        // (the UVM hook would provide the trigger at interrupt time; here
        // the ioctl provides it synchronously so the workload can block
        // until physical memory is resident).
        let fault_result = polaris_resolve_gpu_fault(
            home_gpu,
            0,
            0,
            gpu_vaddr,
            0,
        )?;
        if fault_result != PolarisUvmFaultResult::Handled {
            return Err(EIO);
        }

        Ok(0)
    }

    fn handle_block_release(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockReleaseArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let (session_bids, session_home_gpu, release_start) = {
            let session = inner
                .sessions
                .iter()
                .find(|s| s.session_id == arg.session_id)
                .ok_or(ENOENT)?;
            let mut ids: KVec<u64> = KVec::new();
            for &bid in &session.block_ids {
                ids.push(bid, GFP_KERNEL)?;
            }
            (
                ids,
                session.home_gpu,
                session
                    .gpu_vas_base
                    .saturating_add((arg.token_start as u64).saturating_mul(session.bytes_per_token)),
            )
        };
        let idx = inner.blocks.iter().position(|b| {
            session_bids.iter().any(|bid| *bid == b.block_id)
                && b.token_start == arg.token_start
                && b.token_count == arg.token_count
        }).ok_or(ENOENT)?;
        let block_id = inner.blocks[idx].block_id;
        let block_home_gpu = inner.blocks[idx].home_gpu;
        let block_state = inner.blocks[idx].state;
        let block_size = inner.blocks[idx].size_bytes;
        let block_had_cpu_buf = inner.blocks[idx].cpu_buf_addr != 0;
        let caller_owns_backing = arg.flags & POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING != 0;
        let release_end = release_start.saturating_add(inner.blocks[idx].size_bytes);
        if inner.blocks[idx].refcount > 1 {
            inner.blocks[idx].refcount -= 1;
            if inner.blocks[idx].refcount == 1 {
                inner.blocks[idx].flags = inner.blocks[idx].flags & !PolarisBlockFlag::Shared;
            }
            inner.block_mappings.retain(|m| {
                !(m.block_id == block_id
                    && m.gpu_id == session_home_gpu
                    && m.base < release_end
                    && m.base.saturating_add(m.length) > release_start)
            });
            if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                session.block_ids.retain(|bid| *bid != block_id);
            }
        } else {
            let needs_free_decision = polaris_block_needs_free_decision(&inner.blocks[idx]);
            if block_state == PolarisBlockState::FreePending {
                return Err(EBUSY);
            }
            if needs_free_decision
                && !caller_owns_backing
                && inner.daemon_attached > 0
            {
                let _ = polaris_queue_free_decision(inner, idx)?;
                polaris_forget_static_blocks_for_block(inner, block_id)?;
                inner.block_mappings.retain(|m| m.block_id != block_id);
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                    session.block_ids.retain(|bid| *bid != block_id);
                }
            } else {
                polaris_account_direct_block_removal(
                    inner,
                    block_home_gpu,
                    block_state,
                    block_size,
                    block_had_cpu_buf,
                );
                polaris_forget_static_blocks_for_block(inner, block_id)?;
                let _ = inner.blocks.remove(idx);
                inner.block_mappings.retain(|m| m.block_id != block_id);
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                    session.block_ids.retain(|bid| *bid != block_id);
                }
            }
        }
        Ok(0)
    }

    fn polaris_handle_gpu_fault(
        &self,
        gpu_id: u32,
        rm_client_token: u64,
        va_space_token: u64,
        fault_address: u64,
        access_type: u32,
    ) -> Result<isize> {
        match polaris_resolve_gpu_fault(
            gpu_id,
            rm_client_token,
            va_space_token,
            fault_address,
            access_type,
        )? {
            PolarisUvmFaultResult::Handled => Ok(0),
            PolarisUvmFaultResult::NotMine => Err(ENOENT),
            PolarisUvmFaultResult::Error => Err(EIO),
        }
    }

    fn handle_block_touch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockTouchArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
        let touch_end = arg.token_start + arg.token_count;

        // Collect block IDs to touch (covers both directly-owned and COW-shared blocks).
        let mut bids_to_touch: KVec<u64> = KVec::new();
        if let Some(sess) = inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
            for &bid in &sess.block_ids {
                bids_to_touch.push(bid, GFP_KERNEL)?;
            }
        }
        // Also scan by session_id for directly-owned blocks not yet in block_ids list.
        for block in inner.blocks.iter() {
            if block.session_id == arg.session_id {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start < touch_end && end > arg.token_start {
                    if !bids_to_touch.iter().any(|&bid| bid == block.block_id) {
                        bids_to_touch.push(block.block_id, GFP_KERNEL)?;
                    }
                }
            }
        }

        for bid in &bids_to_touch {
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == *bid) {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start < touch_end && end > arg.token_start {
                    block.last_touch_ns = now;
                }
            }
        }
        Ok(0)
    }

    fn handle_block_get_state(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockGetStateArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        // Look up by session_id and token_start.  For COW child sessions the
        // block's session_id is the parent's, so fall back to the session's
        // block_ids list.
        let maybe_block = {
            let direct = inner
                .blocks
                .iter()
                .find(|b| b.session_id == arg.session_id && b.token_start == arg.token_start);
            if direct.is_some() {
                direct
            } else {
                let session = inner.sessions.iter().find(|s| s.session_id == arg.session_id);
                session.and_then(|sess| {
                    sess.block_ids.iter().find_map(|&bid| {
                        inner.blocks.iter()
                            .find(|b| b.block_id == bid && b.token_start == arg.token_start)
                    })
                })
            }
        };

        match maybe_block {
            Some(block) => {
                arg.block_id = block.block_id;
                arg.state = block.state as u32;
                arg.refcount = block.refcount;
                arg.gpu_vaddr = block.gpu_vaddr;
            }
            None => return Err(ENOENT),
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_get_decision(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetDecisionArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Gate: return empty list when no daemon is attached.
        if inner.daemon_attached == 0 {
            arg.count = 0;
            drop(guard);
            let mut writer = UserSlice::new(user_ptr, size).writer();
            writer.write(&arg)?;
            return Ok(0);
        }

        let count = core::cmp::min(inner.pending_decisions.len(), POLARIS_MAX_DECISIONS_PER_POLL);
        arg.count = count as u32;

        for i in 0..count {
            arg.decisions[i] = inner.pending_decisions[i];
        }
        if count < inner.pending_decisions.len() {
            let remaining = inner.pending_decisions.len() - count;
            for i in 0..remaining {
                inner.pending_decisions[i] = inner.pending_decisions[count + i];
            }
            inner.pending_decisions.truncate(remaining);
        } else {
            inner.pending_decisions.clear();
        }

        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
        for i in 0..count {
            let block_id = arg.decisions[i].block_id;
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                block.last_touch_ns = now;
            }
        }

        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_complete_operation(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisCompleteOperationArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Find the block tied to this decision by pending_decision_id.
        let block_idx = match inner
            .blocks
            .iter()
            .position(|b| b.pending_decision_id == arg.decision_id)
        {
            Some(idx) => idx,
            None => {
                dev_err!(
                    self.dev,
                    "POLARIS: COMPLETE_OPERATION for unknown decision_id {}\n",
                    arg.decision_id
                );
                return Ok(0);
            }
        };

        if inner.blocks[block_idx].pending_generation != 0
            && inner.blocks[block_idx].pending_generation != arg.generation
        {
            dev_warn!(
                self.dev,
                "POLARIS: stale COMPLETE_OPERATION decision={} generation={} active={}\n",
                arg.decision_id,
                arg.generation,
                inner.blocks[block_idx].pending_generation
            );
            return Ok(0);
        }

        // ── Success path ──
        if arg.result == 0 {
            let completed_rm_backing = polaris_complete_rm_backing(&arg)?;
            if let Some(backing) = completed_rm_backing {
                if backing.length < inner.blocks[block_idx].size_bytes {
                    return Err(EINVAL);
                }
            }
            let prev_state;
            let gpu_id;
            let sz;
            let comp_ptr: *mut bindings::completion;
            let was_cpu_offloaded: bool;
            {
                let block = &mut inner.blocks[block_idx];
                block.retry_count = 0;
                block.pending_decision_id = 0;
                block.pending_fault_id = 0;
                block.pending_generation = 0;
                prev_state = block.state;
                gpu_id = block.home_gpu;
                sz = block.size_bytes;
                was_cpu_offloaded = block.cpu_buf_addr != 0;
                match block.state {
                    PolarisBlockState::AllocPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                        polaris_apply_completed_rm_backing(block, completed_rm_backing);
                        block.map_time_ns = unsafe { bindings::ktime_get_mono_fast_ns() };
                    }
                    PolarisBlockState::OffloadPending => {
                        block.state = PolarisBlockState::CpuOffloaded;
                        block.cpu_buf_addr = arg.output_cpu_addr;
                        // Phys handle was released by the daemon during offload.
                        block.gpu_phys_handle = 0;
                        polaris_clear_block_rm_backing(block);
                    }
                    PolarisBlockState::ReloadPending | PolarisBlockState::CowPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                        polaris_apply_completed_rm_backing(block, completed_rm_backing);
                        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
                        block.map_time_ns = now;
                        block.last_touch_ns = now;
                        block.cpu_buf_addr = 0;
                    }
                    PolarisBlockState::FreePending => {
                        block.state = PolarisBlockState::Evicted;
                        polaris_clear_block_rm_backing(block);
                    }
                    _ => {}
                }
                // Track per-policy statistics.
                match prev_state {
                    PolarisBlockState::OffloadPending => {
                        inner.offload_count = inner.offload_count.saturating_add(1);
                    }
                    PolarisBlockState::ReloadPending => {
                        inner.reload_count = inner.reload_count.saturating_add(1);
                    }
                    PolarisBlockState::CowPending => {
                        inner.cow_break_count = inner.cow_break_count.saturating_add(1);
                        inner.cow_copy_bytes = inner.cow_copy_bytes.saturating_add(sz);
                    }
                    _ => {}
                }
                // Capture the completion pointer before the mutable borrow ends.
                comp_ptr = block.completion_ptr;
                block.completion_ptr = core::ptr::null_mut();
            }
            // Signal the bounded fault waiter, if this decision came from the
            // replayable GPU fault path.
            if !comp_ptr.is_null() {
                unsafe { bindings::complete(comp_ptr); }
            }

            // Update GPU and CPU pool used_bytes based on the state transition.
            let mut should_remove = false;
            let mut sid_to_clean = 0u64;
            let mut bid_to_clean = 0u64;
            match prev_state {
                PolarisBlockState::AllocPending | PolarisBlockState::CowPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes += sz;
                    }
                }
                PolarisBlockState::ReloadPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes += sz;
                        gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(sz);
                    }
                }
                PolarisBlockState::OffloadPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                        gpu.cpu_pool_used_bytes += sz;
                    }
                }
                PolarisBlockState::FreePending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        if was_cpu_offloaded {
                            gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(sz);
                        } else {
                            gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                        }
                    }
                    sid_to_clean = inner.blocks[block_idx].session_id;
                    bid_to_clean = inner.blocks[block_idx].block_id;
                    should_remove = true;
                }
                _ => {}
            }

            if should_remove {
                polaris_forget_static_blocks_for_block(inner, bid_to_clean)?;
                let _ = inner.blocks.remove(block_idx);
                inner.block_mappings.retain(|m| m.block_id != bid_to_clean);
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == sid_to_clean) {
                    session.block_ids.retain(|bid| *bid != bid_to_clean);
                }
            }
            dev_info!(
                self.dev,
                "POLARIS: operation {} completed (handle=0x{:x})\n",
                arg.decision_id,
                arg.output_handle
            );
            return Ok(0);
        }

        // ── Error handling contract (G4) ──
        // Standard Linux errno values (negative i32 from daemon).
        const E_NOMEM: i32 = -(bindings::ENOMEM as i32);
        const E_NODEV: i32 = -(bindings::ENODEV as i32);
        const E_INVAL: i32 = -(bindings::EINVAL as i32);
        const E_FAULT: i32 = -(bindings::EFAULT as i32);

        // Increment retry counter (borrow dropped before match body).
        inner.blocks[block_idx].retry_count += 1;
        let retries = inner.blocks[block_idx].retry_count;
        let block_id = inner.blocks[block_idx].block_id;
        let home_gpu = inner.blocks[block_idx].home_gpu;

        match arg.result {
            E_NOMEM => {
                dev_warn!(
                    self.dev,
                    "POLARIS: ENOMEM on decision {} (block {}, retry {}/{})\n",
                    arg.decision_id,
                    block_id,
                    retries,
                    POLARIS_MAX_RETRIES,
                );
                if retries < POLARIS_MAX_RETRIES {
                    self.requeue_decision(inner, block_idx);
                } else {
                    dev_err!(
                        self.dev,
                        "POLARIS: block {} evicted after {} ENOMEM failures\n",
                        block_id,
                        POLARIS_MAX_RETRIES,
                    );
                    inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                    polaris_clear_block_rm_backing(&mut inner.blocks[block_idx]);
                    inner.blocks[block_idx].pending_decision_id = 0;
                    inner.blocks[block_idx].pending_fault_id = 0;
                    inner.blocks[block_idx].pending_generation = 0;
                }
            }
            E_NODEV => {
                dev_err!(
                    self.dev,
                    "POLARIS: ENODEV on decision {} — marking GPU {} unhealthy\n",
                    arg.decision_id,
                    home_gpu,
                );
                if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == home_gpu) {
                    gpu.healthy = false;
                }
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                polaris_clear_block_rm_backing(&mut inner.blocks[block_idx]);
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
            E_INVAL => {
                dev_err!(
                    self.dev,
                    "POLARIS: EINVAL on decision {} — marking block {} FAILED\n",
                    arg.decision_id,
                    block_id,
                );
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                polaris_clear_block_rm_backing(&mut inner.blocks[block_idx]);
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
            E_FAULT => {
                dev_warn!(
                    self.dev,
                    "POLARIS: EFAULT on decision {} (block {}, retry {})\n",
                    arg.decision_id,
                    block_id,
                    retries,
                );
                if retries < 2 {
                    // Policy: retry once.
                    self.requeue_decision(inner, block_idx);
                } else {
                    dev_err!(
                        self.dev,
                        "POLARIS: block {} evicted after EFAULT retries\n",
                        block_id,
                    );
                    inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                    polaris_clear_block_rm_backing(&mut inner.blocks[block_idx]);
                    inner.blocks[block_idx].pending_decision_id = 0;
                    inner.blocks[block_idx].pending_fault_id = 0;
                    inner.blocks[block_idx].pending_generation = 0;
                }
            }
            other => {
                dev_err!(
                    self.dev,
                    "POLARIS: unknown failure {} on decision {} — evicting block {}\n",
                    other,
                    arg.decision_id,
                    block_id,
                );
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                polaris_clear_block_rm_backing(&mut inner.blocks[block_idx]);
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
        }

        // Signal any bounded fault waiter blocked on this completion if the block
        // reached a terminal state.  Retry paths (ENOMEM/EFAULT with retries
        // remaining) keep the AllocPending state and a new pending_decision_id —
        // the waiter should NOT be woken yet.
        if inner.blocks[block_idx].state == PolarisBlockState::Evicted {
            inner.total_evictions = inner.total_evictions.saturating_add(1);
            let comp_ptr = inner.blocks[block_idx].completion_ptr;
            inner.blocks[block_idx].completion_ptr = core::ptr::null_mut();
            if !comp_ptr.is_null() {
                unsafe { bindings::complete(comp_ptr); }
            }
        }

        Ok(0)
    }

    /// Re-queue a decision for a block (G4 retry path).
    /// Preserves the original operation type and relevant fields
    /// (phys handle for OFFLOAD/COW_BREAK, CPU addr for RELOAD).
    fn requeue_decision(&self, inner: &mut PolarisInner, block_idx: usize) {
        let block = &mut inner.blocks[block_idx];
        let dec_id = inner.next_decision_id;
        inner.next_decision_id += 1;
        block.pending_decision_id = dec_id;

        let op = match block.state {
            PolarisBlockState::AllocPending => PolarisDecisionOp::Alloc as u32,
            PolarisBlockState::OffloadPending => PolarisDecisionOp::Offload as u32,
            PolarisBlockState::ReloadPending => PolarisDecisionOp::Reload as u32,
            PolarisBlockState::CowPending => PolarisDecisionOp::CowBreak as u32,
            _ => PolarisDecisionOp::Alloc as u32,
        };

        let src_handle = match block.state {
            PolarisBlockState::OffloadPending => block.gpu_phys_handle,
            PolarisBlockState::CowPending => block.cow_src_handle,
            _ => 0,
        };

        let cpu_addr = match block.state {
            PolarisBlockState::ReloadPending => block.cpu_buf_addr,
            _ => 0,
        };

        let _ = inner.pending_decisions.push(
            PolarisDecision {
                decision_id: dec_id,
                fault_id: 0,
                generation: 0,
                op,
                gpu_id: block.home_gpu,
                block_id: block.block_id,
                session_id: block.session_id,
                src_handle,
                dst_handle: 0,
                src_vaddr: 0,
                dst_vaddr: 0,
                size_bytes: block.size_bytes,
                cpu_addr,
                access_flags: 0,
                timeout_ms: 0,
                _reserved: [0u64; 4],
            },
            GFP_KERNEL,
        );
    }

    fn handle_get_global_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetGlobalStatsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        arg.total_gpus = inner.gpus.len() as u32;
        arg.total_sessions = inner.sessions.len() as u32;
        arg.total_blocks = inner.blocks.len() as u32;

        let mut resident: u32 = 0;
        let mut offloaded: u32 = 0;
        let mut evicted: u32 = 0;
        let mut shared: u64 = 0;
        let mut private: u64 = 0;
        let mut memory_saved: u64 = 0;
        let mut total_gpu: u64 = 0;
        let mut used_gpu: u64 = 0;
        let mut cpu_total: u64 = 0;
        let mut cpu_used: u64 = 0;

        for block in &inner.blocks {
            match block.state {
                PolarisBlockState::Resident => resident += 1,
                PolarisBlockState::CpuOffloaded => offloaded += 1,
                PolarisBlockState::Evicted => evicted += 1,
                _ => {}
            }
            if block.flags.contains(PolarisBlockFlag::Shared) {
                shared += block.size_bytes;
                // Naive allocation without COW: each session that references
                // this block would need its own private copy.  The block
                // already exists as one copy, so we saved (refcount-1) copies.
                memory_saved = memory_saved.saturating_add(
                    block.refcount.saturating_sub(1).saturating_mul(block.size_bytes),
                );
            } else {
                private += block.size_bytes;
            }
        }
        for gpu in &inner.gpus {
            total_gpu += gpu.total_bytes;
            used_gpu += gpu.used_bytes;
            cpu_total += gpu.cpu_pool_total_bytes;
            cpu_used += gpu.cpu_pool_used_bytes;
        }
        let policy = inner.eviction_policy as u32;
        let offload_cnt = inner.offload_count;
        let reload_cnt = inner.reload_count;
        let evictions = inner.total_evictions;
        let cow_cnt = inner.cow_break_count;
        let cow_bytes = inner.cow_copy_bytes;
        let memory_saved_naive = memory_saved.saturating_sub(cow_bytes);
        drop(guard);

        arg.blocks_resident = resident;
        arg.blocks_offloaded = offloaded;
        arg.blocks_evicted = evicted;
        arg.shared_gpu_bytes = shared;
        arg.private_gpu_bytes = private;
        arg.cow_break_count = cow_cnt;
        arg.cow_copy_bytes = cow_bytes;
        arg.memory_saved_vs_naive = memory_saved_naive;
        arg.total_gpu_bytes = total_gpu;
        arg.used_gpu_bytes = used_gpu;
        arg.cpu_pool_total = cpu_total;
        arg.cpu_pool_used = cpu_used;
        arg.eviction_policy = policy;
        arg.offload_count = offload_cnt;
        arg.reload_count = reload_cnt;
        arg.total_evictions = evictions;

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_set_policy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisSetPolicyArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        match arg.policy {
            0 => inner.eviction_policy = PolarisEvictionPolicy::Fifo,
            1 => inner.eviction_policy = PolarisEvictionPolicy::Lru,
            2 => inner.eviction_policy = PolarisEvictionPolicy::PhaseAware,
            _ => {
                dev_err!(
                    self.dev,
                    "POLARIS: unknown eviction policy {} (valid: 0=fifo, 1=lru, 2=phase_aware)\n",
                    arg.policy
                );
                return Err(EINVAL);
            }
        }

        // Reset per-policy counters on policy switch.
        inner.offload_count = 0;
        inner.reload_count = 0;

        dev_info!(
            self.dev,
            "POLARIS: eviction policy set to {:?}\n",
            inner.eviction_policy
        );
        Ok(0)
    }

    fn handle_list_sessions(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisListSessionsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        let count = core::cmp::min(inner.sessions.len(), POLARIS_MAX_SESSIONS_PER_LIST);
        arg.count = count as u32;
        for (i, session) in inner.sessions.iter().take(count).enumerate() {
            arg.session_ids[i] = session.session_id;
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    // v4: libpolaris-shim announces a fault-capable VA-space it has just
    // created and registered with UVM. We stash (gpu_id, rm_client_token,
    // va_space_token, managed window) so the UVM fault hook can resolve incoming faults
    // back to a worker. The M2 static-block path also publishes this entry
    // into the fast fault-hook lookup table.
    fn handle_register_va_space(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterVaSpaceArg = reader.read()?;

        if arg.va_space_token == 0 || arg.managed_length == 0 {
            return Err(EINVAL);
        }
        if self.registered_v4_token.load(Relaxed) != 0 {
            return Err(EBUSY);
        }

        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        if !inner.gpus.iter().any(|g| g.gpu_id == arg.gpu_id) {
            return Err(ENOENT);
        }
        if inner.va_spaces.iter().any(|v| {
            v.gpu_id == arg.gpu_id
                && v.rm_client_token == arg.rm_client_token
                && v.va_space_token == arg.va_space_token
        }) {
            return Err(EEXIST);
        }

        let pid = polaris_current_pid();
        inner.va_spaces.push(
            PolarisVaSpace {
                gpu_id: arg.gpu_id,
                pid,
                rm_client_token: arg.rm_client_token,
                va_space_token: arg.va_space_token,
                managed_base: arg.managed_base,
                managed_length: arg.managed_length,
            },
            GFP_KERNEL,
        )?;
        polaris_fast_vaspace_register(
            arg.gpu_id,
            arg.rm_client_token,
            arg.va_space_token,
            arg.managed_base,
            arg.managed_length,
        )?;
        self.registered_v4_gpu.store(arg.gpu_id, Relaxed);
        self.registered_v4_client.store(arg.rm_client_token, Relaxed);
        self.registered_v4_token.store(arg.va_space_token, Relaxed);

        dev_info!(
            self.dev,
            "POLARIS: registered v4 VA-space gpu={} pid={} client=0x{:x} token=0x{:x} base=0x{:x} len=0x{:x}\n",
            arg.gpu_id,
            pid,
            arg.rm_client_token,
            arg.va_space_token,
            arg.managed_base,
            arg.managed_length
        );
        Ok(0)
    }

    fn handle_unregister_va_space(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisUnregisterVaSpaceArg = reader.read()?;

        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        let idx = inner.va_spaces.iter().position(|v| {
            v.gpu_id == arg.gpu_id
                && v.rm_client_token == arg.rm_client_token
                && v.va_space_token == arg.va_space_token
        }).ok_or(ENOENT)?;
        let _ = inner.va_spaces.remove(idx);
        inner.static_blocks.retain(|b| {
            !(b.gpu_id == arg.gpu_id
                && b.rm_client_token == arg.rm_client_token
                && b.va_space_token == arg.va_space_token)
        });
        inner.block_mappings.retain(|m| {
            !(m.gpu_id == arg.gpu_id
                && m.rm_client_token == arg.rm_client_token
                && m.va_space_token == arg.va_space_token)
        });
        polaris_fast_vaspace_unregister(arg.gpu_id, arg.rm_client_token, arg.va_space_token);
        polaris_fast_static_blocks_unregister_va_space(
            arg.gpu_id,
            arg.rm_client_token,
            arg.va_space_token,
        );
        if self.registered_v4_gpu.load(Relaxed) == arg.gpu_id
            && self.registered_v4_client.load(Relaxed) == arg.rm_client_token
            && self.registered_v4_token.load(Relaxed) == arg.va_space_token
        {
            self.registered_v4_gpu.store(0, Relaxed);
            self.registered_v4_client.store(0, Relaxed);
            self.registered_v4_token.store(0, Relaxed);
        }

        dev_info!(
            self.dev,
            "POLARIS: unregistered v4 VA-space gpu={} client=0x{:x} token=0x{:x}\n",
            arg.gpu_id, arg.rm_client_token, arg.va_space_token
        );
        Ok(0)
    }

    fn handle_register_static_block(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterStaticBlockArg = reader.read()?;

        if arg.va_space_token == 0 || arg.length == 0 || arg.h_memory == 0 {
            return Err(EINVAL);
        }

        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        let va_space = inner
            .va_spaces
            .iter()
            .find(|v| {
                v.gpu_id == arg.gpu_id
                    && v.rm_client_token == arg.rm_client_token
                    && v.va_space_token == arg.va_space_token
            })
            .ok_or(ENOENT)?;

        let block_end = arg.base.checked_add(arg.length).ok_or(EINVAL)?;
        let managed_end = va_space
            .managed_base
            .checked_add(va_space.managed_length)
            .ok_or(EINVAL)?;
        if arg.base < va_space.managed_base || block_end > managed_end {
            return Err(EINVAL);
        }

        if inner.static_blocks.iter().any(|b| {
            b.gpu_id == arg.gpu_id
                && b.va_space_token == arg.va_space_token
                && b.rm_client_token == arg.rm_client_token
                && b.base == arg.base
        }) {
            return Err(EEXIST);
        }

        let block = PolarisStaticBlock {
            gpu_id: arg.gpu_id,
            rm_client_token: arg.rm_client_token,
            va_space_token: arg.va_space_token,
            base: arg.base,
            length: arg.length,
            offset: arg.offset,
            rm_control_fd: arg.rm_control_fd,
            h_client: arg.h_client,
            h_memory: arg.h_memory,
        };

        inner.static_blocks.push(block, GFP_KERNEL)?;
        let registered = inner.static_blocks.last().ok_or(ENOMEM)?;
        if let Err(e) = polaris_fast_static_block_register(registered) {
            let _ = inner.static_blocks.pop();
            return Err(e);
        }

        dev_info!(
            self.dev,
            "POLARIS: registered static block gpu={} client=0x{:x} token=0x{:x} base=0x{:x} len=0x{:x} hClient=0x{:x} hMemory=0x{:x}\n",
            arg.gpu_id,
            arg.rm_client_token,
            arg.va_space_token,
            arg.base,
            arg.length,
            arg.h_client,
            arg.h_memory
        );
        Ok(0)
    }

    fn handle_unmap_static_block(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisUnmapStaticBlockArg = reader.read()?;

        if arg.va_space_token == 0 || arg.length == 0 {
            return Err(EINVAL);
        }

        {
            let guard = POLARIS_STATE.lock();
            let inner = guard.as_ref().ok_or(ENODEV)?;
            let block = inner
                .static_blocks
                .iter()
                .find(|b| {
                    b.gpu_id == arg.gpu_id
                        && b.va_space_token == arg.va_space_token
                        && b.rm_client_token == arg.rm_client_token
                        && b.base == arg.base
                })
                .ok_or(ENOENT)?;

            if arg.length > block.length {
                return Err(EINVAL);
            }
        }

        let mut gpu_va_space_ptr = 0;
        for slot in &POLARIS_FAST_STATIC_BLOCKS {
            if slot.va_space_token.load(Acquire) == arg.va_space_token
                && slot.gpu_id.load(Relaxed) == arg.gpu_id
                && slot.rm_client_token.load(Relaxed) == arg.rm_client_token
                && slot.base.load(Relaxed) == arg.base
            {
                gpu_va_space_ptr = slot.last_gpu_va_space_ptr.load(Acquire);
                break;
            }
        }

        if gpu_va_space_ptr == 0 {
            return Err(ENOENT);
        }

        polaris_uvm_unmap_external_allocation(gpu_va_space_ptr, arg.base, arg.length)?;

        for slot in &POLARIS_FAST_STATIC_BLOCKS {
            if slot.va_space_token.load(Acquire) == arg.va_space_token
                && slot.gpu_id.load(Relaxed) == arg.gpu_id
                && slot.rm_client_token.load(Relaxed) == arg.rm_client_token
                && slot.base.load(Relaxed) == arg.base
            {
                slot.last_gpu_va_space_ptr.store(0, Release);
                break;
            }
        }

        dev_info!(
            self.dev,
            "POLARIS: unmapped static block gpu={} client=0x{:x} token=0x{:x} base=0x{:x} len=0x{:x}\n",
            arg.gpu_id,
            arg.rm_client_token,
            arg.va_space_token,
            arg.base,
            arg.length
        );
        Ok(0)
    }

    fn handle_register_block_mapping(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterBlockMappingArg = reader.read()?;

        if arg.block_id == 0 || arg.va_space_token == 0 || arg.length == 0 {
            return Err(EINVAL);
        }

        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        let va_space = inner
            .va_spaces
            .iter()
            .find(|v| {
                v.gpu_id == arg.gpu_id
                    && v.rm_client_token == arg.rm_client_token
                    && v.va_space_token == arg.va_space_token
            })
            .ok_or(ENOENT)?;

        let block = inner
            .blocks
            .iter()
            .find(|b| b.block_id == arg.block_id)
            .ok_or(ENOENT)?;
        if block.home_gpu != arg.gpu_id {
            return Err(EINVAL);
        }

        let mapping_end = arg.base.checked_add(arg.length).ok_or(EINVAL)?;
        let managed_end = va_space
            .managed_base
            .checked_add(va_space.managed_length)
            .ok_or(EINVAL)?;
        if arg.base < va_space.managed_base || mapping_end > managed_end {
            return Err(EINVAL);
        }

        if let Some(mapping) = inner.block_mappings.iter_mut().find(|m| {
            m.block_id == arg.block_id
                && m.gpu_id == arg.gpu_id
                && m.rm_client_token == arg.rm_client_token
                && m.va_space_token == arg.va_space_token
                && m.base == arg.base
        }) {
            mapping.length = arg.length;
            mapping.last_gpu_va_space_ptr = 0;
        } else {
            inner.block_mappings.push(
                PolarisBlockMapping {
                    block_id: arg.block_id,
                    gpu_id: arg.gpu_id,
                    rm_client_token: arg.rm_client_token,
                    va_space_token: arg.va_space_token,
                    base: arg.base,
                    length: arg.length,
                    last_gpu_va_space_ptr: 0,
                },
                GFP_KERNEL,
            )?;
        }

        dev_info!(
            self.dev,
            "POLARIS: registered block mapping block={} gpu={} client=0x{:x} token=0x{:x} base=0x{:x} len=0x{:x}\n",
            arg.block_id,
            arg.gpu_id,
            arg.rm_client_token,
            arg.va_space_token,
            arg.base,
            arg.length
        );
        Ok(0)
    }

    fn handle_register_block_backing(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterBlockBackingArg = reader.read()?;

        if arg.block_id == 0 || arg.length == 0 || arg.h_client == 0 || arg.h_memory == 0 {
            return Err(EINVAL);
        }

        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let block = inner
            .blocks
            .iter_mut()
            .find(|b| b.block_id == arg.block_id)
            .ok_or(ENOENT)?;

        if block.home_gpu != arg.gpu_id || arg.length < block.size_bytes {
            return Err(EINVAL);
        }
        if matches!(
            block.state,
            PolarisBlockState::FreePending
                | PolarisBlockState::OffloadPending
                | PolarisBlockState::ReloadPending
                | PolarisBlockState::CowPending
                | PolarisBlockState::Evicted
        ) {
            return Err(EBUSY);
        }

        block.rm_control_fd = arg.rm_control_fd;
        block.rm_h_client = arg.h_client;
        block.rm_h_memory = arg.h_memory;
        block.rm_backing_length = arg.length;
        block.rm_backing_offset = arg.offset;

        dev_info!(
            self.dev,
            "POLARIS: registered block backing block={} gpu={} len=0x{:x} hClient=0x{:x} hMemory=0x{:x}\n",
            arg.block_id,
            arg.gpu_id,
            arg.length,
            arg.h_client,
            arg.h_memory
        );
        Ok(0)
    }

    fn handle_probe_rm_phys(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisProbeRmPhysArg = reader.read()?;

        if arg.block_id == 0 {
            return Err(EINVAL);
        }

        let target = {
            let guard = POLARIS_STATE.lock();
            let inner = guard.as_ref().ok_or(ENODEV)?;
            polaris_snapshot_rm_phys_probe_target(inner, arg.block_id)?
        };

        let query_length = if arg.length == 0 {
            target.length
        } else {
            arg.length
        };
        if query_length == 0 || arg.offset >= target.length || query_length > target.length.saturating_sub(arg.offset) {
            return Err(EINVAL);
        }

        let mut page_size = 0u64;
        let mut phys_addr_count = 0u64;
        let mut first_phys_addr = 0u64;
        let mut last_phys_addr = 0u64;
        let mut contiguous = 0u64;
        let mut sysmem = 0u64;
        let mut egm = 0u64;
        let mut fabricmem = 0u64;

        let ret = unsafe {
            uvm_polaris_probe_external_allocation(
                target.gpu_va_space_ptr,
                arg.offset,
                query_length,
                target.rm_control_fd,
                target.h_client,
                target.h_memory,
                &mut page_size,
                &mut phys_addr_count,
                &mut first_phys_addr,
                &mut last_phys_addr,
                &mut contiguous,
                &mut sysmem,
                &mut egm,
                &mut fabricmem,
            )
        };
        if ret != 0 {
            return Err(Error::from_errno(-ret));
        }

        arg.length = query_length;
        arg.page_size = page_size;
        arg.phys_addr_count = phys_addr_count;
        arg.first_phys_addr = first_phys_addr;
        arg.last_phys_addr = last_phys_addr;
        arg.flags = (contiguous & 1)
            | ((sysmem & 1) << 1)
            | ((egm & 1) << 2)
            | ((fabricmem & 1) << 3);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        dev_info!(
            self.dev,
            "POLARIS: probed RM phys block={} gpu={} page=0x{:x} count={} first=0x{:x} last=0x{:x} flags=0x{:x}\n",
            target.block_id,
            target.gpu_id,
            arg.page_size,
            arg.phys_addr_count,
            arg.first_phys_addr,
            arg.last_phys_addr,
            arg.flags
        );
        Ok(0)
    }

    fn handle_probe_rm_copy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisProbeRmCopyArg = reader.read()?;

        if arg.block_id == 0 {
            return Err(EINVAL);
        }

        let target = {
            let guard = POLARIS_STATE.lock();
            let inner = guard.as_ref().ok_or(ENODEV)?;
            polaris_snapshot_rm_phys_probe_target(inner, arg.block_id)?
        };

        let query_length = if arg.length == 0 {
            target.length
        } else {
            arg.length
        };
        if query_length == 0 || arg.offset >= target.length || query_length > target.length.saturating_sub(arg.offset) {
            return Err(EINVAL);
        }

        let mut page_size = 0u64;
        let mut phys_addr_count = 0u64;
        let mut first_phys_addr = 0u64;
        let mut last_phys_addr = 0u64;
        let mut flags = 0u64;
        let mut bytes_checked = 0u64;
        let mut first_mismatch_offset = POLARIS_RM_COPY_NO_MISMATCH;
        let mut expected_byte = 0u64;
        let mut actual_byte = 0u64;

        let ret = unsafe {
            uvm_polaris_probe_external_copy(
                target.gpu_va_space_ptr,
                arg.offset,
                query_length,
                target.rm_control_fd,
                target.h_client,
                target.h_memory,
                arg.pattern_seed,
                &mut page_size,
                &mut phys_addr_count,
                &mut first_phys_addr,
                &mut last_phys_addr,
                &mut flags,
                &mut bytes_checked,
                &mut first_mismatch_offset,
                &mut expected_byte,
                &mut actual_byte,
            )
        };
        if ret != 0 {
            return Err(Error::from_errno(-ret));
        }

        arg.length = query_length;
        arg.page_size = page_size;
        arg.phys_addr_count = phys_addr_count;
        arg.first_phys_addr = first_phys_addr;
        arg.last_phys_addr = last_phys_addr;
        arg.flags = flags;
        arg.bytes_checked = bytes_checked;
        arg.first_mismatch_offset = first_mismatch_offset;
        arg.expected_byte = expected_byte;
        arg.actual_byte = actual_byte;

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        dev_info!(
            self.dev,
            "POLARIS: probed RM copy block={} gpu={} len=0x{:x} page=0x{:x} count={} first=0x{:x} last=0x{:x} mismatch=0x{:x}\n",
            target.block_id,
            target.gpu_id,
            arg.length,
            arg.page_size,
            arg.phys_addr_count,
            arg.first_phys_addr,
            arg.last_phys_addr,
            arg.first_mismatch_offset
        );
        Ok(0)
    }

    fn handle_rm_copy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisRmCopyArg = reader.read()?;

        if arg.block_id == 0 || arg.user_cpu_addr == 0 {
            return Err(EINVAL);
        }
        if arg.direction != POLARIS_RM_COPY_TO_CPU && arg.direction != POLARIS_RM_COPY_FROM_CPU {
            return Err(EINVAL);
        }

        let target = {
            let guard = POLARIS_STATE.lock();
            let inner = guard.as_ref().ok_or(ENODEV)?;
            polaris_snapshot_rm_phys_probe_target(inner, arg.block_id)?
        };

        let query_length = if arg.length == 0 {
            target.length
        } else {
            arg.length
        };
        if query_length == 0 || arg.offset >= target.length || query_length > target.length.saturating_sub(arg.offset) {
            return Err(EINVAL);
        }

        let mut page_size = 0u64;
        let mut phys_addr_count = 0u64;
        let mut first_phys_addr = 0u64;
        let mut last_phys_addr = 0u64;
        let mut flags = 0u64;
        let mut bytes_copied = 0u64;

        let ret = unsafe {
            uvm_polaris_copy_external_allocation(
                target.gpu_va_space_ptr,
                arg.offset,
                query_length,
                target.rm_control_fd,
                target.h_client,
                target.h_memory,
                arg.user_cpu_addr,
                arg.direction,
                &mut page_size,
                &mut phys_addr_count,
                &mut first_phys_addr,
                &mut last_phys_addr,
                &mut flags,
                &mut bytes_copied,
            )
        };
        if ret != 0 {
            return Err(Error::from_errno(-ret));
        }

        arg.length = query_length;
        arg.page_size = page_size;
        arg.phys_addr_count = phys_addr_count;
        arg.first_phys_addr = first_phys_addr;
        arg.last_phys_addr = last_phys_addr;
        arg.flags = flags;
        arg.bytes_copied = bytes_copied;

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        dev_info!(
            self.dev,
            "POLARIS: RM copy block={} gpu={} dir={} bytes=0x{:x} page=0x{:x} count={} flags=0x{:x}\n",
            target.block_id,
            target.gpu_id,
            arg.direction,
            arg.bytes_copied,
            arg.page_size,
            arg.phys_addr_count,
            arg.flags
        );
        Ok(0)
    }

    fn handle_unmap_block_mappings(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisUnmapBlockMappingsArg = reader.read()?;

        if arg.block_id == 0 {
            return Err(EINVAL);
        }

        arg.unmapped_count = polaris_unmap_observed_block_mappings(arg.block_id)?;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        dev_info!(
            self.dev,
            "POLARIS: unmapped {} mapping(s) for block {}\n",
            arg.unmapped_count,
            arg.block_id
        );
        Ok(0)
    }

    fn handle_spill_block(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSpillBlockArg = reader.read()?;

        if arg.block_id == 0 {
            return Err(EINVAL);
        }

        let decision_id = {
            let mut guard = POLARIS_STATE.lock();
            let inner = guard.as_mut().ok_or(ENODEV)?;

            let block_idx = inner
                .blocks
                .iter()
                .position(|b| b.block_id == arg.block_id)
                .ok_or(ENOENT)?;

            match inner.blocks[block_idx].state {
                PolarisBlockState::CpuOffloaded => {
                    0
                }
                PolarisBlockState::Resident => {
                    polaris_queue_offload_decision(inner, block_idx)?
                }
                PolarisBlockState::AllocPending
                | PolarisBlockState::OffloadPending
                | PolarisBlockState::ReloadPending
                | PolarisBlockState::CowPending
                | PolarisBlockState::FreePending => {
                    return Err(EBUSY);
                }
                PolarisBlockState::Unmapped | PolarisBlockState::Evicted => {
                    return Err(ENOENT);
                }
            }
        };

        let unmapped_count = polaris_unmap_observed_block_mappings(arg.block_id)?;

        arg.unmapped_count = unmapped_count;
        arg.decision_id = decision_id;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        dev_info!(
            self.dev,
            "POLARIS: queued spill for block {} as decision {} after unmapping {} mapping(s)\n",
            arg.block_id,
            arg.decision_id,
            arg.unmapped_count
        );
        Ok(0)
    }
}
