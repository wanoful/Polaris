# SPDX-License-Identifier: Apache-2.0
"""
POLARIS ioctl ABI — Python ctypes bindings.

Auto-generated from kernel/polaris_abi.rs.  These structs must match the
kernel-side definitions exactly.  When the ABI changes, update this file
accordingly.
"""

import ctypes
import fcntl
import os
from typing import Optional

# ─── Magic and ioctl helpers ────────────────────────────────────────────────

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2
_IOC_READWRITE = 3


def _ioc(dir_: int, type_: int, nr: int, size: int) -> int:
    return (dir_ << _IOC_DIRSHIFT) | (size << _IOC_SIZESHIFT) | (type_ << _IOC_TYPESHIFT) | nr


def _io(type_: int, nr: int) -> int:
    return _ioc(_IOC_NONE, type_, nr, 0)


def _iow(type_: int, nr: int, size: int) -> int:
    return _ioc(_IOC_WRITE, type_, nr, size)


def _ior(type_: int, nr: int, size: int) -> int:
    return _ioc(_IOC_READ, type_, nr, size)


def _iowr(type_: int, nr: int, size: int) -> int:
    return _ioc(_IOC_READWRITE, type_, nr, size)


MAGIC = ord("P")  # 0x50

# ─── C Structs ──────────────────────────────────────────────────────────────


class PolarisRegisterGpuArg(ctypes.Structure):
    _fields_ = [
        ("gpu_id", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint64),
        ("budget_bytes", ctypes.c_uint64),
        ("cpu_pool_bytes", ctypes.c_uint64),
        ("numa_node", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


class PolarisSessionCreateArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("home_gpu", ctypes.c_uint32),
        ("beam_width", ctypes.c_uint32),
        ("gpu_vas_bytes", ctypes.c_uint64),
        ("bytes_per_token", ctypes.c_uint64),
        ("priority", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


class PolarisSessionDestroyArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("_reserved", ctypes.c_uint64 * 4),
    ]


class PolarisSessionGetStatsArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("home_gpu", ctypes.c_uint32),
        ("beam_width", ctypes.c_uint32),
        ("num_blocks", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint64),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


class PolarisSessionBranchArg(ctypes.Structure):
    _fields_ = [
        ("parent_session_id", ctypes.c_uint64),
        ("child_session_id", ctypes.c_uint64),
        ("_reserved", ctypes.c_uint64 * 4),
    ]


class PolarisBlockGrowArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("token_start", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("block_id", ctypes.c_uint64),
        ("ret_code", ctypes.c_int32),
        ("_reserved", ctypes.c_uint32),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


class PolarisBlockFreeArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("token_start", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint64 * 4),
    ]


class PolarisBlockTouchArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("token_start", ctypes.c_uint64),
        ("token_count", ctypes.c_uint64),
        ("_reserved", ctypes.c_uint64 * 4),
    ]


class PolarisBlockGetStateArg(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_uint64),
        ("token_start", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("block_id", ctypes.c_uint64),
        ("state", ctypes.c_uint32),
        ("refcount", ctypes.c_uint64),
        ("gpu_vaddr", ctypes.c_uint64),
        ("_reserved", ctypes.c_uint64 * 2),
    ]


class PolarisDecision(ctypes.Structure):
    _fields_ = [
        ("decision_id", ctypes.c_uint64),
        ("op", ctypes.c_uint32),
        ("gpu_id", ctypes.c_uint32),
        ("block_id", ctypes.c_uint64),
        ("session_id", ctypes.c_uint64),
        ("src_handle", ctypes.c_uint64),
        ("dst_vaddr", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("cpu_addr", ctypes.c_uint64),
        ("_reserved", ctypes.c_uint64 * 4),
    ]


POLARIS_MAX_DECISIONS_PER_POLL = 16


class PolarisGetDecisionArg(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("decisions", PolarisDecision * POLARIS_MAX_DECISIONS_PER_POLL),
    ]


class PolarisCompleteOperationArg(ctypes.Structure):
    _fields_ = [
        ("decision_id", ctypes.c_uint64),
        ("result", ctypes.c_int32),
        ("_reserved", ctypes.c_uint32),
        ("output_handle", ctypes.c_uint64),
        ("output_cpu_addr", ctypes.c_uint64),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


class PolarisGetGlobalStatsArg(ctypes.Structure):
    _fields_ = [
        ("total_gpus", ctypes.c_uint32),
        ("total_sessions", ctypes.c_uint32),
        ("total_blocks", ctypes.c_uint32),
        ("blocks_resident", ctypes.c_uint32),
        ("blocks_offloaded", ctypes.c_uint32),
        ("blocks_evicted", ctypes.c_uint32),
        ("shared_gpu_bytes", ctypes.c_uint64),
        ("private_gpu_bytes", ctypes.c_uint64),
        ("cow_break_count", ctypes.c_uint64),
        ("cow_copy_bytes", ctypes.c_uint64),
        ("memory_saved_vs_naive", ctypes.c_uint64),
        ("total_gpu_bytes", ctypes.c_uint64),
        ("used_gpu_bytes", ctypes.c_uint64),
        ("cpu_pool_total", ctypes.c_uint64),
        ("cpu_pool_used", ctypes.c_uint64),
        ("eviction_policy", ctypes.c_uint32),
        ("_policy_pad", ctypes.c_uint32),
        ("offload_count", ctypes.c_uint64),
        ("reload_count", ctypes.c_uint64),
        ("total_evictions", ctypes.c_uint64),
    ]


class PolarisSetPolicyArg(ctypes.Structure):
    _fields_ = [
        ("policy", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("_reserved2", ctypes.c_uint64 * 2),
    ]


POLARIS_MAX_SESSIONS_PER_LIST = 64


class PolarisListSessionsArg(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("session_ids", ctypes.c_uint64 * POLARIS_MAX_SESSIONS_PER_LIST),
    ]


# ─── Ioctl Command Codes ────────────────────────────────────────────────────

POLARIS_REGISTER_GPU = _iow(MAGIC, 0x01, ctypes.sizeof(PolarisRegisterGpuArg))
POLARIS_SESSION_CREATE = _iowr(MAGIC, 0x02, ctypes.sizeof(PolarisSessionCreateArg))
POLARIS_SESSION_DESTROY = _iow(MAGIC, 0x03, ctypes.sizeof(PolarisSessionDestroyArg))
POLARIS_SESSION_GET_STATS = _iowr(MAGIC, 0x04, ctypes.sizeof(PolarisSessionGetStatsArg))
POLARIS_SESSION_BRANCH = _iowr(MAGIC, 0x05, ctypes.sizeof(PolarisSessionBranchArg))
POLARIS_BLOCK_GROW = _iowr(MAGIC, 0x06, ctypes.sizeof(PolarisBlockGrowArg))
POLARIS_BLOCK_FREE = _iow(MAGIC, 0x07, ctypes.sizeof(PolarisBlockFreeArg))
POLARIS_BLOCK_TOUCH = _iow(MAGIC, 0x08, ctypes.sizeof(PolarisBlockTouchArg))
POLARIS_BLOCK_GET_STATE = _iowr(MAGIC, 0x09, ctypes.sizeof(PolarisBlockGetStateArg))
POLARIS_GET_DECISION = _iowr(MAGIC, 0x0A, ctypes.sizeof(PolarisGetDecisionArg))
POLARIS_COMPLETE_OPERATION = _iowr(MAGIC, 0x0B, ctypes.sizeof(PolarisCompleteOperationArg))
POLARIS_GET_GLOBAL_STATS = _iowr(MAGIC, 0x0C, ctypes.sizeof(PolarisGetGlobalStatsArg))
POLARIS_LIST_SESSIONS = _iowr(MAGIC, 0x0D, ctypes.sizeof(PolarisListSessionsArg))
POLARIS_SET_POLICY = _iow(MAGIC, 0x0E, ctypes.sizeof(PolarisSetPolicyArg))

# ─── Flag Constants ─────────────────────────────────────────────────────────

POLARIS_BLOCK_FLAG_SHARED = 1 << 0
POLARIS_GROW_FLAG_OVERWRITE = 1 << 0

# ─── Phase Constants ────────────────────────────────────────────────────────

POLARIS_PHASE_PREFILL = 1
POLARIS_PHASE_DECODE = 2

# ─── Convenience Wrappers ───────────────────────────────────────────────────


def _ioctl(fd: int, cmd: int, arg: ctypes.Structure) -> None:
    """Issue an ioctl, raising OSError on failure."""
    ret = fcntl.ioctl(fd, cmd, arg)
    if ret < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def polaris_session_create(
    fd: int,
    *,
    home_gpu: int = 0,
    beam_width: int = 1,
    gpu_vas_bytes: int = 16 * 1024 * 1024 * 1024,  # 16 GiB default
    bytes_per_token: int = 524_288,
    priority: int = 0,
) -> int:
    """Create a POLARIS session.  Returns the kernel-assigned session_id."""
    arg = PolarisSessionCreateArg()
    arg.home_gpu = home_gpu
    arg.beam_width = beam_width
    arg.gpu_vas_bytes = gpu_vas_bytes
    arg.bytes_per_token = bytes_per_token
    arg.priority = priority
    _ioctl(fd, POLARIS_SESSION_CREATE, arg)
    return arg.session_id


def polaris_session_destroy(fd: int, session_id: int) -> None:
    """Destroy a POLARIS session and free all its blocks."""
    arg = PolarisSessionDestroyArg()
    arg.session_id = session_id
    _ioctl(fd, POLARIS_SESSION_DESTROY, arg)


def polaris_session_branch(fd: int, parent_session_id: int) -> int:
    """Branch (COW fork) a session.  Returns the child session_id."""
    arg = PolarisSessionBranchArg()
    arg.parent_session_id = parent_session_id
    _ioctl(fd, POLARIS_SESSION_BRANCH, arg)
    return arg.child_session_id


def polaris_block_grow(
    fd: int,
    session_id: int,
    token_start: int,
    token_count: int,
    *,
    flags: int = 0,
    phase: int = POLARIS_PHASE_DECODE,
) -> int:
    """
    Request a new KV block from POLARIS (page-fault entry).
    Returns the kernel-assigned block_id.
    """
    arg = PolarisBlockGrowArg()
    arg.session_id = session_id
    arg.token_start = token_start
    arg.token_count = token_count
    arg.flags = flags
    arg.phase = phase
    _ioctl(fd, POLARIS_BLOCK_GROW, arg)
    if arg.ret_code != 0:
        raise OSError(-arg.ret_code, f"POLARIS_BLOCK_GROW failed: ret_code={arg.ret_code}")
    return arg.block_id


def polaris_block_free(
    fd: int,
    session_id: int,
    token_start: int,
    token_count: int,
) -> None:
    """Free a token range in a POLARIS session."""
    arg = PolarisBlockFreeArg()
    arg.session_id = session_id
    arg.token_start = token_start
    arg.token_count = token_count
    _ioctl(fd, POLARIS_BLOCK_FREE, arg)


def polaris_block_touch(
    fd: int,
    session_id: int,
    token_start: int,
    token_count: int,
) -> None:
    """Update LRU timestamp for a token range."""
    arg = PolarisBlockTouchArg()
    arg.session_id = session_id
    arg.token_start = token_start
    arg.token_count = token_count
    _ioctl(fd, POLARIS_BLOCK_TOUCH, arg)


def polaris_get_global_stats(fd: int) -> PolarisGetGlobalStatsArg:
    """Fetch global POLARIS statistics."""
    arg = PolarisGetGlobalStatsArg()
    _ioctl(fd, POLARIS_GET_GLOBAL_STATS, arg)
    return arg
