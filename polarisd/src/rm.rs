use libc::{c_ulong, c_void};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::mem;
use std::os::fd::AsRawFd;

const NV_IOCTL_MAGIC: u8 = b'F';
const NV_ESC_RM_ALLOC: u32 = 0x2b;
const NV_ESC_RM_FREE: u32 = 0x29;

const NV_ERR_GENERIC: u32 = 0x0000_ffff;
const NV_STATUS_WARN_BIT: u32 = 0x0001_0000;

const NV01_NULL_OBJECT: u32 = 0;
const NV01_ROOT_CLIENT: u32 = 0x41;
const NV01_DEVICE_0: u32 = 0x80;
const NV20_SUBDEVICE_0: u32 = 0x2080;
const NV01_MEMORY_LOCAL_USER: u32 = 0x40;

const NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES: i32 = 0x2;
const NVOS32_TYPE_IMAGE: u32 = 0;
const NVOS32_ATTR_LOCATION_VIDMEM: u32 = 0;
const NVOS32_ATTR_PHYSICALITY_CONTIGUOUS: u32 = 0x2;
const NVOS32_ATTR_PHYSICALITY_SHIFT: u32 = 27;
const NVOS32_ALLOC_FLAGS_USE_BEGIN_END: u32 = 0x0002_0000;
const DEFAULT_PRESSURE_RANGE_BASE: u64 = 1u64 << 40;

const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

fn ioc(dir: u32, ty: u32, nr: u32, size: u32) -> c_ulong {
    ((dir << 30) | (size << 16) | (ty << 8) | nr) as c_ulong
}

fn nv_ioctl_cmd(esc: u32, size: usize) -> c_ulong {
    ioc(
        IOC_READ | IOC_WRITE,
        NV_IOCTL_MAGIC as u32,
        esc,
        size as u32,
    )
}

fn nv_status_ok(status: u32) -> bool {
    status == 0 || (status != NV_ERR_GENERIC && (status & NV_STATUS_WARN_BIT) != 0)
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let lower = v.to_ascii_lowercase();
            !(lower.is_empty() || lower == "0" || lower == "false" || lower == "no")
        })
        .unwrap_or(false)
}

fn env_u64(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<u64>().ok()
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvOs00Parameters {
    h_root: u32,
    h_object_parent: u32,
    h_object_old: u32,
    status: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvOs21Parameters {
    h_root: u32,
    h_object_parent: u32,
    h_object_new: u32,
    h_class: u32,
    p_alloc_parms: u64,
    params_size: u32,
    status: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Nv0000AllocParameters {
    h_client: u32,
    process_id: u32,
    process_name: [u8; 100],
    p_os_pid_info: u64,
}

impl Default for Nv0000AllocParameters {
    fn default() -> Self {
        Self {
            h_client: 0,
            process_id: 0,
            process_name: [0; 100],
            p_os_pid_info: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Nv0080AllocParameters {
    device_id: u32,
    h_client_share: u32,
    h_target_client: u32,
    h_target_device: u32,
    flags: i32,
    va_space_size: u64,
    va_start_internal: u64,
    va_limit_internal: u64,
    va_mode: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Nv2080AllocParameters {
    sub_device_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvMemoryAllocationParameters {
    owner: u32,
    type_: u32,
    flags: u32,
    width: u32,
    height: u32,
    pitch: i32,
    attr: u32,
    attr2: u32,
    format: u32,
    compr_covg: u32,
    zcull_covg: u32,
    range_lo: u64,
    range_hi: u64,
    size: u64,
    alignment: u64,
    offset: u64,
    limit: u64,
    address: u64,
    ctag_offset: u32,
    h_va_space: u32,
    internal_flags: u32,
    tag: u32,
    numa_node: i32,
    _padding: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmAllocation {
    pub h_memory: u32,
    pub size: u64,
    /// Physical framebuffer offset returned by NVOS32 alloc (params.offset).
    pub phys_fb_addr: u64,
}

pub struct RmBackend {
    rm_control: File,
    _gpu: File,
    pub h_client: u32,
    h_device: u32,
    h_subdevice: u32,
    force_contiguous: bool,
    pressure_outside_fb_range: bool,
    pressure_range_base: u64,
    allocations: HashMap<u64, RmAllocation>,
}

impl RmBackend {
    pub fn new(cuda_ordinal: i32) -> Result<Self, String> {
        let rm_control = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/nvidiactl")
            .map_err(|e| format!("open /dev/nvidiactl failed: {e}"))?;
        let gpu_path = format!("/dev/nvidia{cuda_ordinal}");
        let gpu = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&gpu_path)
            .map_err(|e| format!("open {gpu_path} failed: {e}"))?;

        let fd = rm_control.as_raw_fd();
        let mut root_params = Nv0000AllocParameters::default();
        copy_cstr(&mut root_params.process_name, b"polarisd");
        let mut h_client = 0;
        rm_alloc(
            fd,
            NV01_NULL_OBJECT,
            NV01_NULL_OBJECT,
            &mut h_client,
            NV01_ROOT_CLIENT,
            &mut root_params,
            "RM_ALLOC root client",
        )?;
        if root_params.h_client != 0 {
            h_client = root_params.h_client;
        }

        let mut device_params = Nv0080AllocParameters {
            device_id: cuda_ordinal as u32,
            h_client_share: h_client,
            va_mode: NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES,
            ..Default::default()
        };
        let mut h_device = 0;
        rm_alloc(
            fd,
            h_client,
            h_client,
            &mut h_device,
            NV01_DEVICE_0,
            &mut device_params,
            "RM_ALLOC device",
        )?;

        let mut subdevice_params = Nv2080AllocParameters::default();
        let mut h_subdevice = 0;
        rm_alloc(
            fd,
            h_client,
            h_device,
            &mut h_subdevice,
            NV20_SUBDEVICE_0,
            &mut subdevice_params,
            "RM_ALLOC subdevice",
        )?;

        let force_contiguous = env_enabled("POLARISD_RM_FORCE_CONTIGUOUS");
        if force_contiguous {
            eprintln!(
                "polarisd: RM allocations forced contiguous by POLARISD_RM_FORCE_CONTIGUOUS"
            );
        }
        let pressure_outside_fb_range = env_enabled("POLARISD_RM_PRESSURE_OUTSIDE_FB_RANGE");
        let pressure_range_base =
            env_u64("POLARISD_RM_PRESSURE_RANGE_BASE_BYTES").unwrap_or(DEFAULT_PRESSURE_RANGE_BASE);
        if pressure_outside_fb_range {
            eprintln!(
                "polarisd: RM allocations constrained outside FB by POLARISD_RM_PRESSURE_OUTSIDE_FB_RANGE base=0x{pressure_range_base:x}"
            );
        }

        Ok(Self {
            rm_control,
            _gpu: gpu,
            h_client,
            h_device,
            h_subdevice,
            force_contiguous,
            pressure_outside_fb_range,
            pressure_range_base,
            allocations: HashMap::new(),
        })
    }

    pub fn rm_control_fd(&self) -> i32 {
        self.rm_control.as_raw_fd()
    }

    pub fn alloc_for_block(&mut self, block_id: u64, size: u64) -> Result<RmAllocation, String> {
        if let Some(allocation) = self.allocations.get(&block_id).copied() {
            return Ok(allocation);
        }

        let mut params = NvMemoryAllocationParameters {
            owner: self.h_client,
            type_: NVOS32_TYPE_IMAGE,
            attr: NVOS32_ATTR_LOCATION_VIDMEM << 25,
            size,
            ..Default::default()
        };
        if self.force_contiguous {
            params.attr |= NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << NVOS32_ATTR_PHYSICALITY_SHIFT;
        }
        if self.pressure_outside_fb_range {
            let range_hi = self
                .pressure_range_base
                .checked_add(size)
                .and_then(|end| end.checked_sub(1))
                .ok_or_else(|| {
                    format!(
                        "RM pressure range overflow base=0x{:x} size=0x{:x}",
                        self.pressure_range_base, size
                    )
                })?;
            params.flags |= NVOS32_ALLOC_FLAGS_USE_BEGIN_END;
            params.range_lo = self.pressure_range_base;
            params.range_hi = range_hi;
            params.attr |= NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << NVOS32_ATTR_PHYSICALITY_SHIFT;
        }
        let mut h_memory = 0;
        rm_alloc(
            self.rm_control.as_raw_fd(),
            self.h_client,
            self.h_device,
            &mut h_memory,
            NV01_MEMORY_LOCAL_USER,
            &mut params,
            "RM_ALLOC daemon RM memory",
        )?;

        let allocation = RmAllocation {
            h_memory,
            size: params.size,
            phys_fb_addr: params.offset,
        };
        self.allocations.insert(block_id, allocation);
        eprintln!(
            "polarisd: RM ALLOC block {} hClient=0x{:x} hMemory=0x{:x} size=0x{:x} phys_fb=0x{:x}",
            block_id, self.h_client, allocation.h_memory, allocation.size, allocation.phys_fb_addr
        );
        Ok(allocation)
    }

    pub fn free_block(&mut self, block_id: u64) -> Result<(), String> {
        let Some(allocation) = self.allocations.get(&block_id).copied() else {
            return Ok(());
        };
        self.free_allocation(allocation)?;
        self.allocations.remove(&block_id);
        eprintln!(
            "polarisd: RM FREE block {} hMemory=0x{:x}",
            block_id, allocation.h_memory
        );
        Ok(())
    }

    pub fn has_block(&self, block_id: u64) -> bool {
        self.allocations.contains_key(&block_id)
    }

    fn free_allocation(&self, allocation: RmAllocation) -> Result<(), String> {
        rm_free(
            self.rm_control.as_raw_fd(),
            self.h_client,
            self.h_device,
            allocation.h_memory,
            "RM_FREE daemon RM memory",
        )
    }
}

impl Drop for RmBackend {
    fn drop(&mut self) {
        let allocations: Vec<_> = self.allocations.drain().map(|(_, alloc)| alloc).collect();
        for allocation in allocations {
            let _ = rm_free(
                self.rm_control.as_raw_fd(),
                self.h_client,
                self.h_device,
                allocation.h_memory,
                "RM_FREE daemon RM memory",
            );
        }
        let fd = self.rm_control.as_raw_fd();
        let _ = rm_free(
            fd,
            self.h_client,
            self.h_device,
            self.h_subdevice,
            "RM_FREE subdevice",
        );
        let _ = rm_free(
            fd,
            self.h_client,
            self.h_client,
            self.h_device,
            "RM_FREE device",
        );
        let _ = rm_free(
            fd,
            self.h_client,
            self.h_client,
            self.h_client,
            "RM_FREE root client",
        );
        self.h_subdevice = 0;
        self.h_device = 0;
        self.h_client = 0;
    }
}

fn copy_cstr(dst: &mut [u8], src: &[u8]) {
    let len = dst.len().saturating_sub(1).min(src.len());
    dst[..len].copy_from_slice(&src[..len]);
}

fn rm_alloc<T>(
    fd: i32,
    h_root: u32,
    h_parent: u32,
    h_new: &mut u32,
    h_class: u32,
    params: &mut T,
    what: &str,
) -> Result<(), String> {
    let mut alloc = NvOs21Parameters {
        h_root,
        h_object_parent: h_parent,
        h_object_new: *h_new,
        h_class,
        p_alloc_parms: params as *mut T as u64,
        params_size: mem::size_of::<T>() as u32,
        status: 0,
    };
    let cmd = nv_ioctl_cmd(NV_ESC_RM_ALLOC, mem::size_of::<NvOs21Parameters>());
    let ret = unsafe { libc::ioctl(fd, cmd, &mut alloc as *mut _ as *mut c_void) };
    if ret != 0 {
        return Err(format!(
            "{what} ioctl failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    if !nv_status_ok(alloc.status) {
        return Err(format!("{what} RM status=0x{:x}", alloc.status));
    }
    *h_new = alloc.h_object_new;
    Ok(())
}

fn rm_free(fd: i32, h_root: u32, h_parent: u32, h_object: u32, what: &str) -> Result<(), String> {
    if fd < 0 || h_root == 0 || h_object == 0 {
        return Ok(());
    }
    let mut free = NvOs00Parameters {
        h_root,
        h_object_parent: h_parent,
        h_object_old: h_object,
        status: 0,
    };
    let cmd = nv_ioctl_cmd(NV_ESC_RM_FREE, mem::size_of::<NvOs00Parameters>());
    let ret = unsafe { libc::ioctl(fd, cmd, &mut free as *mut _ as *mut c_void) };
    if ret != 0 {
        return Err(format!(
            "{what} ioctl failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    if !nv_status_ok(free.status) {
        return Err(format!("{what} RM status=0x{:x}", free.status));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn rm_struct_layout_matches_nvidia_headers() {
        assert_eq!(NV_ESC_RM_ALLOC, 0x2b);
        assert_eq!(NV_ESC_RM_FREE, 0x29);

        assert_eq!(size_of::<NvOs00Parameters>(), 16);
        assert_eq!(align_of::<NvOs00Parameters>(), 4);

        assert_eq!(size_of::<NvOs21Parameters>(), 32);
        assert_eq!(align_of::<NvOs21Parameters>(), 8);
        assert_eq!(offset_of!(NvOs21Parameters, p_alloc_parms), 16);
        assert_eq!(offset_of!(NvOs21Parameters, params_size), 24);
        assert_eq!(offset_of!(NvOs21Parameters, status), 28);

        assert_eq!(size_of::<Nv0000AllocParameters>(), 120);
        assert_eq!(align_of::<Nv0000AllocParameters>(), 8);
        assert_eq!(offset_of!(Nv0000AllocParameters, p_os_pid_info), 112);

        assert_eq!(size_of::<Nv0080AllocParameters>(), 56);
        assert_eq!(align_of::<Nv0080AllocParameters>(), 8);
        assert_eq!(offset_of!(Nv0080AllocParameters, va_space_size), 24);
        assert_eq!(offset_of!(Nv0080AllocParameters, va_start_internal), 32);
        assert_eq!(offset_of!(Nv0080AllocParameters, va_limit_internal), 40);
        assert_eq!(offset_of!(Nv0080AllocParameters, va_mode), 48);

        assert_eq!(size_of::<Nv2080AllocParameters>(), 4);
        assert_eq!(align_of::<Nv2080AllocParameters>(), 4);

        assert_eq!(size_of::<NvMemoryAllocationParameters>(), 128);
        assert_eq!(align_of::<NvMemoryAllocationParameters>(), 8);
        assert_eq!(offset_of!(NvMemoryAllocationParameters, range_lo), 48);
        assert_eq!(offset_of!(NvMemoryAllocationParameters, size), 64);
        assert_eq!(offset_of!(NvMemoryAllocationParameters, address), 96);
        assert_eq!(offset_of!(NvMemoryAllocationParameters, ctag_offset), 104);
        assert_eq!(offset_of!(NvMemoryAllocationParameters, numa_node), 120);
    }

    #[test]
    fn rm_status_helper_rejects_nv_errors() {
        assert!(nv_status_ok(0));
        assert!(nv_status_ok(NV_STATUS_WARN_BIT));
        assert!(!nv_status_ok(0x51));
        assert!(!nv_status_ok(NV_ERR_GENERIC));
    }

    #[test]
    #[ignore = "requires /dev/nvidiactl access and a live NVIDIA RM stack"]
    fn rm_backend_allocates_and_frees_device_memory() {
        let mut backend = RmBackend::new(0).expect("create RM backend");
        assert_ne!(backend.rm_control_fd(), 0);
        assert_ne!(backend.h_client, 0);

        let allocation = backend
            .alloc_for_block(0xfeed_beef, 2 * 1024 * 1024)
            .expect("allocate RM device memory");
        assert_ne!(allocation.h_memory, 0);
        assert!(allocation.size >= 2 * 1024 * 1024);

        backend
            .free_block(0xfeed_beef)
            .expect("free RM device memory");
    }
}
