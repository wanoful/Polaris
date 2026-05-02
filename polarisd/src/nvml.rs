use nvml_wrapper::Nvml;

#[allow(dead_code)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub total_memory: u64,
    pub free_memory: u64,
}

pub fn discover_gpus() -> Result<Vec<GpuInfo>, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;

    let count = nvml
        .device_count()
        .map_err(|e| format!("NVML device_count failed: {e}"))?;

    let mut gpus = Vec::with_capacity(count as usize);
    for i in 0..count {
        let dev = nvml
            .device_by_index(i)
            .map_err(|e| format!("NVML device_by_index({i}) failed: {e}"))?;

        let name = dev
            .name()
            .map_err(|e| format!("NVML name({i}) failed: {e}"))?;

        let mem = dev
            .memory_info()
            .map_err(|e| format!("NVML memory_info({i}) failed: {e}"))?;

        gpus.push(GpuInfo {
            index: i,
            name,
            total_memory: mem.total,
            free_memory: mem.free,
        });

        eprintln!(
            "polarisd: NVML GPU {i}: {} (total={} MiB, free={} MiB)",
            gpus.last().unwrap().name,
            mem.total / (1024 * 1024),
            mem.free / (1024 * 1024),
        );
    }

    Ok(gpus)
}
