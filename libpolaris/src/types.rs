//! POLARIS ioctl argument types and constants (userspace side).
//!
//! Every type in this module is generated from `kernel/polaris_abi.rs`
//! at build time.  To add or change a type, edit the kernel-side file
//! and recompile — both the kernel module and libpolaris will pick it up.

include!(concat!(env!("OUT_DIR"), "/polaris_abi_generated.rs"));
