use super::ProgramError;

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("OpenCL error: {0}")]
    OclError(#[from] ocl::Error),
    #[error("OpenCL error: {0}")]
    ProgramError(#[from] ProgramError),
    #[error("GPU accelerator is disabled!")]
    GpuDisabled,
    #[error("GPU work aborted")]
    Aborted,
    #[error("No GPU found!")]
    NoGpu,
    #[error("Unsupported curve")]
    UnsupportedCurve,
}

pub type GpuResult<T> = std::result::Result<T, GpuError>;
