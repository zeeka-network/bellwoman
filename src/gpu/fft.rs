use super::*;

use crate::multicore::Worker;
use ff::PrimeField;
use log::{error, info};
use pairing::Engine;
use std::cmp;
use std::sync::{Arc, RwLock};

const LOG2_MAX_ELEMENTS: usize = 27; // At most 2^27 elements is supported.
const MAX_LOG2_RADIX: u32 = 9; // Radix512
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 8; // 256
const DISTRIBUTE_POWERS_DEGREE: u32 = 3;

const LOCAL_WORK_SIZE: usize = 256;

/// Performs FFT on `input`
/// * `omega` - Special value `omega` is used for FFT over finite-fields
/// * `log_n` - Specifies log2 of number of elements
pub fn radix_fft<S: PrimeField>(
    program: &Program,
    src_buffer: &mut Buffer<S>,
    dst_buffer: &mut Buffer<S>,
    omega: &S,
    log_n: u32,
    maybe_abort: &Option<Arc<RwLock<bool>>>,
) -> GpuResult<()> {
    let n = 1 << log_n;
    // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

    // Precalculate:
    // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
    let mut pq = vec![S::ZERO; 1 << max_deg >> 1];
    let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
    pq[0] = S::ONE;
    if max_deg > 1 {
        pq[1] = twiddle;
        for i in 2..(1 << max_deg >> 1) {
            pq[i] = pq[i - 1];
            pq[i].mul_assign(&twiddle);
        }
    }
    let pq_buffer = program.create_buffer_from_slice(&pq)?;

    // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
    let mut omegas = vec![S::ZERO; 32];
    omegas[0] = *omega;
    for i in 1..LOG2_MAX_ELEMENTS {
        omegas[i] = omegas[i - 1].pow_vartime([2u64]);
    }
    let omegas_buffer = program.create_buffer_from_slice(&omegas)?;
    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    let mut log_p = 0u32;
    // Each iteration performs a FFT round
    while log_p < log_n {
        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        let deg = cmp::min(max_deg, log_n - log_p);

        if let Some(maybe_abort) = maybe_abort.clone() {
            if *maybe_abort.read().unwrap() {
                return Err(GpuError::Aborted);
            }
        }

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = (n >> deg) * local_work_size;
        let kernel = program.create_kernel(
            "radix_fft",
            global_work_size as usize,
            local_work_size as usize,
        );
        kernel
            .arg(&*src_buffer)
            .arg(&*dst_buffer)
            .arg(&pq_buffer)
            .arg(&omegas_buffer)
            .arg(LocalBuffer::<S>::new(1 << deg))
            .arg(n)
            .arg(log_p)
            .arg(deg)
            .arg(max_deg)
            .run()?;

        log_p += deg;
        std::mem::swap(src_buffer, dst_buffer);
    }

    Ok(())
}

#[derive(Debug, Copy, PartialEq, Default, Clone)]
struct Chunk256([u64; 4]);
unsafe impl ocl::OclPrm for Chunk256 {}

pub fn mul_by_field<S: PrimeField>(
    program: &Program,
    src_buffer: &Buffer<S>,
    mul: &S,
    log_n: u32,
) -> GpuResult<()> {
    let n = 1u32 << log_n;
    let kernel = program.create_kernel("mul_by_field", n as usize, LOCAL_WORK_SIZE);
    kernel
        .arg(src_buffer)
        .arg(n)
        .arg(*unsafe { std::mem::transmute::<&S, &Chunk256>(mul) })
        .run()?;
    Ok(())
}

pub fn mul_assign<S: PrimeField>(
    program: &Program,
    src_buffer: &mut Buffer<S>,
    aux_buffer: &Buffer<S>,
    log_n: u32,
) -> GpuResult<()> {
    let n = 1u32 << log_n;
    let kernel = program.create_kernel("mul_assign", n as usize, LOCAL_WORK_SIZE);
    kernel.arg(&*src_buffer).arg(aux_buffer).arg(n).run()?;
    Ok(())
}

pub fn sub_assign<S: PrimeField>(
    program: &Program,
    src_buffer: &mut Buffer<S>,
    aux_buffer: &Buffer<S>,
    log_n: u32,
) -> GpuResult<()> {
    let n = 1u32 << log_n;
    let kernel = program.create_kernel("sub_assign", n as usize, LOCAL_WORK_SIZE);
    kernel.arg(&*src_buffer).arg(aux_buffer).arg(n).run()?;
    Ok(())
}

pub fn distribute_powers<S: PrimeField>(
    program: &Program,
    src_buffer: &mut Buffer<S>,
    log_n: u32,
    g: &S,
) -> GpuResult<()> {
    let n = 1u32 << log_n;
    let max_deg: u32 = cmp::min(DISTRIBUTE_POWERS_DEGREE, log_n);
    let kernel = program.create_kernel(
        "distribute_powers",
        (n >> max_deg) as usize,
        LOCAL_WORK_SIZE,
    );
    kernel
        .arg(&*src_buffer)
        .arg(n)
        .arg(max_deg)
        .arg(*unsafe { std::mem::transmute::<&S, &Chunk256>(g) })
        .run()?;
    Ok(())
}

#[allow(dead_code)]
pub fn unmont<S: PrimeField>(
    program: &Program,
    src_buffer: &mut Buffer<S>,
    log_n: u32,
) -> GpuResult<()> {
    let n = 1u32 << log_n;
    let kernel = program.create_kernel("unmont", n as usize, LOCAL_WORK_SIZE);
    kernel.arg(&*src_buffer).run()?;
    Ok(())
}

/// FFT kernel for a single GPU.
pub struct SingleFftKernel<E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    /// An optional function which will be called at places where it is possible to abort the FFT
    /// calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<Arc<RwLock<bool>>>,
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Engine + GpuEngine> SingleFftKernel<E> {
    pub fn create(device: &Device, maybe_abort: Option<Arc<RwLock<bool>>>) -> GpuResult<Self> {
        let source = match device.brand() {
            Brand::Nvidia => gen_source::<E, Limb32>(),
            _ => gen_source::<E, Limb64>(),
        };
        let program = Program::from_opencl(device, &source)?;

        Ok(SingleFftKernel {
            program,
            maybe_abort,
            _phantom: Default::default(),
        })
    }

    pub fn ifft_coset_fft_mul_sub_divide_by_z_icoset_fft(
        &mut self,
        a: &mut [E::Fr],
        b: &[E::Fr],
        c: &[E::Fr],
        log_n: u32,
        omega: &E::Fr,
        omegainv: &E::Fr,
        minv: &E::Fr,
        geninv: &E::Fr,
        div_z: &E::Fr,
    ) -> GpuResult<()> {
        let n = 1 << log_n;
        // All usages are safe as the buffers are initialized from either the host or the GPU
        // before they are read.
        let mut buffer = self.program.create_buffer::<E::Fr>(n)?;
        let mut aux_buffer = self.program.create_buffer::<E::Fr>(n)?;
        buffer.write_from(a)?;
        radix_fft::<E::Fr>(
            &self.program,
            &mut buffer,
            &mut aux_buffer,
            omegainv,
            log_n,
            &self.maybe_abort,
        )?;
        mul_by_field::<E::Fr>(&self.program, &mut buffer, minv, log_n)?;
        distribute_powers::<E::Fr>(
            &self.program,
            &mut buffer,
            log_n,
            &E::Fr::MULTIPLICATIVE_GENERATOR,
        )?;
        radix_fft::<E::Fr>(
            &self.program,
            &mut buffer,
            &mut aux_buffer,
            omega,
            log_n,
            &self.maybe_abort,
        )?;
        aux_buffer.write_from(b)?;
        mul_assign::<E::Fr>(&self.program, &mut buffer, &aux_buffer, log_n)?;
        aux_buffer.write_from(c)?;
        sub_assign::<E::Fr>(&self.program, &mut buffer, &aux_buffer, log_n)?;
        mul_by_field::<E::Fr>(&self.program, &mut buffer, div_z, log_n)?;
        radix_fft::<E::Fr>(
            &self.program,
            &mut buffer,
            &mut aux_buffer,
            omegainv,
            log_n,
            &self.maybe_abort,
        )?;
        mul_by_field::<E::Fr>(&self.program, &mut buffer, minv, log_n)?;
        distribute_powers::<E::Fr>(&self.program, &mut buffer, log_n, geninv)?;
        //unmont::<S>(&self.program, &mut buffer, log_n)?;
        buffer.read_into(a)?;
        Ok(())
    }
    pub fn ifft_coset_fft(
        &mut self,
        a: &mut [E::Fr],
        log_n: u32,
        omega: &E::Fr,
        omegainv: &E::Fr,
        minv: &E::Fr,
    ) -> GpuResult<()> {
        let n = 1 << log_n;
        // All usages are safe as the buffers are initialized from either the host or the GPU
        // before they are read.
        let mut buffer = self.program.create_buffer::<E::Fr>(n)?;
        let mut aux_buffer = self.program.create_buffer::<E::Fr>(n)?;
        buffer.write_from(a)?;
        radix_fft::<E::Fr>(
            &self.program,
            &mut buffer,
            &mut aux_buffer,
            omegainv,
            log_n,
            &self.maybe_abort,
        )?;
        mul_by_field::<E::Fr>(&self.program, &mut buffer, minv, log_n)?;
        distribute_powers::<E::Fr>(
            &self.program,
            &mut buffer,
            log_n,
            &E::Fr::MULTIPLICATIVE_GENERATOR,
        )?;
        radix_fft::<E::Fr>(
            &self.program,
            &mut buffer,
            &mut aux_buffer,
            omega,
            log_n,
            &self.maybe_abort,
        )?;
        buffer.read_into(a)?;
        Ok(())
    }
}

/// One FFT kernel for each GPU available.
pub struct FftKernel<E>
where
    E: Engine + GpuEngine,
{
    kernels: Vec<Arc<RwLock<SingleFftKernel<E>>>>,
}

impl<E> FftKernel<E>
where
    E: Engine + GpuEngine,
{
    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        devices: &[(Device, OptParams)],
        maybe_abort: Option<Arc<RwLock<bool>>>,
    ) -> GpuResult<Self> {
        let kernels: Vec<_> = devices
            .iter()
            .filter_map(|(device, _)| {
                let kernel = SingleFftKernel::<E>::create(device, maybe_abort.clone());
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(GpuError::NoGpu);
        }
        info!("FFT: {} working device(s) selected. ", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("FFT: Device {}: {}", i, k.program.device().name(),);
        }

        Ok(Self {
            kernels: kernels
                .into_iter()
                .map(|k| Arc::new(RwLock::new(k)))
                .collect(),
        })
    }

    pub fn ifft_coset_fft_mul_sub_divide_by_z_icoset_fft(
        &mut self,
        a: &mut [E::Fr],
        b: &[E::Fr],
        c: &[E::Fr],
        log_n: u32,
        omega: &E::Fr,
        omegainv: &E::Fr,
        minv: &E::Fr,
        geninv: &E::Fr,
        div_z: &E::Fr,
    ) -> GpuResult<()> {
        self.kernels[0]
            .write()
            .unwrap()
            .ifft_coset_fft_mul_sub_divide_by_z_icoset_fft(
                a, b, c, log_n, omega, omegainv, minv, geninv, div_z,
            )
    }

    pub fn many_ifft_coset_fft(
        &mut self,
        pool: &Worker,
        all: Vec<(&mut [E::Fr], u32, E::Fr, E::Fr, E::Fr)>,
    ) -> GpuResult<()> {
        let result = pool.scope(all.len(), |s, _| {
            let results = Arc::new(RwLock::new(Vec::new()));
            for ((a, log_n, omega, omegainv, minv), kern) in
                all.into_iter().zip(self.kernels.iter().cycle())
            {
                let res = Arc::clone(&results);
                s.spawn(move |_| {
                    res.write().unwrap().push(
                        kern.write()
                            .unwrap()
                            .ifft_coset_fft(a, log_n, &omega, &omegainv, &minv),
                    );
                });
            }
            results
        });

        Arc::try_unwrap(result)
            .unwrap()
            .into_inner()
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<()>, GpuError>>()?;

        Ok(())
    }
}
