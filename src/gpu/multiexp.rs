use super::*;

use crate::multicore::Worker;
use ff::PrimeField;
use ff::PrimeFieldBits;
use group::{prime::PrimeCurveAffine, Group};
use log::{error, info};
use pairing::Engine;
use std::any::TypeId;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

const LOCAL_WORK_SIZE: usize = 256;

#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    pub n_g1: usize,
    pub window_size_g1: usize,
    pub groups_g1: usize,
    pub n_g2: usize,
    pub window_size_g2: usize,
    pub groups_g2: usize,
}

impl OptParams {
    pub fn default() -> Self {
        Self {
            n_g1: 32 * 1024 * 1024,
            window_size_g1: 11,
            groups_g1: 360,
            n_g2: 16 * 1024 * 1024,
            window_size_g2: 10,
            groups_g2: 332,
        }
    }
}

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    opt_params: OptParams,
    maybe_abort: Option<Arc<RwLock<bool>>>,
    _phantom: std::marker::PhantomData<E>,
}

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    /// Create a new kernel for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        device: Device,
        opt_params: OptParams,
        maybe_abort: Option<Arc<RwLock<bool>>>,
    ) -> GpuResult<Self> {
        let source = match device.brand() {
            Brand::Nvidia => gen_source::<E, Limb32>(),
            _ => gen_source::<E, Limb64>(),
        };
        let program = Program::from_opencl(&device, &source)?;

        Ok(SingleMultiexpKernel {
            program,
            opt_params,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the actual multiexp computation on the GPU.
    pub fn multiexp<G>(
        &self,
        bases: &[G],
        exps: &[<G::Scalar as PrimeField>::Repr],
        n: usize,
    ) -> GpuResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        if let Some(maybe_abort) = self.maybe_abort.clone() {
            if *maybe_abort.read().unwrap() {
                return Err(GpuError::Aborted);
            }
        }

        let exp_bits = exp_size::<E>() * 8;
        let window_size = if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
            self.opt_params.window_size_g1
        } else {
            self.opt_params.window_size_g2
        };
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
            self.opt_params.groups_g1
        } else {
            self.opt_params.groups_g2
        };
        let bucket_len = 1 << window_size;
        let work_count = num_windows * num_groups;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let base_buffer = self.program.create_buffer_from_slice(bases)?;
        let exp_buffer = self.program.create_buffer_from_slice(exps)?;

        // It is safe as the GPU will initialize that buffer
        let bucket_buffer = self
            .program
            .create_buffer::<<G as PrimeCurveAffine>::Curve>(work_count * bucket_len)?;
        // It is safe as the GPU will initialize that buffer
        let result_buffer = self
            .program
            .create_buffer::<<G as PrimeCurveAffine>::Curve>(work_count)?;

        let global_work_size =
            work_count + ((LOCAL_WORK_SIZE - (work_count % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE);
        info!(
            "Single Multiexp: Type: {} N: {} Windows-size: {} Num-groups: {} Global-work-size: {}",
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1"
            } else {
                "G2"
            },
            n,
            window_size,
            num_groups,
            global_work_size
        );

        let kernel = self.program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GpuError::UnsupportedCurve);
            },
            global_work_size,
            LOCAL_WORK_SIZE,
        );

        kernel
            .arg(&base_buffer)
            .arg(&bucket_buffer)
            .arg(&result_buffer)
            .arg(&exp_buffer)
            .arg(n as u32)
            .arg(num_groups as u32)
            .arg(num_windows as u8)
            .arg(window_size as u8)
            .run()?;

        let mut results = vec![<G as PrimeCurveAffine>::Curve::identity(); work_count];
        result_buffer.read_into(&mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

/// A struct that containts several multiexp kernels for different devices.
pub struct MultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    kernels: Vec<Arc<RwLock<SingleMultiexpKernel<E>>>>,
}

impl<E> MultiexpKernel<E>
where
    E: Engine + GpuEngine,
    E::Fr: PrimeFieldBits,
{
    pub fn create(
        devices: &[(Device, OptParams)],
        maybe_abort: Option<Arc<RwLock<bool>>>,
    ) -> GpuResult<Self> {
        let kernels: Vec<_> = devices
            .iter()
            .filter_map(|(device, opt_params)| {
                let kernel = SingleMultiexpKernel::<E>::create(
                    device.clone(),
                    opt_params.clone(),
                    maybe_abort.clone(),
                )
                .unwrap();
                Some(kernel)
                /*if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()*/
            })
            .collect();

        if kernels.is_empty() {
            return Err(GpuError::NoGpu);
        }
        info!("Multiexp: {} working device(s) selected.", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("Multiexp: Device {}: {}", i, k.program.device().name());
        }
        Ok(MultiexpKernel::<E> {
            kernels: kernels
                .into_iter()
                .map(|k| Arc::new(RwLock::new(k)))
                .collect(),
        })
    }

    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: &[G],
        exps: &[<G::Scalar as PrimeField>::Repr],
        skip: usize,
    ) -> GpuResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        const GPU_MINIMUM: usize = 1024;

        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + exps.len())];
        let exps = &exps[..];

        if exps.len() < GPU_MINIMUM {
            panic!("GPU multiexp smaller than minimum!");
        }

        let num_devices = self.kernels.len();
        let num_exps = exps.len();
        let chunk_size = ((num_exps as f64) / (num_devices as f64)).ceil() as usize;

        let result = pool.scope(num_devices, |s, _| {
            let results = Arc::new(RwLock::new(Vec::new()));
            for ((bases, exps), kern) in bases
                .chunks(chunk_size)
                .zip(exps.chunks(chunk_size))
                .zip(self.kernels.iter())
            {
                let res = Arc::clone(&results);
                s.spawn(move |_| {
                    let mut acc = <G as PrimeCurveAffine>::Curve::identity();
                    let kern = kern.write().unwrap();
                    let n = if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                        kern.opt_params.n_g1
                    } else {
                        kern.opt_params.n_g2
                    };
                    for (bases, exps) in bases.chunks(n).zip(exps.chunks(n)) {
                        match kern.multiexp(bases, exps, bases.len()) {
                            Ok(v) => acc.add_assign(v),
                            Err(e) => {
                                res.write().unwrap().push(Err(e));
                                return;
                            }
                        }
                    }
                    res.write().unwrap().push(Ok(acc));
                });
            }
            results
        });

        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        for v in Arc::try_unwrap(result)
            .unwrap()
            .into_inner()
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<<G as PrimeCurveAffine>::Curve>, GpuError>>()?
        {
            acc.add_assign(v);
        }

        Ok(acc)
    }
}
