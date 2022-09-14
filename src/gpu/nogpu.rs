use super::error::{GpuError, GpuResult};
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use std::marker::PhantomData;

pub struct MultiexpKernel<E>(PhantomData<E>)
where
    E: Engine;

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(_: bool) -> GpuResult<Self> {
        Err(GpuError::GpuDisabled)
    }

    pub fn multiexp<G>(
        &mut self,
        _: &Worker,
        _: &[G],
        _: &[<G::Scalar as PrimeField>::Repr],
        _: usize,
        _: usize,
    ) -> GpuResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        Err(GpuError::GpuDisabled)
    }
}

pub struct FftKernel<E>(PhantomData<E>)
where
    E: Engine;

impl<E> FftKernel<E>
where
    E: Engine,
{
    pub fn ifft_coset_fft_mul_sub_divide_by_z_icoset_fft_unmont(
        &mut self,
        _: &mut [E::Fr],
        _: &[E::Fr],
        _: &[E::Fr],
        _: u32,
        _: &E::Fr,
        _: &E::Fr,
        _: &E::Fr,
        _: &E::Fr,
        _: &E::Fr,
    ) -> GpuResult<()> {
        Err(GpuError::GpuDisabled)
    }

    pub fn many_ifft_coset_fft(
        &mut self,
        _: Vec<(&mut [E::Fr], u32, &E::Fr, &E::Fr, &E::Fr)>,
    ) -> GpuResult<()> {
        Err(GpuError::GpuDisabled)
    }
}
