mod error;
pub use error::*;

mod program;
pub use program::*;

mod fft;
pub use fft::*;

mod source;
pub use source::*;

mod multiexp;
pub use multiexp::*;

/// Describes how to generate the elliptic curve operations for
/// - `Scalar`
/// - `Fp`
/// - `Fp2`
/// - `G1`
/// - `G2`
pub trait GpuEngine {
    type Scalar: GpuField;
    type Fq: GpuField;
}

/// Describes how to generate the gpu sources for a Field.
pub trait GpuField {
    /// Returns `1` as a vector of 32bit limbs.
    fn one() -> Vec<u32>;

    /// Returns `R ^ 2 mod P` as a vector of 32bit limbs.
    fn r2() -> Vec<u32>;

    /// Returns the field modulus in non-Montgomery form (least significant limb first).
    fn modulus() -> Vec<u32>;

    fn b3_coeff() -> Option<Vec<u32>>;
}

pub struct Bls12Fr;
impl GpuField for Bls12Fr {
    fn one() -> Vec<u32> {
        vec![
            4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881,
        ]
    }
    fn modulus() -> Vec<u32> {
        vec![
            1, 4294967295, 4294859774, 1404937218, 161601541, 859428872, 698187080, 1944954707,
        ]
    }
    fn r2() -> Vec<u32> {
        vec![
            4092763245, 3382307216, 2274516003, 728559051, 1918122383, 97719446, 2673475345,
            122214873,
        ]
    }

    fn b3_coeff() -> Option<Vec<u32>> {
        None
    }
}

pub struct Bls12Fq;
impl GpuField for Bls12Fq {
    fn one() -> Vec<u32> {
        vec![
            196605, 1980301312, 3289120770, 3958636555, 1405573306, 1598593111, 1884444485,
            2010011731, 2723605613, 1543969431, 4202751123, 368467651,
        ]
    }
    fn modulus() -> Vec<u32> {
        vec![
            4294945451, 3120496639, 2975072255, 514588670, 4138792484, 1731252896, 4085584575,
            1685539716, 1129032919, 1260103606, 964683418, 436277738,
        ]
    }
    fn r2() -> Vec<u32> {
        vec![
            473175878, 4108263220, 164693233, 175564454, 1284880085, 2380613484, 2476573632,
            1743489193, 3038352685, 2591637125, 2462770090, 295210981,
        ]
    }
    fn b3_coeff() -> Option<Vec<u32>> {
        Some(vec![
            2577710, 1148583936, 1128792096, 3703046298, 1248758617, 1870588366, 3232324550,
            2969776311, 4213068983, 1631629820, 2131473633, 58834441,
        ])
    }
}

impl GpuEngine for bls12_381::Bls12 {
    type Scalar = Bls12Fr;
    type Fq = Bls12Fq;
}
