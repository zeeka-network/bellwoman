use bellman::gpu::*;
use bellman::multiexp::Exponent;
use bls12_381::{Bls12, Scalar};
use ff::PrimeFieldBits;
use pairing::Engine;
fn main() {
    //println!("{}", gen_source::<Bls12, Limb64>());
    let mut g = <Bls12 as Engine>::G1::generator();
    //g=g.double();
    println!("{:?}", g); //*<Bls12 as Engine>::G1::generator());
                         //println!("{}", std::mem::size_of::<<Bls12 as Engine>::G1Affine>());
                         //let devs = Device::by_brand(Brand::Nvidia).unwrap();
                         //println!("{}", "hi");
}
