use super::*;
use std::fmt::Write;
use std::mem;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");
static FIELD2_SRC: &str = include_str!("cl/field2.cl");
static EC_SRC: &str = include_str!("cl/ec.cl");
static FFT_SRC: &str = include_str!("cl/fft.cl");
static MULTIEXP_SRC: &str = include_str!("cl/multiexp.cl");

/// Generates the source for FFT and Multiexp operations.
pub fn gen_source<E: GpuEngine, L: Limb>() -> String {
    vec![
        common(),
        gen_ec_source::<E::Fq, L>(),
        field::<E::Scalar, L>("Fr"),
        fft("Fr"),
        multiexp("G1", "Fr"),
        multiexp("G2", "Fr"),
    ]
    .join("\n\n")
}

/// Generates the source for the elliptic curve and group operations, as defined by `E`.
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn gen_ec_source<Fq: GpuField, L: Limb>() -> String {
    vec![
        field::<Fq, L>("Fq"),
        field2("Fq2", "Fq"),
        ec("Fq", "G1"),
        ec("Fq2", "G2"),
    ]
    .join("\n\n")
}

fn ec(field: &str, point: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
}

fn field2(field2: &str, field: &str) -> String {
    String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}

fn multiexp(point: &str, exp: &str) -> String {
    String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
}

/// Trait to implement limbs of different underlying bit sizes.
pub trait Limb: Sized + Clone + Copy {
    fn inc(&mut self);
    /// The underlying size of the limb, e.g. `u32`
    type LimbType: Clone + std::fmt::Display;
    /// Returns the value representing zero.
    fn zero() -> Self;
    /// Returns a new limb.
    fn new(val: Self::LimbType) -> Self;
    /// Returns the raw value of the limb.
    fn value(&self) -> Self::LimbType;
    /// Returns the bit size of the limb.
    fn bits() -> usize {
        mem::size_of::<Self::LimbType>() * 8
    }
    /// Returns a tuple with the strings that PTX is using to describe the type and the register.
    fn ptx_info() -> (&'static str, &'static str);
    /// Returns the type that OpenCL is using to represent the limb.
    fn opencl_type() -> &'static str;
    fn calc_inv(a: Self) -> Self;
    fn limbs_of(value: Vec<u32>) -> Vec<Self>;
}

#[derive(Clone, Copy, Debug)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;
    fn inc(&mut self) {
        self.0 += 1;
    }
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        32
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn limbs_of(value: Vec<u32>) -> Vec<Self> {
        value.into_iter().map(|v| Self(v)).collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;
    fn inc(&mut self) {
        self.0 += 1;
    }
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        64
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn limbs_of(value: Vec<u32>) -> Vec<Self> {
        value
            .chunks(2)
            .map(|v| Self((v[0] as u64) + ((v[1] as u64) << 32)))
            .collect()
    }
}

fn const_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "CONSTANT FIELD {} = {{ {{ {} }} }};",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: GpuField,
{
    let one = L::limbs_of(F::one()); // Get Montgomery form of F::one()
    let p = L::limbs_of(F::modulus()); // Get field modulus in non-Montgomery form
    let r2 = L::limbs_of(F::r2());
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = const_field("FIELD_P", p);
    let r2_def = const_field("FIELD_R2", r2);
    let b3_coeff_def = F::b3_coeff().map(|v| const_field("FIELD_B3_COEFF", L::limbs_of(v)));
    let one_def = const_field("FIELD_ONE", one);
    let zero_def = const_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        inv_def,
        typedef,
        one_def,
        p_def,
        r2_def,
        zero_def,
        b3_coeff_def.unwrap_or_default(),
    ]
    .join("\n")
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L: Limb>() -> Result<String, std::fmt::Error>
where
    F: GpuField,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    writeln!(result, "#if defined(OPENCL_NVIDIA) || defined(CUDA)\n")?;
    for op in &["sub", "add"] {
        let len = L::limbs_of(F::one()).len();

        writeln!(
            result,
            "DEVICE FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{",
            op
        )?;
        if len > 1 {
            write!(result, "asm(")?;
            writeln!(result, "\"{}.cc.{} %0, %0, %{};\\r\\n\"", op, ptx_type, len)?;

            for i in 1..len - 1 {
                writeln!(
                    result,
                    "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"",
                    op,
                    ptx_type,
                    i,
                    i,
                    len + i
                )?;
            }
            writeln!(
                result,
                "\"{}c.{} %{}, %{}, %{};\\r\\n\"",
                op,
                ptx_type,
                len - 1,
                len - 1,
                2 * len - 1
            )?;

            write!(result, ":")?;
            for n in 0..len {
                write!(result, "\"+{}\"(a.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }

            write!(result, "\n:")?;
            for n in 0..len {
                write!(result, "\"{}\"(b.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }
            writeln!(result, ");")?;
        }
        writeln!(result, "return a;\n}}")?;
    }
    writeln!(result, "#endif")?;

    Ok(result)
}

/// Generates PTX-Assembly implementation of FIELD_mul_inner
pub fn field_mul_inner_nvidia<F, L: Limb>() -> String
where
    F: GpuField,
{
    let p = L::limbs_of(F::modulus()); // Get regular form of field modulus
    let inv = L::calc_inv(p[0]).value();

    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    result.push_str("#ifdef OPENCL_NVIDIA\n");
    let len = p.len();

    result.push_str("DEVICE void FIELD_mul_inner(FIELD a, FIELD_limb b, FIELD *result, FIELD_limb *t0, FIELD_limb *t1) {\n");
    let t_pos = 0;
    let a_pos = len + 2;
    let b_pos = len + len + 2;

    result.push_str(format!("asm(\"{{.reg .{} m;\\r\\n\"\n", ptx_type).as_str());
    for lo_or_hi in 0..2 {
        for i in 0..len {
            result.push_str(
                format!(
                    "\"mad{}.{}.cc.{} %{}, %{}, %{}, %{};\\r\\n\"\n",
                    if i == 0 { "" } else { "c" },
                    if i % 2 == 0 { "lo" } else { "hi" },
                    ptx_type,
                    t_pos + i + lo_or_hi,
                    a_pos + lo_or_hi + 2 * (i / 2),
                    b_pos,
                    t_pos + i + lo_or_hi
                )
                .as_str(),
            );
        }
        if lo_or_hi == 0 {
            result
                .push_str(format!("\"addc.{} %{}, %{}, 0;\\r\\n\"\n", ptx_type, len, len).as_str());
        } else {
            result.push_str(format!("\"addc.{} %{}, 0, 0;\\r\\n\"\n", ptx_type, len + 1).as_str());
        }
    }

    result.push_str(format!("\"mul.lo.{} m, %0, {};\\r\\n\"\n", ptx_type, inv).as_str());
    for lo_or_hi in 0..2 {
        for i in 0..len {
            result.push_str(
                format!(
                    "\"mad{}.{}.cc.{} %{}, {}, m, %{};\\r\\n\"\n",
                    if i == 0 { "" } else { "c" },
                    if i % 2 == 0 { "lo" } else { "hi" },
                    ptx_type,
                    t_pos + i,
                    p[lo_or_hi + 2 * (i / 2)].value(),
                    t_pos + i + lo_or_hi
                )
                .as_str(),
            );
        }
        result.push_str(
            format!(
                "\"addc.{} %{}, %{}, 0;\\r\\n\"\n",
                ptx_type,
                len,
                len + lo_or_hi
            )
            .as_str(),
        );
    }
    result.push_str("\"}\"\n");
    result.push_str(":");
    let inps = (0..len)
        .map(|n| format!("\"+{}\"(result->val[{}])", ptx_reg, n))
        .collect::<Vec<_>>()
        .join(", ");
    result.push_str(inps.as_str());
    result.push_str(format!(", \"+{}\"(*t0), \"+{}\"(*t1)", ptx_reg, ptx_reg).as_str());

    result.push_str("\n:");
    let outs = (0..len)
        .map(|n| format!("\"{}\"(a.val[{}])", ptx_reg, n))
        .collect::<Vec<_>>()
        .join(", ");
    result.push_str(outs.as_str());
    result.push_str(format!(", \"{}\"(b));", ptx_reg).as_str());

    result.push_str("\n}\n");
    result.push_str("#endif\n");

    result
}

/// Returns CUDA/OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn field<F, L: Limb>(name: &str) -> String
where
    F: GpuField,
{
    [
        params::<F, L>(),
        field_add_sub_nvidia::<F, L>().expect("preallocated"),
        field_mul_inner_nvidia::<F, L>(),
        String::from(FIELD_SRC),
    ]
    .join("\n")
    .replace("FIELD", name)
}

/// Returns CUDA/OpenCL source-code that contains definitions/functions that are shared across
/// fields.
///
/// It needs to be called before any other function like [`field`] or [`gen_ec_source`] is called,
/// as it contains deinitions, used in those.
pub fn common() -> String {
    COMMON_SRC.to_string()
}
