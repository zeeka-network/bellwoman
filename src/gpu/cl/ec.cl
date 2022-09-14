// Elliptic curve operations (Short Weierstrass Jacobian form)

#define POINT_ZERO ((POINT_projective){FIELD_ZERO, FIELD_ONE, FIELD_ZERO})

typedef struct {
  FIELD x;
  FIELD y;
  bool _inf;
  #if Fq_LIMB_BITS == 32
   uint _padding;
 #endif
} POINT_affine;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD z;
} POINT_projective;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE POINT_projective POINT_double(POINT_projective inp) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(inp.z, local_zero)) {
      return inp;
  }

  FIELD t0 = FIELD_sqr(inp.y);
  FIELD z3 = FIELD_double(FIELD_double(FIELD_double(t0)));
  FIELD t1 = FIELD_mul(inp.y, inp.z);
  FIELD t2 = FIELD_sqr(inp.z);
  t2 = FIELD_mul_3b(t2);
  FIELD x3 = FIELD_mul(t2, z3);
  FIELD y3 = FIELD_add(t0, t2);
  z3 = FIELD_mul(t1, z3);
  t1 = FIELD_double(t2);
  t2 = FIELD_add(t1, t2);
  t0 = FIELD_sub(t0, t2);
  y3 = FIELD_mul(t0, y3);
  y3 = FIELD_add(x3, y3);
  t1 = FIELD_mul(inp.x, inp.y);
  x3 = FIELD_mul(t0, t1);
  x3 = FIELD_double(x3);

  inp.x = x3;
  inp.y = y3;
  inp.z = z3;
  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE POINT_projective POINT_add_mixed(POINT_projective a, POINT_affine b) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) {
    const FIELD local_one = FIELD_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  FIELD t0 = FIELD_mul(a.x, b.x);
  FIELD t1 = FIELD_mul(a.y, b.y);
  FIELD t3 = FIELD_add(b.x, b.y);
  FIELD t4 = FIELD_add(a.x, a.y);
  t3 = FIELD_mul(t3, t4);
  t4 = FIELD_add(t0, t1);
  t3 = FIELD_sub(t3, t4);
  t4 = FIELD_mul(b.y, a.z);
  t4 = FIELD_add(t4, a.y);
  FIELD y3 = FIELD_mul(b.x, a.z);
  y3 = FIELD_add(y3, a.x);
  FIELD x3 = FIELD_double(t0);
  t0 = FIELD_add(x3, t0);
  FIELD t2 = FIELD_mul_3b(a.z);
  FIELD z3 = FIELD_add(t1, t2);
  t1 = FIELD_sub(t1, t2);
  y3 = FIELD_mul_3b(y3);
  x3 = FIELD_mul(t4, y3);
  t2 = FIELD_mul(t3, t1);
  x3 = FIELD_sub(t2, x3);
  y3 = FIELD_mul(y3, t0);
  t1 = FIELD_mul(t1, z3);
  y3 = FIELD_add(t1, y3);
  t0 = FIELD_mul(t0, t3);
  z3 = FIELD_mul(z3, t4);
  z3 = FIELD_add(z3, t0);

  POINT_projective ret;
  ret.x = x3;
  ret.y = y3;
  ret.z = z3;

  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE POINT_projective POINT_add(POINT_projective a, POINT_projective b) {

  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) return b;
  if(FIELD_eq(b.z, local_zero)) return a;

  FIELD t0 = FIELD_mul(a.x, b.x);
  FIELD t1 = FIELD_mul(a.y, b.y);
  FIELD t2 = FIELD_mul(a.z, b.z);
  FIELD t3 = FIELD_add(a.x, a.y);
  FIELD t4 = FIELD_add(b.x, b.y);
  t3 = FIELD_mul(t3, t4);
  t4 = FIELD_add(t0, t1);
  t3 = FIELD_sub(t3, t4);
  t4 = FIELD_add(a.y, a.z);
  FIELD x3 = FIELD_add(b.y, b.z);
  t4 = FIELD_mul(t4, x3);
  x3 = FIELD_add(t1, t2);
  t4 = FIELD_sub(t4, x3);
  x3 = FIELD_add(a.x, a.z);
  FIELD y3 = FIELD_add(b.x, b.z);
  x3 = FIELD_mul(x3, y3);
  y3 = FIELD_add(t0, t2);
  y3 = FIELD_sub(x3, y3);
  x3 = FIELD_double(t0);
  t0 = FIELD_add(x3, t0);
  t2 = FIELD_mul_3b(t2);
  FIELD z3 = FIELD_add(t1, t2);
  t1 = FIELD_sub(t1, t2);
  y3 = FIELD_mul_3b(y3);
  x3 = FIELD_mul(t4, y3);
  t2 = FIELD_mul(t3, t1);
  x3 = FIELD_sub(t2, x3);
  y3 = FIELD_mul(y3, t0);
  t1 = FIELD_mul(t1, z3);
  y3 = FIELD_add(t1, y3);
  t0 = FIELD_mul(t0, t3);
  z3 = FIELD_mul(z3, t4);
  z3 = FIELD_add(z3, t0);

  POINT_projective ret;
  ret.x = x3;
  ret.y = y3;
  ret.z = z3;

  return ret;
}
