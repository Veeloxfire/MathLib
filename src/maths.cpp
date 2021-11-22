#include "maths.h"
#include <math.h>

#include <assert.h>//Contains assert
#include <string.h>//Contains memcpy_s

f32 MATH::abs(f32 f) {
  return fabsf(f);
}

rad32 MATH::arcsin(f32 f) {
  return asinf(f);
}

rad32 MATH::arctan(f32 f) {
  return atanf(f);
}

f32 MATH::sin(rad32 f) {
  return sinf(f);
}

f32 MATH::cos(rad32 f) {
  return cosf(f);
}

f32 MATH::tan(rad32 f) {
  return tanf(f);
}


f32 MATH::sqrt(f32 f) {
  return sqrtf(f);
}

f32 MATH::maximum(f32 a, f32 b) {
  return a > b ? a : b;
}

f32 MATH::log2(f32 f) {
  return log2f(f);
}

i32 MATH::round(f32 f) {
  return (i32)(f + (0.5 - (f < 0)));
}

i32 MATH::ulp(const f32 f) {
  i32 val;
  memcpy_s(&val, 4, &f, 4);

  return val;
}

f32 MATH::from_ulp(u32 u) {
  f32 val;
  memcpy_s(&val, 4, &u, 4);

  return val;
}

static constexpr u32 F32_INF_OR_NAN_MASK = 0x7F800000;
static constexpr u32 SIGN_BIT = 0x80000000;

bool MATH::is_inf(f32 f) {
  i32 v = ulp(f);
  v &= ~SIGN_BIT;
  v ^= F32_INF_OR_NAN_MASK;
  return v == 0;
}

bool MATH::is_nan(f32 f) {
  i32 v = ulp(f);
  v &= ~SIGN_BIT;
  v ^= F32_INF_OR_NAN_MASK;
  return 0 < v && v <= 0x007FFFFF;
}

bool MATH::is_nan_or_inf(f32 f) {
  i32 v = ulp(f);
  v &= ~SIGN_BIT;
  v ^= F32_INF_OR_NAN_MASK;
  return v <= 0x007FFFFF;
}

bool MATH::abs_diff_eq(f32 a, f32 b, f32 max_diff) {
  return abs(a - b) <= max_diff;
}

bool MATH::rel_diff_eq(f32 a, f32 b, f32 max_diff) {
  const f32 diff = abs(a - b);
  a = abs(a);
  b = abs(b);

  f32 scaled_max_diff = max_diff * maximum(a, b);

  return diff <= scaled_max_diff;
}

u32 MATH::ulp_difference(f32 a, f32 b) {
  const i32 ulp_a = ulp(a);
  const i32 ulp_b = ulp(b);

  if (((ulp_a & (~SIGN_BIT)) ^ F32_INF_OR_NAN_MASK) <= 0x007FFFFF
      || ((ulp_b & (~SIGN_BIT)) ^ F32_INF_OR_NAN_MASK) <= 0x007FFFFF
      || (ulp_a < 0) != (ulp_b < 0)) return I32_MAX;


  return abs(ulp_a - ulp_b);
}

bool MATH::approx_eq(f32 a, f32 b, f32 max_abs_diff) {
  if (a == b) return true;

  u32 u_diff = ulp_difference(a, b);
  if (u_diff < 1000) return true;


  return abs_diff_eq(a, b, max_abs_diff);
}

bool MATH::approx_eq(const vec3& v1, const vec3& v2,
                     f32 max_abs_diff) {
  return approx_eq(v1.x, v2.x, max_abs_diff)
    && approx_eq(v1.y, v2.y, max_abs_diff)
    && approx_eq(v1.z, v2.z, max_abs_diff);
}

bool MATH::approx_eq(const vec4& v1, const vec4& v2,
                     f32 max_abs_diff) {
  return approx_eq(v1.x, v2.x, max_abs_diff)
    && approx_eq(v1.y, v2.y, max_abs_diff)
    && approx_eq(v1.z, v2.z, max_abs_diff)
    && approx_eq(v1.w, v2.w, max_abs_diff);
}

bool MATH::approx_eq(const quat4& q1, const quat4& q2,
                     f32 max_abs_diff) {
  return approx_eq(q1.r, q2.r, max_abs_diff)
    && approx_eq(q1.i, q2.i, max_abs_diff)
    && approx_eq(q1.j, q2.j, max_abs_diff)
    && approx_eq(q1.k, q2.k, max_abs_diff);
}

void vec3::normalise() {
  const f32 len = MATH::sqrt(x * x + y * y + z * z);

  x /= len;
  y /= len;
  z /= len;
}

void vec3::scale(f32 f) {
  x *= f;
  y *= f;
  z *= f;
}

void vec3::sub(const vec3& v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
}

void vec3::add(const vec3& v) {
  x += v.x;
  y += v.y;
  z += v.z;
}

void vec3::piecewise_mul(const vec3& v) {
  x *= v.x;
  y *= v.y;
  z *= v.z;
}

mat3x3 rotate_around_axis_matrix(vec3 axis, rad32 theta) {
  const f32 cos_theta = MATH::cos(theta);
  const f32 sin_theta = MATH::sin(theta);

  mat3x3 res = mat3x3::identity();
  res.scale(cos_theta);

  mat3x3 cpm_axis = cross_product_matrix(axis);
  cpm_axis.scale(sin_theta);

  mat3x3 op_u2 = outer_product(axis, axis);
  op_u2.scale(1 - cos_theta);

  res.add(cpm_axis);
  res.add(op_u2);

  return res;
}

mat3x3 rotate_around_origin_m3(const vec3& rot) {
  const f32 cos_x = MATH::cos(rot.x);
  const f32 cos_y = MATH::cos(rot.y);
  const f32 cos_z = MATH::cos(rot.z);

  const f32 sin_x = MATH::sin(rot.x);
  const f32 sin_y = MATH::sin(rot.y);
  const f32 sin_z = MATH::sin(rot.z);

  return {
    cos_x * cos_y, cos_x * sin_y * sin_z - sin_x * cos_z, cos_x * sin_y * sin_z + sin_x * sin_z,
    sin_x * cos_y, sin_x * sin_y * sin_z - cos_x * cos_z, sin_x * sin_y * cos_z + cos_x * sin_z,
    -sin_y, cos_y * sin_z, cos_y * cos_z
  };
}

mat4x4 rotation_matrix(const vec3& angles) {
  const f32 cos_a = MATH::cos(angles.z);
  const f32 cos_b = MATH::cos(angles.y);
  const f32 cos_c = MATH::cos(angles.x);

  const f32 sin_a = MATH::sin(angles.z);
  const f32 sin_b = MATH::sin(angles.y);
  const f32 sin_c = MATH::sin(angles.x);

  return {
    (cos_a * cos_b), (cos_a * sin_b * sin_c) - (sin_a * cos_c), (cos_a * sin_b * cos_c) + (sin_a * sin_c), 0,
    (sin_a * cos_b), (sin_a * sin_b * sin_c) + (cos_a * cos_c), (sin_a * sin_b * cos_c) - (cos_a * sin_c), 0,
    (-sin_b), (cos_b * sin_c), (cos_b * cos_c), 0,
    0, 0, 0, 1
  };
}

mat4x4 scale_rotate_translate(const f32 scale,
                              const vec3& rotate,
                              const vec3& translate) {
  mat4x4 res = scale_matrix(scale);
  res = mat_mul(res, rotation_matrix(rotate));
  res = mat_mul(res, translation_matrix(translate));

  return res;
}

mat4x4 view_matrix(const quat4& quaternions,
                   const vec3& position) {
  return mat_mul(translation_matrix(neg(position)),
                 rotation_matrix(inverse_normal(quaternions)));
}

mat4x4 view_matrix(const vec3& angles,
                   const vec3& position) {
  return mat_mul(translation_matrix(neg(position)),
                 rotation_matrix(neg(angles)));
}

quat4 quat_from_angles(const vec3& angles) {
  const f32 cos_a = MATH::cos(angles.z * 0.5f);
  const f32 cos_b = MATH::cos(angles.y * 0.5f);
  const f32 cos_c = MATH::cos(angles.x * 0.5f);

  const f32 sin_a = MATH::sin(angles.z * 0.5f);
  const f32 sin_b = MATH::sin(angles.y * 0.5f);
  const f32 sin_c = MATH::sin(angles.x * 0.5f);

  return {
    cos_a * cos_b * cos_c + sin_a * sin_b * sin_c,
    cos_a * cos_b * sin_c - sin_a * sin_b * cos_c,
    cos_a * sin_b * cos_c + sin_a * cos_b * sin_c,
    sin_a * cos_b * cos_c - cos_a * sin_b * sin_c,
  };
}

f32 magnitude(const quat4& q) {
  return MATH::sqrt(q.r * q.r + q.i * q.i + q.j * q.j + q.k * q.k);
}

mat4x4 rotation_matrix(const quat4& q) {
  assert(MATH::approx_eq(magnitude(q), 1, 0.001f));

  //const f32 q_rr = q.r * q.r;
  const f32 q_ii = q.i * q.i;
  const f32 q_jj = q.j * q.j;
  const f32 q_kk = q.k * q.k;

  const f32 q_ri = q.r * q.i;
  const f32 q_rj = q.r * q.j;
  const f32 q_rk = q.r * q.k;
  const f32 q_ij = q.i * q.j;
  const f32 q_ik = q.i * q.k;
  const f32 q_jk = q.j * q.k;

  return {
    1.0f - 2.0f * (q_jj + q_kk), 2.0f * (q_ij - q_rk), 2.0f * (q_ik + q_rj), 0,
    2.0f * (q_ij + q_rk), 1.0f - 2.0f * (q_ii + q_kk), 2.0f * (q_jk - q_ri), 0,
    2.0f * (q_ik - q_rj), 2.0f * (q_jk + q_ri), 1.0f - 2.0f * (q_ii + q_jj), 0,
    0, 0, 0, 1
  };
}

mat4x4 scale_rotate_translate(const f32 scale,
                              const quat4& q,
                              const vec3& translate) {
  mat4x4 res = scale_matrix(scale);
  res = mat_mul(res, rotation_matrix(q));
  res = mat_mul(res, translation_matrix(translate));

  return res;
}

quat4 rotate_around_axis(vec3 axis_normal, rad32 angle) {
  const f32 s_a = MATH::sin(angle * 0.5f);

  return {
    MATH::cos(angle * 0.5f),
    axis_normal.x * s_a,
    axis_normal.y * s_a,
    axis_normal.z * s_a,
  };
}

mat4x4 perspective_projection(f32 aspect, rad32 fov, f32 far_, f32 near_) {
  const f32 inv_depth = 1.0f / (far_ - near_);

  const f32 tan_fov = MATH::tan((PI - fov) * 0.5f);

  mat4x4 proj ={};
  proj.arr[mat4x4::index(0, 0)] = tan_fov / aspect;
  proj.arr[mat4x4::index(1, 1)] = tan_fov;
  proj.arr[mat4x4::index(2, 2)] = (far_ + near_) * inv_depth;

  proj.arr[mat4x4::index(3, 2)] = (2.0f * far_ * near_) * inv_depth;
  proj.arr[mat4x4::index(2, 3)] = -1.0f;

  return proj;
}

static constexpr u32 POW10_U32[] ={
  1,
  10,
  100,
  1000,
  10000,
  100000,
  1000000,
  10000000,
  100000000,
  1000000000,
};

u32 build_u32(const u8* digits, u32 num_digits) {
  u32 val = 0;

  for (u8 i = 0; i < num_digits; i++) {
    u32 d = (u32)digits[num_digits - i - 1];
    val += (d * POW10_U32[i]);
  }

  return val;
}

f32 build_f32(const FloatParseData* float_parse) {
  assert(float_parse->num_digits != 0);//must have digits
  assert(float_parse->digits[0] != 0);//no leading zeros

  //Calculate the decimal versions of the mandissa and exponent
  const u32 mantissa_dec = build_u32(float_parse->digits, float_parse->num_digits);

  const i32 exp_pow_10 = float_parse->exponent
    + (i32)build_u32(float_parse->exp_digits, float_parse->num_exp_digits);

  auto quot = BIG_INT::from(mantissa_dec);

  BIG_INT worker;
  if (exp_pow_10 < 0) {
    auto denom = BIG_INT::pow_10(-exp_pow_10);

    i32 current_range = 0;
    if (denom.less_than(quot)) {
      worker = denom;

      while (worker.less_than(quot)) {
        worker.shift_left_small(1);
        current_range += 1;
      }
    }
    else {
      worker = quot;

      while (worker.less_than(denom)) {
        worker.shift_left_small(1);
        current_range -= 1;
      }
    }

    quot.shift_left(23 - current_range);

    quot.div(denom, &worker);
    denom.unsigned_shift_right_small(1);

    EQ cmp = worker.cmp(denom);
    switch (cmp) {
      case EQ::GREAT: {
          quot.add(1);
          break;
        }
      case EQ::EQ: {
          if (quot.is_even()) {
            quot.add(1);
          }
          break;
        }
    }

    return load_to_float(quot, current_range - 23, float_parse->negative);
  }
  else {
    const auto exponent = BIG_INT::pow_10(exp_pow_10);

    quot.mul(exponent);

    return load_to_float(quot, 0, float_parse->negative);
  }
}

f32 load_to_float(BIG_INT& b, i32 exp_offset, bool negative) {
  u32 sig = b.signif_bit_index();

  if (sig == 0xffffffff) { return 0.0f; }

  BIG_INT remainder = b;
  b.signif_bits(24);

  remainder.sub(b);

  u32 flt = 0;

  if (sig < 24) {
    b.shift_left_small(23 - sig);
    flt = b.arr[0];
  }
  else {
    b.unsigned_shift_right_small(sig - 23);

    flt = b.arr[0];
    u32 diff = sig - 24;

    BIG_INT round = BIG_INT::pow_2(diff);
    round.unsigned_shift_right_small(1);//div 2



    switch (remainder.cmp(round)) {
      case EQ::GREAT: {
          flt += 1;
          break;
        }
      case EQ::EQ: {
          flt += remainder.is_even();
          break;
        }
    }
  }

  flt &= 0x7fffff;//assumed 1st bit

  u32 exp_v = sig + 127 + exp_offset;

  exp_v &= 0xff;
  exp_v <<= 23;

  u32 sign = ((u32)negative) << 31;

  return MATH::from_ulp(sign | exp_v | flt);
}