#ifndef MATH_LIB_H
#define MATH_LIB_H
using f32 = float;
using u8 = unsigned char;
using u32 = unsigned int;
using i32 = int;
using usize = size_t;

struct vec3;
struct vec4;
struct quat4;
using rad32 = f32;

inline constexpr u32 U32_MAX = 0xFFFFFFFF;
inline constexpr u32 I32_MAX = 0x7FFFFFFF;

inline constexpr f32 PI =
3.141592653589793238462643383279502884197169399375105820974944592307816406286f;

inline constexpr f32 TAU =
6.28318530717958647692528676655900576839433879875021f;

inline constexpr f32 ROOT_2 =
1.4142135623730950488016887242096980785696718753769480731766797379907324784621f;

inline constexpr f32 RECIP_ROOT_2 =
0.7071067811865475244008443621048490392848359376884740365883398689f;

inline constexpr f32 ROOT_3 =
1.7320508075688772935274463415058723669428052538103806280558069794f;

namespace MATH {
  template<typename T, usize len>
  static inline constexpr bool arr_eq(const T(&t1)[len], const T(&t2)[len]) {
    for (usize i = 0; i < len; i++) {
      if (t1[i] != t2[i]) return false;
    }
    return true;
  }
}

#pragma warning(push)
#pragma warning(disable:4201)

struct vec2 {
  union {
    f32 arr[2] ={ 0 };
    struct {
      f32 x;
      f32 y;
    };
  };
};

struct vec3 {
  union {
    f32 arr[3] ={ 0 };
    struct {
      f32 x;
      f32 y;
      f32 z;
    };
  };

  constexpr bool operator==(const vec3& v) const {
    return x == v.x && y == v.y && z == v.z;
  }

  void scale(f32 f);
  void sub(const vec3& v);
  void add(const vec3& v);
  void piecewise_mul(const vec3& v);
  void normalise();
};

struct vec4 {
  union {
    f32 arr[4] ={ 0 };
    struct {
      f32 x;
      f32 y;
      f32 z;
      f32 w;
    };
  };
};

#pragma warning(pop)

struct quat4 {
  f32 r;
  f32 i;
  f32 j;
  f32 k;

  constexpr bool operator==(const quat4& q) const {
    return q.r == r && q.i == i && q.j == j && q.k == k;
  }

  constexpr static quat4 identity() { return { 1.0f, 0.0f, 0.0f, 0.0f }; }
};

template<usize _N>
struct matNxN {
  static constexpr usize N = _N;
  f32 arr[N * N] ={ 0 };

  constexpr void scale(f32 s) {
    for (usize i = 0; i < (N * N); i++) {
      arr[i] *= s;
    }
  }

  constexpr void add(const matNxN<N>& m) {
    for (usize i = 0; i < (N * N); i++) {
      arr[i] += m.arr[i];
    }
  }

  constexpr bool operator==(const matNxN<N>& m) const {
    return MATHS::arr_eq(arr, m.arr, N * N);
  }

  constexpr static inline usize index(usize x, usize y) {
    return (N * x) + y;
  }

  constexpr static auto identity() {
    matNxN<N> res ={};

    for (usize i = 0; i < N; i++) {
      res.arr[index(i, i)] = 1;
    }

    return res;
  }

  constexpr void transpose() {
    for (u32 x = 0; x < N; x++) {
      for (u32 y = x + 1; y < N; y++) {
        swap(arr[index(x, y)], arr[index(y, x)]);
      }
    }
  }
};

using mat3x3 = matNxN<3>;
static_assert(sizeof(mat3x3) == sizeof(f32[3][3]), "Be the same size");

using mat4x4 = matNxN<4>;
static_assert(sizeof(mat4x4) == sizeof(f32[4][4]), "Be the same size");

constexpr mat4x4 orthographic_projection(f32 right, f32 left,
                                         f32 top, f32 bottom,
                                         f32 far_, f32 near_) {
  const f32 width = right - left;
  const f32 height = top - bottom;
  const f32 depth = far_ - near_;

  mat4x4 proj ={};
  proj.arr[mat4x4::index(0, 0)] = 2.0f / width;
  proj.arr[mat4x4::index(1, 1)] = 2.0f / height;
  proj.arr[mat4x4::index(2, 2)] = 2.0f / depth;

  proj.arr[mat4x4::index(3, 0)] = -(right + left) / width;
  proj.arr[mat4x4::index(3, 1)] = -(top + bottom) / height;
  proj.arr[mat4x4::index(3, 2)] = -(far_ + near_) / depth;

  proj.arr[mat4x4::index(3, 3)] = 1.0f;

  return proj;
}

mat4x4 perspective_projection(f32 aspect, rad32 fov, f32 far_, f32 near_);

constexpr quat4 quat_mul(const quat4& q1, const quat4& q2) {
  return {
    q1.r * q2.r - q1.i * q2.i - q1.j * q2.j - q1.k *  q2.k,
    q1.r * q2.i + q1.i * q2.r + q1.j * q2.k - q1.k *  q2.j,
    q1.r * q2.j - q1.i * q2.k + q1.j * q2.r + q1.k *  q2.i,
    q1.r * q2.k + q1.i * q2.j - q1.j * q2.i + q1.k *  q2.r,
  };
}

constexpr vec4 vec_mul(const mat4x4& m, const vec4& v) {
  return vec4{
    (m.arr[mat4x4::index(0,  0)] * v.x) + (m.arr[mat4x4::index(1,  0)] * v.y)
    + (m.arr[mat4x4::index(2,  0)] * v.z) + (m.arr[mat4x4::index(3,  0)] * v.w),
    (m.arr[mat4x4::index(0,  1)] * v.x) + (m.arr[mat4x4::index(1,  1)] * v.y)
    + (m.arr[mat4x4::index(2,  1)] * v.z) + (m.arr[mat4x4::index(3,  1)] * v.w),
    (m.arr[mat4x4::index(0,  2)] * v.x) + (m.arr[mat4x4::index(1,  2)] * v.y)
    + (m.arr[mat4x4::index(2,  2)] * v.z) + (m.arr[mat4x4::index(3,  2)] * v.w),
    (m.arr[mat4x4::index(0,  3)] * v.x) + (m.arr[mat4x4::index(1,  3)] * v.y)
    + (m.arr[mat4x4::index(2,  3)] * v.z) + (m.arr[mat4x4::index(3,  3)] * v.w),
  };
}

inline constexpr vec3 vec_scale(f32 scale, const vec3& v) {
  return vec3{
    v.x * scale,
    v.y * scale,
    v.z * scale,
  };
}

inline constexpr vec3 vec_piecewise_mul(const vec3& v1, const vec3& v2) {
  return vec3{
    v1.x * v2.x,
    v1.y * v2.y,
    v1.z * v2.z,
  };
}

inline constexpr vec3 vec_sub(const vec3& v1, const vec3& v2) {
  return vec3{
    v1.x - v2.x,
    v1.y - v2.y,
    v1.z - v2.z,
  };
}

inline constexpr vec3 vec_add(const vec3& v1, const vec3& v2) {
  return vec3{
    v1.x + v2.x,
    v1.y + v2.y,
    v1.z + v2.z,
  };
}

inline constexpr vec3 vec_cross(const vec3& v1, const vec3 v2) {
  return  {
    v1.y * v2.z - v1.z * v2.y,
    v1.z * v2.x - v1.x * v2.z,
    v1.x * v2.y - v1.y * v2.x,
  };
}

inline constexpr f32 vec_dot(const vec3& v1, const vec3 v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<usize N>
constexpr auto mat_mul(const matNxN<N>& m1, const matNxN<N>& m2) {
  using MAT = matNxN<N>;

  MAT res;

  for (usize x = 0; x < MAT::N; x++) {
    for (usize y = 0; y < MAT::N; y++) {
      for (usize i = 0; i < MAT::N; i++) {
        res.arr[MAT::index(y, x)] += m1.arr[MAT::index(i, x)] * m2.arr[MAT::index(y, i)];
      }
    }
  }

  return res;
}

constexpr vec4 mat_mul(const mat4x4& m, const vec4& v) {
  vec4 res;

  for (usize y = 0; y < 4; y++) {
    for (usize i = 0; i < 4; i++) {
      res.arr[y] += m.arr[mat4x4::index(y, i)] * v.arr[i];
    }
  }

  return res;
}

constexpr f32 mat_minor(const mat3x3& m, u32 x, u32 y) {
  //Work out the rows not selected
  //e.g.
  //if x == 0 -> x_0 = 1 and x_1 = 2
  //if x == 1 -> x_0 = 0 and x_1 = 2
  //if x == 2 -> x_0 = 0 and x_1 = 1

  const u32 x_0 = 0 + (x == 0);
  const u32 x_1 = 1 + (x <= 1);

  const u32 y_0 = 0 + (y == 0);
  const u32 y_1 = 1 + (y <= 1);

  return ((m.arr[mat3x3::index(x_0, y_0)]
           * m.arr[mat3x3::index(x_1, y_1)])
          - (m.arr[mat3x3::index(x_1, y_0)]
             * m.arr[mat3x3::index(x_0, y_1)]));
}

//a * det[e, f, h, i] - b * det[d, f, g, i] + c * det[d, e, g, h]
#define MAT3X3_DET(a, b, c, d, e, f, g, h, i) (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))

constexpr f32 mat_determinant(const mat3x3& m) {
  using MAT = mat3x3;

  const f32 a = m.arr[MAT::index(0, 0)];
  const f32 b = m.arr[MAT::index(1, 0)];
  const f32 c = m.arr[MAT::index(2, 0)];
  const f32 d = m.arr[MAT::index(0, 1)];
  const f32 e = m.arr[MAT::index(1, 1)];
  const f32 f = m.arr[MAT::index(2, 1)];
  const f32 g = m.arr[MAT::index(0, 2)];
  const f32 h = m.arr[MAT::index(1, 2)];
  const f32 i = m.arr[MAT::index(2, 2)];

  return MAT3X3_DET(a, b, c, d, e, f, g, h, i);
}

constexpr f32 mat_minor(const mat4x4& m, u32 x, u32 y) {
  using MAT = mat4x4;

  //Work out the rows not selected
  //e.g.
  //if x == 0 -> x_0 = 1 and x_1 = 2 and x_2 = 3
  //if x == 1 -> x_0 = 0 and x_1 = 2 and x_2 = 3
  //if x == 2 -> x_0 = 0 and x_1 = 1 and x_2 = 3
  //if x == 3 -> x_0 = 0 and x_1 = 1 and x_2 = 2

  const u32 x_0 = 0 + (x == 0);
  const u32 x_1 = 1 + (x <= 1);
  const u32 x_2 = 2 + (x <= 2);

  const u32 y_0 = 0 + (y == 0);
  const u32 y_1 = 1 + (y <= 1);
  const u32 y_2 = 2 + (y <= 2);

  const f32 a = m.arr[MAT::index(x_0, y_0)];
  const f32 b = m.arr[MAT::index(x_1, y_0)];
  const f32 c = m.arr[MAT::index(x_2, y_0)];
  const f32 d = m.arr[MAT::index(x_0, y_1)];
  const f32 e = m.arr[MAT::index(x_1, y_1)];
  const f32 f = m.arr[MAT::index(x_2, y_1)];
  const f32 g = m.arr[MAT::index(x_0, y_2)];
  const f32 h = m.arr[MAT::index(x_1, y_2)];
  const f32 i = m.arr[MAT::index(x_2, y_2)];

  return MAT3X3_DET(a, b, c, d, e, f, g, h, i);
}

template<usize N>
constexpr f32 mat_cofactor(const matNxN<N>& m, u32 x, u32 y) {
  //cofactor(x, y) = (-1)^(x + y) * minor(x, y)

  // (-1)^(2n) = 1 vs (-1)^(2n + 1) = -1
  if (((x + y) % 2) == 0) {
    return mat_minor(m, x, y);
  }
  else {
    return -mat_minor(m, x, y);
  }
}

constexpr f32 mat_determinant(const mat4x4& m) {
  using MAT = mat4x4;

  f32 det = 0.0f;
  for (u32 x = 0; x < MAT::N; x++) {
    const f32 co = mat_cofactor(m, x, 0);

    det += m.arr[MAT::index(x, 0)] * co;
  }

  return det;
}

template<usize N>
constexpr matNxN<N> mat_inverse(const matNxN<N>& m) {
  using MAT = matNxN<N>;

  //Trasnposed matrix of cofactors
  f32 det = 0.0f;
  MAT res ={};
  if (0 < MAT::N) {
    for (u32 x = 0; x < MAT::N; x++) {
      const f32 co = mat_cofactor(m, x, 0);

      //Build the determinant as we go
      det += m.arr[MAT::index(x, 0)] * co;

      //Transpose by switching x and y for the result
      res.arr[MAT::index(0, x)] = co;
    }
  }

  for (u32 y = 1; y < MAT::N; y++) {
    for (u32 x = 0; x < MAT::N; x++) {
      const f32 co = mat_cofactor(m, x, y);

      //Transpose by switching x and y for the result
      res.arr[MAT::index(y, x)] = co;
    }
  }

  res.scale(1.0f / det);
  return res;
}

constexpr vec3 neg(const vec3& v) {
  return {
    -v.x,
    -v.y,
    -v.z,
  };
}

constexpr quat4 inverse_normal(const quat4& q) {
  return {
    q.r,
    -q.i,
    -q.j,
    -q.k,
  };
}

constexpr vec3 mat_mul(const mat3x3& m, const vec3& v) {
  vec3 res;

  for (usize y = 0; y < 3; y++) {
    for (usize i = 0; i < 3; i++) {
      res.arr[y] += m.arr[mat3x3::index(y, i)] * v.arr[i];
    }
  }

  return res;
}

constexpr mat3x3 cross_product_matrix(const vec3& vec) {
  mat3x3 res;

  res.arr[mat3x3::index(0, 1)] = -vec.z;
  res.arr[mat3x3::index(0, 2)] = vec.y;
  res.arr[mat3x3::index(1, 0)] = vec.z;
  res.arr[mat3x3::index(1, 2)] = -vec.x;
  res.arr[mat3x3::index(2, 0)] = -vec.y;
  res.arr[mat3x3::index(2, 1)] = vec.x;

  return res;
}

constexpr mat3x3 outer_product(const vec3& v1, const vec3 v2) {
  mat3x3 res;

  for (usize x = 0; x < 3; x++) {
    for (usize y = 0; y < 3; y++) {
      res.arr[mat3x3::index(y, x)] = v1.arr[x] * v2.arr[y];
    }
  }

  return res;
}

constexpr quat4 conjugate(const quat4& q) {
  return {
    -q.r,
    -q.i,
    -q.j,
    q.k,
  };
}


constexpr mat4x4 scale_matrix(const f32 scale) {
  return {
    scale, 0, 0, 0,
    0, scale, 0, 0,
    0, 0, scale, 0,
    0, 0, 0, 1,
  };
}

constexpr mat4x4 translation_matrix(const vec3& direction) {
  return {
    1, 0, 0, direction.x,
    0, 1, 0, direction.y,
    0, 0, 1, direction.z,
    0, 0, 0, 1,
  };
}

//TODO: Learn how this actually works
constexpr bool ray_triangle_intersection_culled(const vec3& orig,
                                                const vec3& dir,
                                                const vec3& v0,
                                                const vec3& v1,
                                                const vec3& v2,
                                                f32* out_t,
                                                f32* out_u, f32* out_v) {
  constexpr f32 EPSILON = 0.0001f;

  const vec3 edge1 = vec_sub(v1, v0);
  const vec3 edge2 = vec_sub(v2, v0);

  const vec3 pvec = vec_cross(dir, edge2);
  const f32 det = vec_dot(edge1, pvec);

  //Only front facing triangles
  if (det < EPSILON) return false;

  const vec3 tvec = vec_sub(orig, v0);
  const f32 u = vec_dot(tvec, pvec);
  const vec3 qvec = vec_cross(tvec, edge1);
  const f32 v = vec_dot(dir, qvec);


  if (u < 0.0f || u > det
      || v < 0.0f || u + v > det) return false;

  const f32 t = vec_dot(edge2, qvec);
  const f32 inv_det = 1.0f / det;

  //Assign and check for null
  if (out_t) *out_t = t * inv_det;
  if (out_u) *out_u = u * inv_det;
  if (out_v) *out_v = v * inv_det;
  return true;
}

constexpr u32 find_highest_bit_set(const u32 v) {
  u32 current_max = 32;
  u32 current_min = 0;

  while (current_max - current_min > 1) {
    const u32 mid = (current_max + current_min) / 2;
    const u32 mid_val = (1 << mid);
    if (mid_val > v) {
      current_max = mid;
    }
    else {
      current_min = mid;
    }
  }

  return current_min;
}

constexpr void arbitrary_copy(u32* num1, const u32* num2, u32 num_len) {
  for (u32 i = 0; i < num_len; i++) {
    num1[i] = num2[i];
  }
}

constexpr void arbitrary_zero(u32* num1, u32 num_len) {
  for (u32 i = 0; i < num_len; i++) {
    num1[i] = 0;
  }
}

constexpr void arbitrary_add_no_wrap(u32* num1, const u32* num2, u32 num_len) {
  u32 carry = 0;
  u32 next_carry = 0;
  u32 res = 0;
  u32 oth = 0;

  for (u32 i = 0; i < num_len; i++) {
    next_carry = 0;
    res = num1[i];
    oth = num2[i];

    //Need to do 2 separate additions because
    //each one could overflow leading to a carry

    next_carry += (res > U32_MAX - carry);
    res += carry;

    next_carry += (res > U32_MAX - oth);
    res += oth;

    num1[i] = res;

    carry = next_carry;
  }
}

constexpr void arbitrary_add_no_wrap(u32* num1, u32 num_len, u32 val) {
  u32 carry = val;
  u32 next_carry = 0;
  u32 res = 0;

  for (u32 i = 0; i < num_len; i++) {
    next_carry = 0;
    res = num1[i];

    //Need to do 2 separate additions because
    //each one could overflow leading to a carry

    next_carry += (res > U32_MAX - carry);
    res += carry;

    num1[i] = res;

    carry = next_carry;
  }
}

constexpr void arbitrary_neg(u32* num1, u32 num_len) {
  u32 carry = 0;
  u32 next_carry = 0;
  u32 res = 0;

  for (u32 i = 0; i < num_len; i++) {
    next_carry = 0;
    res = ~num1[i];

    //Need to do 2 separate additions because
    //each one could overflow leading to a carry

    next_carry += (res > U32_MAX - carry);
    res += carry;

    num1[i] = res;

    carry = next_carry;
  }
}

constexpr void arbitrary_sub_no_wrap(u32* num1, const u32* num2, u32 num_len) {
  // a - b = c
  //-(b + (-a)) = c <- allows for a single accumulator

  arbitrary_neg(num1, num_len);
  arbitrary_add_no_wrap(num1, num2, num_len);
  arbitrary_neg(num1, num_len);
}

constexpr void arbitrary_shift_left_large(u32* num1, u32 num_len, u32 shift) {
  //Optimization??
  //if(large_shift > 3) { /* return 0 as cannot shift that far*/ }

  //Shift the larger values across
  //Do this in reverse order to not overwrite
  for (u32 i = num_len; i > shift; i--) {
    num1[i - 1] = num1[i - 1 - shift];
  }

  //Fill zeros below
  for (u32 i = 0; i < num_len && i < shift; i++) {
    num1[i] = 0;
  }
}

//Can only shift small with 31 or less
constexpr void arbitrary_shift_left_small(u32* num1, u32 num_len, u32 shift) {
  shift %= 32;
  if (shift == 0) return;//avoids edge cases

  u32 carry = 0;
  u32 next_carry = 0;
  u32 res = 0;

  for (u32 i = 0; i < num_len; i++) {
    next_carry = 0;
    res = num1[i];

    //Shift is > 0 so this is okay
    next_carry = (res >> (32 - shift));
    num1[i] = (res << shift) | carry;

    carry = next_carry;
  }
}

//Can only shift small with 31 or less
constexpr void arbitrary_unsigned_shift_right_small(u32* num1, u32 num_len, u32 shift) {
  if (shift == 0) return;// avoids edge cases

  u32 carry = 0;
  u32 next_carry = 0;
  u32 res = 0;

  for (u32 i = num_len; i > 0; i--) {
    next_carry = 0;
    res = num1[i - 1];

    next_carry = (res << (32 - shift));
    num1[i - 1] = (res >> shift) | carry;

    carry = next_carry;
  }
}

constexpr void arbitrary_shift_left(u32* num1, u32 num_len, u32 shift) {
  //5 bits are needed to store up to 31
  //Then we do mod that and shift down
  const u32 large_shift = (shift & ~0x1f) >> 5;

  if (large_shift > 0) {
    arbitrary_shift_left_large(num1, num_len, large_shift);
  }

  //5 bits are needed to store up to 31
  const u32 small_shift = (shift & 0x1f);
  arbitrary_shift_left_small(num1, num_len, small_shift);
}

constexpr void arbitrary_multiply(u32* num1, const u32* num2,
                                  u32* worker,
                                  u32 num_len) {
  arbitrary_zero(worker, num_len);

  for (u32 i = 0; i < num_len; i++) {
    u32 current_mul = num2[i];

    //Next section
    if (current_mul == 0) continue;

    u32 next_shift_needed = 0;
    //Could be linear or binary search, etc
    //Will always be valid as we already checked for 0
    u32 highest_set_bit = find_highest_bit_set(current_mul) + 1;

    for (u32 j = 0; j < highest_set_bit; j++, next_shift_needed++) {

      u32 current_shift = (1 << j);

      if ((current_shift & current_mul) != current_shift) {
        continue;
      }

      arbitrary_shift_left_small(num1, num_len, next_shift_needed);
      next_shift_needed = 0;

      arbitrary_add_no_wrap(worker, num1, num_len);
    }

    //Do any shifts remaining to get to next section
    //Will always be small
    arbitrary_shift_left_small(num1, num_len, 32 - (highest_set_bit - 1));
  }

  //Copy it back
  arbitrary_copy(num1, worker, num_len);
}

constexpr u32 arbitrary_significant_bit_index(const u32* num1, u32 num_len) {
  for (u32 i = num_len; i > 0; i--) {
    if (num1[i - 1] != 0) {
      return find_highest_bit_set(num1[i - 1]) + ((i - 1) * 32);
    }
  }

  return 0xffffffff;
}

constexpr void arbitrary_significant_bits(u32* num1, u32 num_len,
                                          u32 bits) {
  for (u32 i = num_len; i > 0; i--) {
    u32 val = num1[i - 1];
    if (val != 0) {
      u32 index = find_highest_bit_set(val);

      if (i <= 1) {
        if (index > bits) {
          //Remove all the lower bits
          val >>= (index - bits + 1);
          val <<= (index - bits + 1);
          num1[i - 1] = val;
        }
      }
      else {
        if (index < bits) {
          bits -= (index + 1);
          index = 31;
          i--;
          val = num1[i - 1];
        }

        if (bits != 0) {
          //Remove all the lower bits
          val >>= (index - bits + 1);
          val <<= (index - bits + 1);
          num1[i - 1] = val;
          i--;
        }
        //Zero whats left
        for (; i > 0; i--) {
          num1[i - 1] = 0;
        }
      }

      return;
    }
  }
}

enum struct EQ {
  LESS,
  GREAT,
  EQ
};


//Worker should be set to 0
constexpr EQ arbitrary_cmp(const u32* num1, const u32* num2,
                           u32 num_len) {
  for (u32 i = num_len; i > 0; i--) {
    if (num1[i - 1] != num2[i - 1])
      return num1[i - 1] < num2[i - 1] ? EQ::LESS : EQ::GREAT;
  }

  return EQ::EQ;
}

constexpr u32 arbitrary_min_set_bit(const u32* num1, u32 num_len) {
  for (u32 b = 0; b < num_len; b++) {
    const u32 bits = num1[b];
    if (bits != 0) {
      for (u32 i = 0; i < 32; i++) {
        const u32 shift = 1 << i;

        if ((shift & bits) == shift) { return i + (b * 32); }
      }
    }
  }

  return num_len * 32;
}

constexpr u32 arbitrary_max_set_bit(const u32* num1, u32 num_len) {
  for (u32 b = num_len; b > 0; b--) {
    const u32 bits = num1[b - 1];
    if (bits != 0) {
      for (u32 i = 32; i > 0; i--) {
        const u32 shift = 1 << (i - 1);

        if ((shift & bits) == shift) { return (i - 1) + ((b - 1) * 32); }
      }
    }
  }

  return num_len * 32;
}

//Worker should be set to 0
constexpr void arbitrary_divide(u32* num1, const u32* num2,
                                u32* mod_out, u32* worker,
                                u32 num_len) {
  arbitrary_copy(mod_out, num1, num_len);
  arbitrary_zero(num1, num_len);
  arbitrary_zero(worker, num_len);

  const u32 max_offset = arbitrary_max_set_bit(num2, num_len);

  while (arbitrary_cmp(num2, mod_out, num_len) != EQ::GREAT) {
    u32 max_shift = (num_len * 32) - max_offset;
    u32 min_shift = 0;


    while (max_shift - min_shift > 1) {
      u32 mid = (max_shift + min_shift) / 2;
      arbitrary_copy(worker, num2, num_len);
      arbitrary_shift_left(worker, num_len, mid);

      if (arbitrary_cmp(worker, mod_out, num_len) == EQ::LESS) {
        min_shift = mid;
      }
      else {
        max_shift = mid;
      }
    }

    arbitrary_copy(worker, num2, num_len);
    arbitrary_shift_left(worker, num_len, min_shift);
    arbitrary_sub_no_wrap(mod_out, worker, num_len);

    arbitrary_zero(worker, num_len);
    worker[0] = 0x1;
    arbitrary_shift_left(worker, num_len, min_shift);
    arbitrary_add_no_wrap(num1, worker, num_len);
  }
}


//A structure big enough to hold a fully multiplied out floating point number
//Needs to be able to hold 2 ^ 127
//128 is 4 32 bit ints
//Little endian
struct BIG_INT {
  u32 arr[4];

  static constexpr BIG_INT from(u32 u) {
    return { u, 0x0, 0x0, 0x0 };
  }

  static constexpr BIG_INT pow_2(u32 p) {
    if (p > (31 + 32 + 32 + 32)) {
      return { 0x0, 0x0, 0x0, 0x0 };
    }
    else if (p > (31 + 32 + 32)) {
      return { 0x0, 0x0, 0x0, (1u << (p - (31 + 32 + 32))) };
    }
    else if (p > (31 + 32)) {
      return { 0x0, 0x0, (1u << (p - (31 + 32))), 0x0 };
    }
    else if (p > 31) {
      return { 0x0, (1u << (p - 31)), 0x0, 0x0 };
    }
    else {
      return { (1u << p), 0x0, 0x0, 0x0 };
    }
  }

  /*
  Caluculated using the python script:

  a = 1
  max_v = pow(2, 128)

  while(a < max_v):
  print(f"{hex(a)}")
  a *= 10


  and then formatted manually :(
  */
  static constexpr u32 POW10_BIG[][4] ={
    { 0x1, 0x0, 0x0, 0x0 },
    { 0xa, 0x0, 0x0, 0x0 },
    { 0x64, 0x0, 0x0, 0x0 },
    { 0x3e8, 0x0, 0x0, 0x0 },
    { 0x2710, 0x0, 0x0, 0x0 },
    { 0x186a0, 0x0, 0x0, 0x0 },
    { 0xf4240, 0x0, 0x0, 0x0 },
    { 0x989680, 0x0, 0x0, 0x0 },
    { 0x5f5e100, 0x0, 0x0, 0x0 },
    { 0x3b9aca00, 0x0, 0x0, 0x0 },
    { 0x540be400, 0x2, 0x0, 0x0 },
    { 0x4876e800, 0x17, 0x0, 0x0 },
    { 0xd4a51000, 0xe8, 0x0, 0x0 },
    { 0x4e72a000, 0x918, 0x0, 0x0 },
    { 0x107a4000, 0x5af3, 0x0, 0x0 },
    { 0xa4c68000, 0x38d7e, 0x0, 0x0 },
    { 0x6fc10000, 0x2386f2, 0x0, 0x0 },
    { 0x5d8a0000, 0x1634578, 0x0, 0x0 },
    { 0xa7640000, 0xde0b6b3, 0x0, 0x0 },
    { 0x89e80000, 0x8ac72304, 0x0, 0x0 },
    { 0x63100000, 0x6bc75e2d, 0x5, 0x0 },
    { 0xdea00000, 0x35c9adc5, 0x36, 0x0 },
    { 0xb2400000, 0x19e0c9ba, 0x21e, 0x0 },
    { 0xf6800000, 0x02c7e14a, 0x152d, 0x0 },
    { 0xa1000000, 0x1bcecced, 0xd3c2, 0x0 },
    { 0x4a000000, 0x16140148, 0x84595, 0x0 },
    { 0xe4000000, 0xdcc80cd2, 0x52b7d2, 0x0 },
    { 0xe8000000, 0x9fd0803c, 0x33b2e3c, 0x0 },
    { 0x10000000, 0x3e250261, 0x204fce5e, 0x0 },
    { 0xa0000000, 0x6d7217ca, 0x431e0fae, 0x1 },
    { 0x40000000, 0x4674edea, 0x9f2c9cd0, 0xc },
    { 0x80000000, 0xc0914b26, 0x37be2022, 0x7e },
    { 0x00000000, 0x85acef81, 0x2d6d415b, 0x4ee },
    { 0x00000000, 0x38c15b0a, 0xc6448d93, 0x314d },
    { 0x00000000, 0x378d8e64, 0xbead87c0, 0x1ed09 },
    { 0x00000000, 0x2b878fe8, 0x72c74d82, 0x134261 },
    { 0x00000000, 0xb34b9f10, 0x7bc90715, 0xc097ce },
    { 0x00000000, 0x00f436a0, 0xd5da46d9, 0x785ee10 },
    { 0x00000000, 0x098a2240, 0x5a86c47a, 0x4b3b4ca8 },
  };

  constexpr static BIG_INT pow_10(u32 p) {
    const auto& p_arr = POW10_BIG[p];
    return { p_arr[0], p_arr[1], p_arr[2], p_arr[3] };
  }

  constexpr bool is_even() const {
    return (arr[0] & 1) == 0;
  }

  constexpr bool less_than(const BIG_INT& bi) const {
    return arbitrary_cmp(arr, bi.arr, 4) == EQ::LESS;
  }

  constexpr EQ cmp(const BIG_INT& bi) const {
    return arbitrary_cmp(arr, bi.arr, 4);
  }

  constexpr void neg() {
    arbitrary_neg(arr, 4);
  }

  constexpr void add(const BIG_INT& bi) {
    arbitrary_add_no_wrap(arr, bi.arr, 4);
  }

  constexpr void add(u32 a) {
    arbitrary_add_no_wrap(arr, 4, a);
  }

  constexpr void sub(const BIG_INT& bi) {
    arbitrary_sub_no_wrap(arr, bi.arr, 4);
  }

  constexpr void shift_left(u32 i) {
    arbitrary_shift_left(arr, 4, i);
  }

  constexpr void shift_left_small(u32 i) {
    arbitrary_shift_left_small(arr, 4, i);
  }

  constexpr void unsigned_shift_right_small(u32 i) {
    arbitrary_unsigned_shift_right_small(arr, 4, i);
  }


  constexpr void mul(const BIG_INT& bi) {
    BIG_INT worker ={ 0x0, 0x0, 0x0, 0x0 };
    arbitrary_multiply(arr, bi.arr, worker.arr, 4);
  }

  constexpr void div(const BIG_INT& bi) {
    BIG_INT mod_out_loc ={ 0x0, 0x0, 0x0, 0x0 };
    BIG_INT worker ={ 0x0, 0x0, 0x0, 0x0 };

    arbitrary_divide(arr, bi.arr, mod_out_loc.arr, worker.arr, 4);
  }

  constexpr void div(const BIG_INT& bi, BIG_INT* mod_out) {
    BIG_INT worker ={ 0x0, 0x0, 0x0, 0x0 };

    arbitrary_divide(arr, bi.arr, mod_out->arr, worker.arr, 4);
  }

  constexpr u32 signif_bit_index() const {
    return arbitrary_significant_bit_index(arr, 4);
  }

  constexpr void signif_bits(u32 bits) {
    arbitrary_significant_bits(arr, 4, bits);
  }
};

namespace MATH {
  constexpr inline u32 abs(i32 i) {
    i32 mask = i >> 31;
    i ^= mask;
    return (i - mask);
  }

  constexpr inline u32 abs(u32 u) { return u; }
  f32 abs(f32);

  f32 maximum(f32 a, f32 b);
  rad32 arcsin(f32);
  rad32 arctan(f32);
  f32 sin(rad32);
  f32 cos(rad32);
  f32 tan(rad32);
  f32 sqrt(f32);
  f32 log2(f32);
  i32 round(f32);
  i32 ulp(f32);
  f32 from_ulp(u32);

  bool is_nan(f32);
  bool is_inf(f32);
  bool is_nan_or_inf(f32);

  bool abs_diff_eq(f32 a, f32 b, f32 max_abs_diff);
  bool rel_diff_eq(f32 a, f32 b, f32 max_rel_diff);

  u32 ulp_difference(f32 a, f32 b);

  bool approx_eq(f32 a, f32 b, f32 max_abs_diff);
  bool approx_eq(const vec3& v1, const vec3& v2, f32 max_abs_diff);
  bool approx_eq(const vec4& v1, const vec4& v2, f32 max_abs_diff);
  bool approx_eq(const quat4& q1, const quat4& q2, f32 max_abs_diff);

  template<usize N>
  bool approx_eq(const matNxN<N>& m1, const matNxN<N>& m2,
                 f32 max_abs_diff) {
    using MAT = matNxN<N>;

    for (usize x = 0; x < MAT::N; x++) {
      for (usize y = 0; y < MAT::N; y++) {
        if (!approx_eq(m1.arr[MAT::index(x, y)],
                       m2.arr[MAT::index(x, y)],
                       max_abs_diff)) {
          return false;
        }
      }
    }

    return true;
  }
}

quat4 quat_from_angles(const vec3& v);
mat4x4 scale_rotate_translate(const f32 scale,
                              const quat4& q,
                              const vec3& translate);

f32 magnitude(const quat4& q);
mat3x3 rotate_around_axis_matrix(vec3 axis, rad32 theta);
mat4x4 rotation_matrix(const quat4& q);
mat4x4 rotation_matrix(const vec3& angles);
mat4x4 scale_rotate_translate(const f32 scale,
                              const vec3& rotate,
                              const vec3& translate);
mat4x4 view_matrix(const quat4& quaternion, const vec3& position);
mat4x4 view_matrix(const vec3& angles, const vec3& position);
quat4 rotate_around_axis(vec3 axis_normal, rad32 angle);

//Digits stored with the most significant digit at digits[0]
struct FloatParseData {
  bool negative;

  u8 num_digits;
  u8 num_exp_digits;

  const u8* digits;
  const u8* exp_digits;
  i32 exponent;
};

f32 build_f32(const FloatParseData* float_parse);

//Digits stored with the most significant digit at digits[0]
u32 build_u32(const u8* digits, u32 num_digits);

f32 load_to_float(BIG_INT& b, i32 exp_offset, bool negative);
#endif