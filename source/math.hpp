#pragma once

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Vec3 {
	float x = 0.0f, y = 0.0f, z = 0.0f;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(const Vec3& v, float s) { return { v.x * s, v.y * s, v.z * s }; }

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
	return { a.y * b.z - a.z * b.y,
	         a.z * b.x - a.x * b.z,
	         a.x * b.y - a.y * b.x };
}

inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }

inline Vec3 normalize(const Vec3& v) {
	const float l = length(v);
	return l > 0.0f ? v * (1.0f / l) : v;
}

struct Vec4 {
	float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
};

struct Mat4 {
	float m[16] = {};

	float& at(int row, int col) { return m[col * 4 + row]; }
	float at(int row, int col) const { return m[col * 4 + row]; }
};

inline Mat4 identity() {
	Mat4 r;
	r.at(0, 0) = r.at(1, 1) = r.at(2, 2) = r.at(3, 3) = 1.0f;
	return r;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.at(row, k) * b.at(k, col);
			}
			r.at(row, col) = sum;
		}
	}
	return r;
}

inline Vec4 operator*(const Mat4& a, const Vec4& v) {
	return { a.at(0, 0) * v.x + a.at(0, 1) * v.y + a.at(0, 2) * v.z + a.at(0, 3) * v.w,
	         a.at(1, 0) * v.x + a.at(1, 1) * v.y + a.at(1, 2) * v.z + a.at(1, 3) * v.w,
	         a.at(2, 0) * v.x + a.at(2, 1) * v.y + a.at(2, 2) * v.z + a.at(2, 3) * v.w,
	         a.at(3, 0) * v.x + a.at(3, 1) * v.y + a.at(3, 2) * v.z + a.at(3, 3) * v.w };
}

inline Mat4 translate(const Vec3& t) {
	Mat4 r = identity();
	r.at(0, 3) = t.x;
	r.at(1, 3) = t.y;
	r.at(2, 3) = t.z;
	return r;
}

inline Mat4 scale(const Vec3& s) {
	Mat4 r = identity();
	r.at(0, 0) = s.x;
	r.at(1, 1) = s.y;
	r.at(2, 2) = s.z;
	return r;
}

inline Mat4 rotateX(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);
	Mat4 r = identity();
	r.at(1, 1) = c; r.at(1, 2) = -s;
	r.at(2, 1) = s; r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateY(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);
	Mat4 r = identity();
	r.at(0, 0) = c;  r.at(0, 2) = s;
	r.at(2, 0) = -s; r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateZ(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);
	Mat4 r = identity();
	r.at(0, 0) = c; r.at(0, 1) = -s;
	r.at(1, 0) = s; r.at(1, 1) = c;
	return r;
}

inline Mat4 rotate(const Vec3& angles) {
	return rotateZ(angles.z) * rotateY(angles.y) * rotateX(angles.x);
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
	const Vec3 f = normalize(target - eye);
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 r = identity();
	r.at(0, 0) = s.x; r.at(0, 1) = s.y; r.at(0, 2) = s.z;
	r.at(1, 0) = u.x; r.at(1, 1) = u.y; r.at(1, 2) = u.z;
	r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z;
	r.at(0, 3) = -dot(s, eye);
	r.at(1, 3) = -dot(u, eye);
	r.at(2, 3) = dot(f, eye);
	return r;
}

inline Mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float t = std::tan(fov_y * 0.5f);

	Mat4 r;
	r.at(0, 0) = 1.0f / (aspect * t);
	r.at(1, 1) = -1.0f / t;
	r.at(2, 2) = z_far / (z_near - z_far);
	r.at(2, 3) = (z_far * z_near) / (z_near - z_far);
	r.at(3, 2) = -1.0f;
	return r;
}

inline Mat4 ortho(float left, float right, float bottom, float top, float z_near, float z_far) {
	Mat4 r = identity();
	r.at(0, 0) = 2.0f / (right - left);
	r.at(1, 1) = -2.0f / (top - bottom);
	r.at(2, 2) = -1.0f / (z_far - z_near);
	r.at(0, 3) = -(right + left) / (right - left);
	r.at(1, 3) = (top + bottom) / (top - bottom);
	r.at(2, 3) = -z_near / (z_far - z_near);
	return r;
}

}
