// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#ifndef CPT_SPHERICAL_GEOMETRY_HPP_
#define CPT_SPHERICAL_GEOMETRY_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace CPT::spherical {
using Vector = std::array<double, 3>;
inline double Dot(const Vector &a, const Vector &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline double Radius(const Vector &x) { return std::sqrt(Dot(x, x)); }
// Stable quadratic roots, including the second outer-sphere crossing of an
// inward ray that misses the inner sphere. R=0 is not a physical boundary.
inline double SphereCrossing(const Vector &x, const Vector &direction, double r) {
  if (r <= 0) return std::numeric_limits<double>::infinity();
  const double a = Dot(direction, direction), b = Dot(x, direction);
  if (a == 0) return std::numeric_limits<double>::infinity();
  const double c = (Radius(x) - r) * (Radius(x) + r);
  const double discriminant = b * b - a * c;
  if (discriminant <= 0) return std::numeric_limits<double>::infinity();
  const double q = -b - std::copysign(std::sqrt(discriminant), b);
  const double roots[2] = {q / a, q != 0 ? c / q : -b / a};
  double result = std::numeric_limits<double>::infinity();
  for (double s : roots)
    if (s > 0) result = std::min(result, s);
  return result;
}
inline double ShellCrossing(const Vector &x, const Vector &direction, double inner,
                            double outer) {
  return std::min(SphereCrossing(x, direction, inner),
                  SphereCrossing(x, direction, outer));
}
inline double BirthRadius(double inner, double outer, double uniform) {
  return std::cbrt(inner * inner * inner +
                   uniform * (outer * outer * outer - inner * inner * inner));
}
struct ConstantSegment {
  double dt, distance, speed, loss;
  bool hit_wall;
};
inline ConstantSegment ConstantDragSegment(double speed, double mass, double stopping,
                                           double remaining, double wall) {
  const double acceleration = stopping / mass;
  const double lifetime = stopping > 0 ? speed / acceleration
                                      : std::numeric_limits<double>::infinity();
  double dt = std::min(remaining, lifetime);
  double distance = speed * dt - 0.5 * acceleration * dt * dt;
  const bool hit_wall = wall <= distance;
  if (hit_wall) {
    distance = wall;
    const double next_speed = std::sqrt(std::max(0.0, speed * speed - 2 * acceleration * wall));
    dt = 2 * wall / (speed + next_speed);
  }
  const double next_speed = !hit_wall && dt == lifetime ? 0.0
                                                       : std::max(0.0, speed - acceleration * dt);
  return {dt, distance, next_speed,
          0.5 * mass * (speed - next_speed) * (speed + next_speed), hit_wall};
}
inline unsigned long long FirstSourceForRank(unsigned long long sequence,
                                             unsigned long long rank,
                                             unsigned long long ranks) {
  return (rank + ranks - sequence % ranks) % ranks;
}
} // namespace CPT::spherical
#endif
