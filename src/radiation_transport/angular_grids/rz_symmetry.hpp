// This file was made in part with generative AI.
#ifndef RADIATION_TRANSPORT_RZ_SYMMETRY_HPP_
#define RADIATION_TRANSPORT_RZ_SYMMETRY_HPP_
#include <ports-of-call/portability.hpp>

namespace RZSymmetry {
// Latlon indexing is theta*nphi + phi. Axis continuation rotates the local
// cylindrical frame by pi, reversing BOTH R and azimuthal components.
// Equatorial reflection reverses only the Z component (cos(theta)).
PORTABLE_INLINE_FUNCTION int AxisAngle(int angle, int nphi) {
  return (angle / nphi) * nphi + (angle % nphi + nphi / 2) % nphi;
}
PORTABLE_INLINE_FUNCTION int EquatorAngle(int angle, int ntheta, int nphi) {
  return (ntheta - 1 - angle / nphi) * nphi + angle % nphi;
}
} // namespace RZSymmetry
#endif
