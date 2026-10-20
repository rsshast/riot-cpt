//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
//========================================================================================
#ifndef CHARON_SOURCE_PROFILES_HPP_
#define CHARON_SOURCE_PROFILES_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

namespace charon {

// Return the discrete cohort size for the sine_test time profile. N0 is sourced at
// t = 0; later cohorts follow floor(A sin(pi t/tf)). Values very close to an integer
// are snapped to it before flooring so that, for example, the peak is exactly A.
template <typename RealType>
inline int SineTestParticleCount(const RealType time, const RealType final_time,
                                 const int initial_particles,
                                 const int particles_per_pulse) {
  if (time <= RealType(0)) return initial_particles;
  if (time >= final_time) return 0;

  const RealType pi = std::acos(RealType(-1));
  RealType count = static_cast<RealType>(particles_per_pulse) *
                   std::sin(pi * time / final_time);
  const RealType nearest = std::round(count);
  const RealType tolerance = RealType(64) * std::numeric_limits<RealType>::epsilon() *
                             std::max(RealType(1), std::abs(count));
  if (std::abs(count - nearest) <= tolerance) count = nearest;
  return std::max(0, static_cast<int>(std::floor(count)));
}

} // namespace charon

#endif // CHARON_SOURCE_PROFILES_HPP_
