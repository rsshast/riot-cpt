#ifndef CHARON_STOPPING_POWER_
#define CHARON_STOPPING_POWER_

// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.

#include <algorithm>
#include <cmath>
#include <math.h>
#include <units.hpp>

namespace charon {

using Real = parthenon::Real;
using namespace units;

class QuadStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit QuadStoppingPower(Real val = Real(0.0)) : value_(val) {}

  KOKKOS_INLINE_FUNCTION
  QuadStoppingPower GetOnDevice() const { return *this; }

  // Constant mean dE/dx
  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real dt) const {
    Real dt2 = dt * dt;
    Real E_loss = value_ * dt2 * dx; // already multiplied by distance here.
    Real E_f = E_in - E_loss;
    const Real edep = E_in - E_f;
    const Real max_edep = E_in - EB;
    return (edep < max_edep) ? edep : max_edep; // GPU compatible
  }

 private:
  Real value_;
};

class DensityLinearStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit DensityLinearStoppingPower(Real base = Real(0.0), Real rho_ref = Real(1.0),
                                      Real slope = Real(1.0))
      : base_(base), rho_ref_(rho_ref), slope_(slope) {}

  KOKKOS_INLINE_FUNCTION
  DensityLinearStoppingPower GetOnDevice() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  Real Value(Real rho) const {
    const Real ref = (rho_ref_ > 0.0) ? rho_ref_ : 1.0;
    const Real factor = 1.0 + slope_ * (rho / ref - 1.0);
    return base_ * ((factor > 0.0) ? factor : Real(0.0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real rho) const {
    const Real E_loss = Value(rho) * dx;
    const Real E_f = E_in - E_loss;
    return (E_f > EB) ? E_f : EB; // GPU compatible
  }

 private:
  Real base_;
  Real rho_ref_;
  Real slope_;
};

class TemperatureLinearStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TemperatureLinearStoppingPower(Real base = Real(0.0),
                                          Real temperature_ref = Real(1.0),
                                          Real slope = Real(1.0))
      : base_(base), temperature_ref_(temperature_ref), slope_(slope) {}

  KOKKOS_INLINE_FUNCTION
  TemperatureLinearStoppingPower GetOnDevice() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  Real Value(Real temperature) const {
    const Real ref = (temperature_ref_ > 0.0) ? temperature_ref_ : 1.0;
    const Real factor = slope_ * (ref / temperature);
    // const Real factor = 1.0 + slope_ * (temperature / ref - 1.0);
    return base_ * ((factor > 0.0) ? factor : Real(0.0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real temperature) const {
    const Real E_loss = Value(temperature) * dx;
    const Real E_f = E_in - E_loss;
    return (E_f > EB) ? E_f : EB; // GPU compatible
  }

 private:
  Real base_;
  Real temperature_ref_;
  Real slope_;
};

// Multispecies LP and shared continuous-slowing helpers.
// Gaussian CGS; positive S means loss per unit relative path length [erg/cm].
// LUMOS equations 30--38. The Coulomb logarithm is used ONCE.
struct PlasmaStopping {
  static constexpr double pi = 3.14159265358979323846;

  KOKKOS_INLINE_FUNCTION
  static Real Psi(Real x) {
    if (x <= 0.0) return 0.0;
    if (x < 1.e-3) {
      // Avoid erf(sqrt(x)) - 2 sqrt(x/pi) exp(-x) cancellation.
      return 4.0 / (3.0 * std::sqrt(pi)) * x * std::sqrt(x) *
             (1.0 - 0.6 * x + 3.0 / 14.0 * x * x - x * x * x / 18.0);
    }
    return std::erf(std::sqrt(x)) - 2.0 / std::sqrt(pi) * std::sqrt(x) * std::exp(-x);
  }

  KOKKOS_INLINE_FUNCTION
  static Real Screening(Real density, Real charge, Real temperature) {
    if (density == 0.0 || charge == 0.0) return 0.0;
    return 4.0 * pi * density * charge * charge * units::Q * units::Q /
           (units::KB * temperature);
  }

  KOKKOS_INLINE_FUNCTION
  static Real CoulombLog(Real inverse_debye_squared, Real mass, Real charge,
                         Real relative_speed_squared, Real projectile_charge = 2.0) {
    const Real reduced_mass = MALPHA * mass / (MALPHA + mass);
    const Real classical = std::abs(projectile_charge * charge) * Q * Q /
                           (reduced_mass * relative_speed_squared);
    const Real quantum = HBAR / (2.0 * reduced_mass * std::sqrt(relative_speed_squared));
    const Real rmin = std::max(classical, quantum);
    return std::log(1.0 / (std::sqrt(inverse_debye_squared) * rmin));
  }

  // Shared LP kernel. Callers supply their background/screening closure; this
  // algebra is identical for the legacy electron interface and isotope channels.
  KOKKOS_INLINE_FUNCTION
  static Real Evaluate(Real v2, Real density, Real mass, Real charge, Real temperature,
                       Real log_lambda, Real projectile_charge = 2.0) {
    const Real x = mass * v2 / (2.0 * KB * temperature);
    const Real psi = Psi(x);
    const Real derivative = 2.0 / std::sqrt(pi) * std::sqrt(x) * std::exp(-x);
    // Algebraically combine G*logLambda; the Coulomb logarithm appears once.
    const Real g_log =
        psi * log_lambda - mass / MALPHA * (derivative * log_lambda - psi - derivative);
    const Real collective = x > 1.0 ? std::log(1.123 * std::sqrt(x)) : 0.0;
    const Real omega2 = 4.0 * pi * density * charge * charge * Q * Q / mass;
    return projectile_charge * projectile_charge * Q * Q / v2 * omega2 *
           (g_log + collective);
  }

  KOKKOS_INLINE_FUNCTION
  static Real Channel(Real energy, Real density, Real mass, Real charge, Real temperature,
                      Real inverse_debye_squared) {
    using namespace units;
    if (density == 0.0 || charge == 0.0) return 0.0;
    PARTHENON_REQUIRE(energy > 0.0 && density > 0.0 && mass > 0.0 && temperature > 0.0 &&
                          inverse_debye_squared > 0.0,
                      "LP requires positive finite plasma inputs");
    const Real v2 = 2.0 * energy / MALPHA;
    // Maxwellian mean-square relative speed, including background thermal motion.
    const Real relative2 = v2 + 3.0 * KB * temperature / mass;
    const Real log_lambda = CoulombLog(inverse_debye_squared, mass, charge, relative2);
    PARTHENON_REQUIRE(std::isfinite(log_lambda) && log_lambda > 0.0,
                      "LP outside supported Coulomb-logarithm regime");
    const Real result = Evaluate(v2, density, mass, charge, temperature, log_lambda);
    PARTHENON_REQUIRE(std::isfinite(result) && result >= 0.0,
                      "LP gives invalid/net-heating channel: thermal diffusion omitted");
    return result;
  }
};

// Compatibility interface for the original fixed-electron-background model.
// Preserve its projectile-temperature screening and nonthermal impact speed;
// use the same Coulomb-log and LP algebra as the multispecies model above.
class LiPetrassoStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit LiPetrassoStoppingPower(int charge = 2, Real ne = 1.0, Real na = 1.0,
                                   Real temperature = 1.0)
      : charge_(charge), ne_(ne), na_(na), temperature_(temperature) {}

  KOKKOS_INLINE_FUNCTION
  LiPetrassoStoppingPower GetOnDevice() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  Real LP_Spower(Real energy) const {
    if (energy <= EB || ne_ == 0.0) return 0.0;
    const Real alpha_temperature = 2.0 * energy / (3.0 * KB);
    const Real screening = PlasmaStopping::Screening(ne_, 1.0, temperature_) +
                           PlasmaStopping::Screening(na_, charge_, alpha_temperature);
    const Real v2 = 2.0 * energy / MALPHA;
    const Real log_lambda =
        PlasmaStopping::CoulombLog(screening, MBETA, 1.0, v2, charge_);
    return PlasmaStopping::Evaluate(v2, ne_, MBETA, 1.0, temperature_, log_lambda,
                                    charge_);
  }

  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real energy, Real stopping, Real distance) const {
    return std::max(EB, energy - stopping * distance);
  }

  KOKKOS_INLINE_FUNCTION
  Real Edep(Real initial, Real final) const {
    return std::min(initial - final, initial - EB);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Real BackgroundEnergy() const { return EB; }

 private:
  int charge_;
  Real ne_, na_, temperature_;
};

struct DragStep {
  Real dt, speed, electron_stopping, ion_stopping;
};

// Embedded Euler/midpoint estimate; velocity is relative to a frozen fluid frame.
template <typename Stopping>
KOKKOS_INLINE_FUNCTION DragStep TakeDragStep(Real energy, Real cutoff, Real h,
                                             Real tolerance, const Stopping &stopping) {
  const Real speed = std::sqrt(2.0 * energy / units::MALPHA);
  Real se, si;
  stopping(energy, se, si);
  const Real total = se + si;
  for (int retry = 0; retry < 48; ++retry) {
    const Real half = std::max(Real(0), speed - 0.5 * total * h / units::MALPHA);
    Real mid_e, mid_i;
    stopping(0.5 * units::MALPHA * half * half, mid_e, mid_i);
    const Real final_speed = speed - (mid_e + mid_i) * h / units::MALPHA;
    const Real euler = speed - total * h / units::MALPHA;
    const Real final_energy = 0.5 * units::MALPHA * final_speed * final_speed;
    if (final_speed >= 0 && final_energy >= cutoff &&
        std::abs(final_speed - euler) <= tolerance * speed)
      return {h, final_speed, mid_e, mid_i};
    h *= 0.5;
  }
  PARTHENON_FAIL("LP integration failed to converge");
  return {0, 0, 0, 0};
}

// Accumulator is physical particles, not histories. No production is discarded.
KOKKOS_INLINE_FUNCTION
Real SourceHistoryCount(Real produced, Real weight, Real &pending) {
  const Real available = pending + produced;
  const Real count = std::floor(available / weight);
  pending = available - count * weight;
  return count;
}

} // namespace charon

namespace mcblock {
using DensityLinearStoppingPower = ::charon::DensityLinearStoppingPower;
using TemperatureLinearStoppingPower = ::charon::TemperatureLinearStoppingPower;
using LPStoppingPower = ::charon::LiPetrassoStoppingPower;
} // namespace mcblock
#endif
