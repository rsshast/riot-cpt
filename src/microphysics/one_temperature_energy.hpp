// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#ifndef MICROPHYSICS_ONE_TEMPERATURE_ENERGY_HPP_
#define MICROPHYSICS_ONE_TEMPERATURE_ENERGY_HPP_

#include <cmath>
#include <ports-of-call/portability.hpp>

namespace OneTemperature {

// Common-T equilibrium of constant-Cv ideal-gas mixtures with arbitrary Gamma.
// pressure_coefficient[m] = P_m/(rho_m*T) = (Gamma_m-1)*Cv_m.
template <typename Real>
PORTABLE_INLINE_FUNCTION void IdealGasMixtureFromEnergy(
    const Real energy, const int nmat, const Real *mass_density,
    const Real *cv, const Real *pressure_coefficient,
    Real &temperature, Real &pressure, Real *volume_fraction) {
  Real capacity = 0, pressure_sum = 0;
  for (int m = 0; m < nmat; ++m) {
    if (mass_density[m] <= 0) continue;
    capacity += mass_density[m] * cv[m];
    pressure_sum += mass_density[m] * pressure_coefficient[m];
  }
  temperature = capacity > 0 ? energy / capacity : 1;
  pressure = temperature * pressure_sum;
  for (int m = 0; m < nmat; ++m) {
    volume_fraction[m] = mass_density[m] > 0 && pressure_sum > 0
                             ? mass_density[m] * pressure_coefficient[m] / pressure_sum
                             : 0;
  }
}

// The monatomic 1T closure has P = 2u/3. Compute its forward T -> u map
// at fixed cell-averaged material masses, including pressure equilibrium.
// charge(m, rho, T) supplies the SAME mean-ionization model as the closure.
// ci and ce are specific heats; ce is evaluated at mean charge one.
template <typename Real, typename Charge>
PORTABLE_INLINE_FUNCTION Real InternalEnergyFromTemperature(
    const Real temperature, const int nmat, const Real *mass_density,
    const Real *ci, const Real *ce, const Real *nuclear_charge, const bool *full,
    const Charge &charge) {
  Real cmin = 0, cmax = 0;
  bool all_full = true;
  for (int m = 0; m < nmat; ++m) {
    if (mass_density[m] <= 0) continue;
    cmin += mass_density[m] * ci[m];
    cmax += mass_density[m] * (ci[m] + ce[m] * nuclear_charge[m]);
    all_full = all_full && full[m];
  }
  if (all_full || cmin == 0) return cmax * temperature;

  // base = P / ((2/3) T). For a trial pressure, each material density
  // obeys rho_m (ci_m + z_m ce_m) = base and sum(mass_density/rho_m) = 1.
  // These are the same equations inverted by ApplyOneTemperatureClosure.
  Real lo = cmin, hi = cmax;
  for (int it = 0; it < 64; ++it) {
    const Real base = 0.5 * (lo + hi);
    Real volume_fraction = 0;
    for (int m = 0; m < nmat; ++m) {
      if (mass_density[m] <= 0) continue;
      Real rho_lo = base / (ci[m] + nuclear_charge[m] * ce[m]);
      Real rho_hi = base / ci[m];
      if (!full[m]) {
        for (int jt = 0; jt < 64; ++jt) {
          const Real rho = 0.5 * (rho_lo + rho_hi);
          const Real z = charge(m, rho, temperature);
          if (rho * (ci[m] + ce[m] * z) > base) rho_hi = rho;
          else rho_lo = rho;
          if (rho_hi - rho_lo < 1.e-12 * rho_hi) break;
        }
      } else {
        rho_hi = rho_lo;
      }
      volume_fraction += mass_density[m] / (0.5 * (rho_lo + rho_hi));
    }
    if (volume_fraction > 1) lo = base;
    else hi = base;
    if (hi - lo < 1.e-12 * hi) break;
  }
  return 0.5 * (lo + hi) * temperature;
}

} // namespace OneTemperature
#endif // MICROPHYSICS_ONE_TEMPERATURE_ENERGY_HPP_
