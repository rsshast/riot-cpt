//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef CHARON_SCATTERING_HPP_
#define CHARON_SCATTERING_HPP_

namespace charon {

//----------------------------------------------------------------------------------------
//! \fn  void scatter
//! \brief TODO(RTW): template on ScatteringModel, when pertinent
KOKKOS_FORCEINLINE_FUNCTION
void ScatterKernel(RngGen &rng_gen, const Real &vv, Real &vx_out, Real &vy_out,
                   Real &vz_out) {
  const Real mu = 2.0 * rng_gen.drand() - 1.0;
  const Real phi = 2.0 * M_PI * rng_gen.drand();
  const Real stheta = std::sqrt(1.0 - mu * mu);
  vx_out = vv * stheta * std::cos(phi);
  vy_out = vv * stheta * std::sin(phi);
  vz_out = vv * mu;
}

KOKKOS_FORCEINLINE_FUNCTION
void ChargedScatterKernel(RngGen &rng_gen, const Real &vv, const Real &A, Real &vx_out, Real &vy_out, Real &vz_out) {
  /*Write the scattering kernel for charged particles
  Pass the PRNG, velocity, and mass (proton or Alpha particle), update components of v */

  // Incident Energy
  const Real Eprime = .5 * A * (vv * vv);
  const Real alpha = std::pow((A - 1) / (A + 1),2.);

  // Outgoing Energy
  int num_iters = 0;
  Real Eout = Eprime;
  Real mu = 0.;
  do { 
    //Sampling direction
    Real mu = 2.0 * rng_gen.drand() - 1.0; //[-1 <= mu <= 1]
    Real numerator = Eprime * (A + 1) * (A + 1);
    Real denominator = A * A + 2 * A * mu + 1;
    Real Eout = numerator / denominator;
    num_iters++;
    if (num_iters == 100) {
      Eout = Eprime;
      break;
    }
  } while (Eout > Eprime && Eout < alpha * Eprime); 
  const Real phi = 2.0 * M_PI * rng_gen.drand();
  const Real stheta = std::sqrt(1.0 - mu * mu);
  Real vout = std::sqrt(2 * Eout / A);
  vx_out = vout * stheta * std::cos(phi);
  vy_out = vout * stheta * std::sin(phi);
  vz_out = vout * mu;

  // Lab frame scattering cosine
  // numerator = (1 + A * mu);
  // denominator = std::sqrt(A * A + 2 * A * mu + 1);
  // double mu_lab = numerator / denominator;
}

} // namespace charon

#endif // CHARON_SCATTERING_HPP_
