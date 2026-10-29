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
// This file was made in part with generative AI.

#ifndef CHARON_TRANSPORT_UTILS_HPP_
#define CHARON_TRANSPORT_UTILS_HPP_

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "charon_utils.hpp"

namespace charon {
using namespace units;

// Interior-cell workspace for the single mesh-data partition used by RadiationStep.
// Sourcing refreshes counts on every call, including calls that create no particles.
// These arrays are not mesh fields: they have no ghosts, output, or communication.
struct TransportScratch {
  parthenon::ParArray2D<int> source_count;
  parthenon::ParArray2D<int> source_prefix;
  parthenon::ParArray2D<int> source_cell_list;
  parthenon::ParArray1D<int> source_cell_count;
  parthenon::ParArray1D<int> particles_per_block;
  parthenon::ParArray1D<parthenon::NewParticlesContext> new_contexts;
  decltype(particles_per_block.GetHostMirror()) particles_per_block_host;
  decltype(new_contexts.GetHostMirror()) new_contexts_host;
  // Weak references detect a remesh without retaining old mesh blocks.
  std::vector<std::weak_ptr<parthenon::MeshBlock>> inactive_blocks;
  parthenon::ParArray2D<parthenon::Real> active_count;
  parthenon::ParArray2D<parthenon::Real> active_weight;
  parthenon::ParArray2D<parthenon::Real> survivor_weight;
  int source_blocks = -1, source_cells = -1;
  int population_blocks = -1, population_cells = -1;

  void ResizeSource(int blocks, int cells) {
    if (blocks == source_blocks && cells == source_cells) return;
    Kokkos::fence();
    source_count = parthenon::ParArray2D<int>("Charon source counts", blocks, cells);
    source_prefix = parthenon::ParArray2D<int>("Charon source prefixes", blocks, cells);
    source_cell_list = parthenon::ParArray2D<int>("Charon source cells", blocks, cells);
    source_cell_count = parthenon::ParArray1D<int>("Charon source cell counts", blocks);
    particles_per_block = parthenon::ParArray1D<int>("Charon source particles", blocks);
    new_contexts = parthenon::ParArray1D<parthenon::NewParticlesContext>(
        "Charon new particle contexts", blocks);
    particles_per_block_host = particles_per_block.GetHostMirror();
    new_contexts_host = new_contexts.GetHostMirror();
    source_blocks = blocks;
    source_cells = cells;
  }

  void ResizePopulation(int blocks, int cells) {
    if (blocks == population_blocks && cells == population_cells) return;
    Kokkos::fence();
    active_count = parthenon::ParArray2D<parthenon::Real>(
        "Charon active counts", blocks, cells);
    active_weight = parthenon::ParArray2D<parthenon::Real>(
        "Charon active weights", blocks, cells);
    survivor_weight = parthenon::ParArray2D<parthenon::Real>(
        "Charon survivor weights", blocks, cells);
    population_blocks = blocks;
    population_cells = cells;
  }
};

KOKKOS_FORCEINLINE_FUNCTION
constexpr Real eps_imc_offset() { return 1.0e6 * EPS; }

KOKKOS_FORCEINLINE_FUNCTION
constexpr Real eps_ddmc_offset() { return 1.0e8 * EPS; }

KOKKOS_FORCEINLINE_FUNCTION
void sample_face_iso_dir(const Real vv, RngGen &rng_gen, Real &v1, Real &v2, Real &v3) {

  // sample principle direction and azimuthal angle about principle direction
  const Real mu = std::sqrt(rng_gen.drand());
  const Real nu = std::sqrt(1.0 - mu * mu);
  const Real phi = 2.0 * M_PI * rng_gen.drand();

  // v1 is principle velocity component
  v1 = vv * mu;
  v2 = vv * nu * std::cos(phi);
  v3 = vv * nu * std::sin(phi);
}

// helper struct to encapsulate transport step arguments
struct tran_step_args {
  // Constants
  RngGen &rng_gen;     // ref to random number generator
  const Real &t_start; // sim time at beginning of time step
  const Real &dt;      // sim time step size/duration
  const Real &ee;      // initial energy
  Real &ee_f;          // final energy
  // const Real &vv;      // particle speed 
  Real &vv_i;          // initial particle speed 
  Real &vv_f;          // final particle speed 
  Real &vx;            // particle x/X1-direction speed
  Real &vy;            // particle y/X2-direction speed
  Real &vz;            // particle z/X3-direction speed
  const Real &dx_push; // minimum spatial cell dimension
  const bool &multi_d; // 2D or 3D
  const bool &three_d; // 3D
  const Real &xl;      // min-x bound (at face) of cell
  const Real &yl;      // min-y bound (at face) of cell
  const Real &zl;      // min-z bound (at face) of cell
  const Real &xu;      // max-x bound (at face) of cell
  const Real &yu;      // max-y bound (at face) of cell
  const Real &zu;      // max-z bound (at face) of cell
  const Real &spower;  // stopping power, required for dt_push
  // Updated during push
  Real &t;            // particle time
  Real &x;            // particle x/X1-coordinate
  Real &y;            // particle y/X2-coordinate
  Real &z;            // particle z/X3-coordinate
  bool &in_background; // indicator for ending the loop
};

/*
KOKKOS_INLINE_FUNCTION
Real cpt_dt_push(tran_step_args tra, Real &dx_push) {
  const Real Edot = tra.spower * tra.vv_i; // works for small timesteps and small velo changes
  Real dt_push = 2. * tra.ee / Edot * std::log(1. + Edot * dx_push / (2. * tra.ee * tra.vv_i));
  const Real tf = tra.t_start + tra.dt;
  Real t_step = tra.t + dt_push;
  Real dx_i = dx_push;

  if (t_step <= tf) {
    return dt_push;
  } else {
      // from conservation of momentum, vf = vi - spower * t / m
      const Real dt_remain = tf - tra.t;
      tra.vv_f = tra.vv_i - (tra.spower * dt_remain) / MALPHA;
      // recalculate dx_push
      dx_i = 0.5 * (tra.vv_i + tra.vv_f) * dt_remain;
  }
  Real dt_cutoff = dx_i / tra.vv_i;

  return std::min(dt_cutoff,dt_push);
}
*/

KOKKOS_FORCEINLINE_FUNCTION
void ptcl_transport_step(tran_step_args tra) {
  // Do not recalculate dx_end or cell boundary limiters here!
  Real dx_push = tra.dx_push;

  //Real dt_push = dx_push / tra.vv_i;
  Real dt_push;
  if (tra.spower > 0.0) {
      Real v_f_sq = tra.vv_i * tra.vv_i - 2.0 * tra.spower * dx_push / MALPHA;
      tra.vv_f = (v_f_sq > 0.0) ? std::sqrt(v_f_sq) : 0.0;

      dt_push = dx_push / (.5 * (tra.vv_i + tra.vv_f));
      // dt = (vi - vf) / |a|
      //dt_push = MALPHA * (tra.vv_i - tra.vv_f) / tra.spower;
  } else {
      dt_push = dx_push / tra.vv_i;
      tra.vv_f = tra.vv_i;
  }

  // push time and position
  tra.t += dt_push;
  Real inv_v = 1.0 / tra.vv_i;
  tra.x += tra.vx * inv_v * dx_push;
  tra.y += tra.multi_d * tra.vy * inv_v * dx_push;
  tra.z += tra.three_d * tra.vz * inv_v * dx_push;

  // push time slightly into next time step if particle is at end of time step (census)
  const bool is_atend =
      fuzzy_equal(tra.t, tra.t_start + tra.dt, tra.dt, eps_imc_offset());
  tra.t = is_atend ? tra.t_start + (1.0 + eps_imc_offset()) * tra.dt : tra.t;

  // handle faces (Keep this identical to your original code)
  const bool leave = !(tra.in_background);
  const Real fdx = eps_imc_offset() * (tra.xu - tra.xl);
  const Real fdy = eps_imc_offset() * (tra.yu - tra.yl);
  const Real fdz = eps_imc_offset() * (tra.zu - tra.zl);
  tra.x = (std::abs(tra.x - tra.xl) < fdx && leave) ? tra.xl - fdx : tra.x;
  tra.x = (std::abs(tra.x - tra.xu) < fdx && leave) ? tra.xu + fdx : tra.x;
  tra.y = (tra.multi_d && std::abs(tra.y - tra.yl) < fdy && leave) ? tra.yl - fdy : tra.y;
  tra.y = (tra.multi_d && std::abs(tra.y - tra.yu) < fdy && leave) ? tra.yu + fdy : tra.y;
  tra.z = (tra.three_d && std::abs(tra.z - tra.zl) < fdz && leave) ? tra.zl - fdz : tra.z;
  tra.z = (tra.three_d && std::abs(tra.z - tra.zu) < fdz && leave) ? tra.zu + fdz : tra.z;
}
} // namespace charon

#endif // CHARON_TRANSPORT_UTILS_HPP_
