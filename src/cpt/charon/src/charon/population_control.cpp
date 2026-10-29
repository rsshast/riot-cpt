//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <iostream>

// Parthenon includes
#include <parthenon_mpi.hpp>

// Charon includes
#include "charon.hpp"
#include "charon_utils.hpp"
#include "transport_utils.hpp"

namespace charon {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ControlPopulation
//! \brief Ensure particle population is near target user number of particles.
//! This currently uses a very simple strategy of killing a particle based on a
//! probability calculated from the number of active in excess of the number of source
//! particles. Removal is enabled only when N_src > 0 and N_act > 4*N_src,
//! with removal probability 1 - 4*N_src/N_act. Survivor weights are renormalized
//! by the ratio of the pre-removal and post-removal per-cell weight sums.
template <typename WeightTag, typename IjkTag>
TaskStatus ControlPopulationImpl(MeshData<Real> *md, const std::string& swarm_name) {
  namespace fj = field::charon;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &cpt_pkg = pm->packages.Get("charon");
  auto &rng_pool = cpt_pkg->template Param<RngPool>("rng_pool");

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<WeightTag, swarm_position::x, swarm_position::y,
                                          swarm_position::z>(swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<IjkTag>(swarm_name);

  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  const int nx1 = ib.e - ib.s + 1;
  const int nx2 = jb.e - jb.s + 1;
  const int cells = nx1 * nx2 * (kb.e - kb.s + 1);
  auto scratch = cpt_pkg->template Param<std::shared_ptr<TransportScratch>>(
      "transport_scratch");
  PARTHENON_REQUIRE(scratch->source_blocks == nblocks && scratch->source_cells == cells,
                    "Population control requires this step's source counts.");
  scratch->ResizePopulation(nblocks, cells);
  auto source_count = scratch->source_count;
  auto active_count = scratch->active_count;
  auto active_weight = scratch->active_weight;
  auto survivor_weight = scratch->survivor_weight;

  // Census migration can leave ijk referring to the previous block. Refresh it
  // independently of output diagnostics before indexing interior-only workspace.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::refresh-indices", DevExecSpace(),
      0, nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          swarm_d.Xtoijk(ppack_r(b, swarm_position::x(), n),
                        ppack_r(b, swarm_position::y(), n),
                        ppack_r(b, swarm_position::z(), n),
                        ppack_i(b, IjkTag(0), n), ppack_i(b, IjkTag(1), n),
                        ppack_i(b, IjkTag(2), n));
        }
      });

  //--------------------------------------------------------------------------------------
  // reset active particle count and energy weight per cell to 0
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::zero-active-counts-1",
      parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const int cell = (k - kb.s) * nx1 * nx2 + (j - jb.s) * nx1 + i - ib.s;
        active_weight(b, cell) = 0.0;
        active_count(b, cell) = 0.0;
      });

  //--------------------------------------------------------------------------------------
  // aggregate active particle count and energy per cell
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::active-ptcl-counts-1", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, IjkTag(0), n);
          const int &jp = ppack_i(b, IjkTag(1), n);
          const int &kp = ppack_i(b, IjkTag(2), n);
          const int cell = (kp - kb.s) * nx1 * nx2 + (jp - jb.s) * nx1 + ip - ib.s;
          const Real &ww = ppack_r(b, WeightTag(), n);

          // add ew and 1 to active particle number at cell
          Real &actew = active_weight(b, cell);
          Real &actnum = active_count(b, cell);
          Kokkos::atomic_add(&actew, ww);
          Kokkos::atomic_add(&actnum, 1.0);
        }
      });

  //--------------------------------------------------------------------------------------
  // sample probability to mark particle for removal
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::active-ptcl-remove", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();

          // logical location and weight of particle
          const int &ip = ppack_i(b, IjkTag(0), n);
          const int &jp = ppack_i(b, IjkTag(1), n);
          const int &kp = ppack_i(b, IjkTag(2), n);
          const int cell = (kp - kb.s) * nx1 * nx2 + (jp - jb.s) * nx1 + ip - ib.s;

          // sample if particle is to be removed
          const Real &actnum = active_count(b, cell);
          const Real srcnum = source_count(b, cell);
          // TODO: fix this arbitrary hard-coded threshold
          if (srcnum > 0.0 && actnum > 4 * srcnum) {
            const Real rand = rng_gen.drand();
            if (rand * actnum < actnum - 4 * srcnum) {
              swarm_d.MarkParticleForRemoval(n);
            }
          }
          rng_pool.free_state(rng_gen);
        }
      });

  long long active_before_local = 0;
  for (int b = 0; b <= nblocks - 1; ++b) {
    active_before_local += md->GetSwarmData(b)->Get(swarm_name)->GetNumActive();
  }

  //--------------------------------------------------------------------------------------
  // remove marked particles
  for (int b = 0; b <= nblocks - 1; ++b) {
    md->GetSwarmData(b)->Get(swarm_name)->RemoveMarkedParticles();
  }

  long long active_after_local = 0;
  for (int b = 0; b <= nblocks - 1; ++b) {
    active_after_local += md->GetSwarmData(b)->Get(swarm_name)->GetNumActive();
  }
  long long active_before = active_before_local;
  long long active_after = active_after_local;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &active_before, 1,
                                    MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &active_after, 1,
                                    MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
#endif
  const long long removed = active_before - active_after;
  if (removed > 0 && parthenon::Globals::my_rank == 0) {
    std::cout << "[Charon Population Control]"
              << " | method: roulette removal where active > 4*source per cell,"
              << " followed by survivor weight renormalization"
              << " | active before: " << active_before
              << " | removed: " << removed
              << " | active after: " << active_after
              << std::endl;
  }

  //--------------------------------------------------------------------------------------
  // reset active particle count per cell to 0
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::zero-active-counts-2",
      parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const int cell = (k - kb.s) * nx1 * nx2 + (j - jb.s) * nx1 + i - ib.s;
        active_count(b, cell) = 0.0;
        survivor_weight(b, cell) = 0.0;
      });

  //--------------------------------------------------------------------------------------
  // store old active energy weight totals per cell, after particle removal
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::store-old-active-ew", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, IjkTag(0), n);
          const int &jp = ppack_i(b, IjkTag(1), n);
          const int &kp = ppack_i(b, IjkTag(2), n);
          const int cell = (kp - kb.s) * nx1 * nx2 + (jp - jb.s) * nx1 + ip - ib.s;
          const Real &ww = ppack_r(b, WeightTag(), n);

          // add ew and 1 to active particle number at cell
          Real &actew = survivor_weight(b, cell);
          Real &actnum = active_count(b, cell);
          Kokkos::atomic_add(&actew, ww);
          Kokkos::atomic_add(&actnum, 1.0);
        }
      });

  //--------------------------------------------------------------------------------------
  // renormalize surviving weights to retain the pre-removal per-cell weight sum
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ControlPopulation::renorm-ptcl-ew", DevExecSpace(), 0,
      nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // logical location and weight of particle
          const int &ip = ppack_i(b, IjkTag(0), n);
          const int &jp = ppack_i(b, IjkTag(1), n);
          const int &kp = ppack_i(b, IjkTag(2), n);
          const int cell = (kp - kb.s) * nx1 * nx2 + (jp - jb.s) * nx1 + ip - ib.s;
          Real &ww = ppack_r(b, WeightTag(), n);

          const Real &oldactew = survivor_weight(b, cell);
          const Real &actew = active_weight(b, cell);
          PARTHENON_DEBUG_REQUIRE(oldactew > 0.0, "Particle in cell with 0 particles!");
          ww *= (actew / oldactew);
        }
      });

  if (cpt_pkg->template Param<bool>("cpt_diagnostics")) {
    auto desc = MakePackDescriptor<fj::active_ew_per_cell, fj::active_num_per_cell,
                                   fj::old_active_ew_per_cell>(resolved_pkgs.get());
    auto vdiag = desc.GetPack(md);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ControlPopulation::copy-diagnostics", DevExecSpace(),
        0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          const int cell = (k - kb.s) * nx1 * nx2 + (j - jb.s) * nx1 + i - ib.s;
          vdiag(b, fj::active_ew_per_cell(), k, j, i) = active_weight(b, cell);
          vdiag(b, fj::active_num_per_cell(), k, j, i) = active_count(b, cell);
          vdiag(b, fj::old_active_ew_per_cell(), k, j, i) = survivor_weight(b, cell);
        });
  }

  return TaskStatus::complete;
}

TaskStatus ControlPopulation(MeshData<Real>* md) {
  auto pm = md->GetParentPointer();
  auto &cpt_pkg = pm->packages.Get("charon");
  const bool do_population_control =
      cpt_pkg->template Param<bool>("do_population_control");
  if (!do_population_control) return TaskStatus::complete;
  //const auto &pt = cpt_pkg->template Param<ParticleType>("particle_type");
  return ControlPopulationImpl<particle::alphas::weight,  particle::alphas::ijk >(md, alphas_swarm_name);
}

} // namespace charon
