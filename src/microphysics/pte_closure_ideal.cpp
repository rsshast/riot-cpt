//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

// Singularity includes
#include <ports-of-call/portability.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/eos/eos.hpp>

// Riot includes
#include <array>
#include <cmath>

#include "eos_riot.hpp"
#include "ionization/ionization.hpp"
#include "materials/materials.hpp"
#include "pte_closure.hpp"
#include "riot_utils/riot_loops.hpp"
#include "variables.hpp"

using namespace parthenon::package::prelude;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn  Closure::ApplyIdealGasClosure
//! \brief
void Closure::ApplyIdealGasClosure(MeshData<Real> *md, IndexDomain domain) {
  using parthenon::ParArray1D;
  using singularity::IdealGas;
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  auto pm = md->GetParentPointer();

  auto riot = pm->packages.Get("riot");
  const bool do_ionization = riot->Param<bool>("do_ionization");

  // Materials params
  auto &materials = pm->packages.Get("materials");
  const auto &matlist = materials->Param<std::vector<int>>("h.pte_matlist");
  const auto &eos = materials->Param<ParArray1D<RiotEOS::EOS>>("d.d.EOS");
  const auto &eos_from_matid = materials->Param<ParArray1D<int>>("d.EOS_from_matid");

  // Hydro params
  auto &hydro = pm->packages.Get("hydro");
  const Real &mfrac_thr = hydro->Param<Real>("mass_frac_thresh");
  const Real &vfrac_thr = hydro->Param<Real>("vol_frac_thresh");

  // Packing and indexing
  auto v =
      riot::MakePack<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                     cm::ionization_zbar, ccbulk::internal_energy, ccbulk::rho,
                     ccbulk::pressure, ccbulk::temperature, cm::lr_cache, cm::lT_cache>(
          md, matlist);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return;

  // Reference temperature and density for calling the EOSs
  const Real rho0 = 1.0;
  const Real T0 = 300.0;

  // Execute Ideal PTE Solve
  using lt = RiotUtils::LoopType<>;
  auto idx_space =
      lt::GetIndexSpace(domain, 0, nblocks, md, parthenon::TopologicalElement::CC);
  idx_space.template AddPerPointScratch<Real>(2);
  RiotLoop::outer(
      idx_space, KOKKOS_LAMBDA(const lt::idx_range_t &idx_range, const int b) {
        const int nmat = v.GetSize(b, ccmat::rho());

        auto rho_cv_sum = RiotLoop::GetPerPointScratch<Real>(idx_range);
        auto rho_gm1_cv_sum = RiotLoop::GetPerPointScratch<Real>(idx_range);
        rho_cv_sum.Zero();
        rho_gm1_cv_sum.Zero();
        idx_range.TeamBarrier();

        // Calculate required sums
        for (int m = 0; m < nmat; ++m) {
          const int mat_id = v(b, ccmat::rho(m)).sparse_id;
          const int phase_id = v(b, ccmat::rho(m)).v;
          const auto &eos_mat = eos(eos_from_matid(mat_id) + phase_id);

          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          auto eos_loop = [&](const auto &eos) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              RiotEOS::LambdaIndexerSingle pl(pv_m, kji);
              const Real cv = eos.SpecificHeatFromDensityTemperature(rho0, T0, pl);
              const Real bmod0 = eos.BulkModulusFromDensityTemperature(rho0, T0, pl);
              const Real p0 = eos.PressureFromDensityTemperature(rho0, T0, pl);
              const Real gamma = bmod0 / p0;
              const Real gm1_cv = (gamma - 1.0) * cv;
              rho_cv_sum(kji) += cv * pv_m(ccmat::rho(), kji);
              rho_gm1_cv_sum(kji) += gm1_cv * pv_m(ccmat::rho(), kji);
            });
          };
          eos_mat.EvaluateDevice(eos_loop);
          idx_range.TeamBarrier();
        }

        // Fill bulk quantities
        auto pv = RiotLoop::make_pack_view(idx_range, v);
        RiotLoop::inner(idx_range, [&](const auto kji) {
          pv(ccbulk::temperature(), kji) =
              pv(ccbulk::internal_energy(), kji) / (rho_cv_sum(kji) + 1.e-50);
          pv(ccbulk::pressure(), kji) =
              pv(ccbulk::temperature(), kji) * rho_gm1_cv_sum(kji);
          rho_gm1_cv_sum(kji) = 1.0 / (rho_gm1_cv_sum(kji) + 1.e-50);
        });
        idx_range.TeamBarrier();

        // Use sums and bulk to fill material arrays
        for (int m = 0; m < nmat; ++m) {
          const int mat_id = v(b, ccmat::rho(m)).sparse_id;
          const int phase_id = v(b, ccmat::rho(m)).v;
          const auto &eos_mat = eos(eos_from_matid(mat_id) + phase_id);

          auto pv_m = RiotLoop::make_sparse_pack_view(idx_range, v, m);
          auto eos_loop = [&](const auto &eos) {
            RiotLoop::inner(idx_range, [&](const auto kji) {
              RiotEOS::LambdaIndexerSingle pl(pv_m, kji);
              const Real cv = eos.SpecificHeatFromDensityTemperature(rho0, T0, pl);
              const Real bmod = eos.BulkModulusFromDensityTemperature(rho0, T0, pl);
              const Real p0 = eos.PressureFromDensityTemperature(rho0, T0, pl);
              const Real gamma = bmod / p0;
              const Real gm1_cv = (gamma - 1.0) * cv;

              // Compute PTE volume fractions
              pv_m(ccmat::volume_fraction(), kji) =
                  pv_m(ccmat::rho(), kji) * gm1_cv * rho_gm1_cv_sum(kji);

              // Apply masks
              const Real mthr_mask =
                  (pv_m(ccmat::rho(), kji) > mfrac_thr * pv(ccbulk::rho(), kji));
              const Real vthr_mask = (pv_m(ccmat::volume_fraction(), kji) > vfrac_thr);
              const Real mask = mthr_mask * vthr_mask;
              pv_m(ccmat::rho(), kji) *= mask;
              pv_m(ccmat::volume_fraction(), kji) *= mask;
            });
          };
          eos_mat.EvaluateDevice(eos_loop);
          idx_range.TeamBarrier();
        }
      });
}

// 1T specialization: both supported EOSs obey P = 2 u / 3, even when
// the IdealElectrons heat capacity depends on the local mean ionization.
// Thus conserved thermal energy fixes total pressure. Nested bracketed solves
// recover T and material densities from total-pressure equilibrium and sum(vf)=1.
void Closure::ApplyOneTemperatureClosure(MeshData<Real> *md, IndexDomain domain) {
  using parthenon::ParArray1D;
  namespace cb = cell_variables::cell_averaged::bulk;
  namespace mat = cell_variables::cell_averaged::mat;
  namespace cm = cell_variables::material_averaged;
  auto pm = md->GetParentPointer();
  auto materials = pm->packages.Get("materials");
  auto ion = pm->packages.Get("ionization");
  const auto eos = materials->Param<EOS_Array_t>("d.d.EOS");
  const auto electrons = materials->Param<EOS_Array_t>("d.d.electron_EOS");
  const auto map = materials->Param<ParArray1D<int>>("d.EOS_from_matid");
  const auto nphase = materials->Param<ParArray1D<int>>("d.nphase");
  const bool full = ion->Param<bool>("fully_ionized");
  const bool hydrogen = ion->Param<bool>("fully_ionize_hydrogen");
  auto v = riot::MakePack<mat::rho, mat::volume_fraction, mat::ionization_zbar,
                         cm::ionization_zbar, cb::internal_energy, cb::pressure,
                         cb::temperature, cb::electron_temperature>(md);
  const auto ib = md->GetBoundsI(domain), jb = md->GetBoundsJ(domain),
             kb = md->GetBoundsK(domain);
  if (v.GetNBlocks() == 0) return;
  parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "Close1T", DevExecSpace(),
      0, v.GetNBlocks()-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const int nm = v.GetSize(b, mat::rho());
        PARTHENON_REQUIRE(nm <= RiotLimits::MAX_MATERIALS,
                          "1T material workspace exceeded");
        Real ci[RiotLimits::MAX_MATERIALS], ce[RiotLimits::MAX_MATERIALS];
        Real aa[RiotLimits::MAX_MATERIALS], zz[RiotLimits::MAX_MATERIALS];
        Real cmin = 0, cmax = 0;
        for (int m = 0; m < nm; ++m) {
          const int id = v(b, mat::rho(m)).sparse_id;
          PARTHENON_REQUIRE(nphase(id) == 1, "1T currently requires single-phase materials");
          const auto &ei = eos(map(id));
          const auto &ee = electrons(map(id));
          aa[m] = ei.MeanAtomicMass();
          zz[m] = ei.MeanAtomicNumber();
          PARTHENON_REQUIRE(aa[m] == ee.MeanAtomicMass() && zz[m] == ee.MeanAtomicNumber(),
                            "1T ion and electron EOS atomic properties must match");
          RiotEOS::LambdaIndexerSimple lambda(1);
          ci[m] = ei.SpecificHeatFromDensityTemperature(1, 300, lambda);
          ce[m] = ee.SpecificHeatFromDensityTemperature(1, 300, lambda);
          const Real rho = v(b, mat::rho(m), k, j, i);
          cmin += rho * ci[m];
          cmax += rho * (ci[m] + zz[m] * ce[m]);
        }
        if (cmin == 0) {
          v(b, cb::temperature(), k, j, i) = 1;
          v(b, cb::electron_temperature(), k, j, i) = 1;
          v(b, cb::pressure(), k, j, i) = 0;
          return;
        }
        const Real u = v(b, cb::internal_energy(), k, j, i);
        PARTHENON_REQUIRE(std::isfinite(u) && u > 0,
                          "1T requires positive finite total thermal energy");
        const Real pressure = (2.0/3.0) * u;
        auto material_density = [&](int m, Real t, Real &z) {
          const Real base = pressure / ((2.0/3.0) * t);
          Real lo = base / (ci[m] + zz[m] * ce[m]);
          Real hi = base / ci[m];
          const bool fully = full || (hydrogen && zz[m] == 1);
          if (fully) { z = zz[m]; return lo; }
          for (int it = 0; it < 64; ++it) {
            const Real rho = 0.5 * (lo + hi);
            Ionization::ComputeIonizationState(aa[m], zz[m], rho, t, z, false);
            if (rho * (ci[m] + ce[m] * z) > base) hi = rho;
            else lo = rho;
            if (hi - lo < 1.e-12 * hi) break;
          }
          const Real rho = 0.5 * (lo + hi);
          Ionization::ComputeIonizationState(aa[m], zz[m], rho, t, z, false);
          return rho;
        };
        Real lo = u / cmax, hi = u / cmin;
        for (int it = 0; it < 64; ++it) {
          const Real t = 0.5 * (lo + hi);
          Real vf = 0;
          for (int m = 0; m < nm; ++m) {
            if (v(b, mat::rho(m), k, j, i) <= 0) continue;
            Real z;
            vf += v(b, mat::rho(m), k, j, i) / material_density(m, t, z);
          }
          if (vf > 1) hi = t; else lo = t;
          if (hi - lo < 1.e-12 * hi) break;
        }
        const Real t = 0.5 * (lo + hi);
        Real vf_sum = 0;
        for (int m = 0; m < nm; ++m) {
          const Real rho = v(b, mat::rho(m), k, j, i);
          Real z = 0, vf = 0;
          if (rho > 0) vf = rho / material_density(m, t, z);
          v(b, mat::volume_fraction(m), k, j, i) = vf;
          v(b, cm::ionization_zbar(m), k, j, i) = z;
          v(b, mat::ionization_zbar(m), k, j, i) = rho * z;
          vf_sum += vf;
        }
        PARTHENON_REQUIRE(std::isfinite(t) && t > 0 && std::abs(vf_sum - 1) < 1.e-9,
                          "1T total-pressure/temperature closure failed");
        v(b, cb::temperature(), k, j, i) = t;
        v(b, cb::electron_temperature(), k, j, i) = t;
        v(b, cb::pressure(), k, j, i) = pressure;
      });
}
