//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.

#include "riot_pgen/pgen.hpp"

#include <singularity-eos/eos/eos.hpp>

namespace cpt_0d {

using parthenon::ParArray1D;
using namespace RiotEOS;

//----------------------------------------------------------------------------------------
//! \fn void cpt_0d::ProblemGenerator
//! \brief Initialize a uniform, stationary, single-material box for CPT tests.
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  using parthenon::MakePackDescriptor;

  auto &rc = pmb->meshblock_data.Get();
  for (auto &var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  PARTHENON_REQUIRE(pmb->packages.AllPackages().count("charon") == 1,
                    "The CPT 0D problem requires physics/cpt=true.")

  static auto desc =
      MakePackDescriptor<ccmat::rho, ccmat::volume_fraction,
                         ccmat::internal_energy,
                         ccbulk::total_material_energy, ccbulk::momentum,
                         ccbulk::temperature,
                         ccbulk::cpt_alpha_source>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(rc.get());

  PARTHENON_REQUIRE(v.GetSize(0, ccmat::rho()) == 1,
                    "The CPT 0D problem requires exactly one material.")

  const Real density = pin->GetOrAddReal("problem", "density", 1.0);
  const Real temperature =
      pin->GetOrAddReal("problem", "temperature", 1.0e7);
  PARTHENON_REQUIRE(density > 0.0, "problem/density must be positive.")
  PARTHENON_REQUIRE(temperature > 0.0,
                    "problem/temperature must be positive.")

  auto eos = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");

  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "ProblemGenerator::cpt_0d", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real sie =
            eos(0).InternalEnergyFromDensityTemperature(density, temperature);
        const Real internal_energy_density = density * sie;

        v(0, ccmat::rho(0), k, j, i) = density;
        v(0, ccmat::volume_fraction(0), k, j, i) = 1.0;
        v(0, ccmat::internal_energy(0), k, j, i) =
            internal_energy_density;
        v(0, ccbulk::total_material_energy(), k, j, i) =
            internal_energy_density;
        v(0, ccbulk::temperature(), k, j, i) = temperature;
        v(0, ccbulk::cpt_alpha_source(), k, j, i) = 0.0;
        for (int d = 0; d < 3; ++d) {
          v(0, ccbulk::momentum(d), k, j, i) = 0.0;
        }
      });
}

} // namespace cpt_0d
