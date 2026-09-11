//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.

#include "cpt/cpt.hpp"

#include "charon.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "variables.hpp"

using namespace parthenon::driver::prelude;

namespace CPT {

namespace {

TaskStatus ApplyDeposition(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;
  namespace fj = field::charon;

  auto pm = md->GetMeshPointer();
  auto &charon_pkg = pm->packages.Get("charon");
  if (!charon_pkg->Param<bool>("do_feedback")) return TaskStatus::complete;

  auto v = riot::MakePack<ccbulk::total_material_energy, ccmat::rho,
                          fj::energy_tally_step,
                          fj::energy_delta_step, fj::mass_delta_step>(md);
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  for (int b = 0; b < nblocks; ++b) {
    PARTHENON_REQUIRE(v.GetSize(b, ccmat::rho()) > 0,
                      "CPT mass deposition requires at least one material per block.")
  }

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "CPT::ApplyDeposition", DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &coords = v.GetCoordinates(b);
        const Real inv_volume = 1.0 / coords.CellVolume(k, j, i);
        const Real deposited_energy =
            v(b, fj::energy_tally_step(), k, j, i) +
            v(b, fj::energy_delta_step(), k, j, i);
        v(b, ccbulk::total_material_energy(), k, j, i) +=
            deposited_energy * inv_volume;

        const Real deposited_mass = v(b, fj::mass_delta_step(), k, j, i);
        if (deposited_mass > 0.0) {
          const int nmat = v.GetSize(b, ccmat::rho());
          int largest_material = 0;
          Real largest_density = v(b, ccmat::rho(0), k, j, i);
          for (int m = 1; m < nmat; ++m) {
            const Real material_density = v(b, ccmat::rho(m), k, j, i);
            if (material_density > largest_density) {
              largest_density = material_density;
              largest_material = m;
            }
          }
          v(b, ccmat::rho(largest_material), k, j, i) +=
              deposited_mass * inv_volume;
        }
      });

  return TaskStatus::complete;
}

TaskCollection DepositionTasks(Mesh *pm) {
  using namespace ::parthenon::Update;

  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pm->DefaultNumPartitions();
  auto &region = tc.AddRegion(num_partitions);
  for (int p = 0; p < num_partitions; ++p) {
    auto &tl = region[p];
    auto &base = pm->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    auto deposition = tl.AddTask(none, ApplyDeposition, md);
    auto interior =
        tl.AddTask(deposition, PreCommFillDerived<MeshData<Real>>, md);
    auto boundaries =
        parthenon::AddBoundaryExchangeTasks(interior, tl, base, pm->multilevel);
    tl.AddTask(boundaries, FillDerived<MeshData<Real>>, md);
  }
  return tc;
}

} // namespace

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  auto pkg = charon::Initialize(pin, "charon");
  Metadata source_metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField<ccbulk::cpt_alpha_source>(source_metadata);
  return pkg;
}

TaskListStatus Step(Mesh *pm, SimTime &tm, const Real dt) {
  auto status = charon::RadiationStep(pm, tm.time, dt).Execute();
  if (status != TaskListStatus::complete) return status;

  if (!pm->packages.Get("charon")->Param<bool>("do_feedback")) return status;
  return DepositionTasks(pm).Execute();
}

} // namespace CPT
