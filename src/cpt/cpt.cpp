//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.

#include "cpt/cpt.hpp"

#include <cmath>

#include <parthenon_mpi.hpp>

#include "charon.hpp"
#include "transport_utils.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "variables.hpp"

using namespace parthenon::driver::prelude;

namespace CPT {

namespace {

bool HasCPTWork(Mesh *pm, const Real dt) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  int local_has_work = 0;
  const auto &pkg = pm->packages.Get("charon");
  const bool qfield = pkg->Param<charon::SourceStrategy>("source_strategy") ==
                      charon::SourceStrategy::qfield;
  const bool source_allowed = pkg->Param<Real>("cumulative_alphas_sourced") <
                              pkg->Param<Real>("alpha_source_physical_cap");
  const Real source_scale =
      pkg->Param<std::string>("alpha_source_mode") == "rate" ? dt : 1.0;
  const int num_partitions = pm->DefaultNumPartitions();
  for (int p = 0; p < num_partitions; ++p) {
    auto &base = pm->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    auto v = riot::MakePack<ccbulk::cpt_alpha_source>(md);
    const int nblocks = v.GetNBlocks();

    if (nblocks > 0) {
      const auto &ib = md->GetBoundsI(IndexDomain::interior);
      const auto &jb = md->GetBoundsJ(IndexDomain::interior);
      const auto &kb = md->GetBoundsK(IndexDomain::interior);
      int source_cells = 0;
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "CPT::CheckSource", DevExecSpace(), 0,
          nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                        int &count) {
            // The derived-field update owns source generation and the threshold
            // decision. This guard only checks whether that source can spawn.
            const Real q = v(b, ccbulk::cpt_alpha_source(), k, j, i);
            if (source_allowed && q * source_scale >= 1.0) ++count;
          },
          Kokkos::Sum<int>(source_cells));
      if (source_cells > 0) local_has_work = 1;
    }

    // A source-free step must still transport census particles left from an earlier
    // step. The swarm count is already available on the host.
    for (int b = 0; b < md->NumBlocks(); ++b) {
      if (md->GetSwarmData(b)->Get(alphas_swarm_name)->GetNumActive() > 0) {
        local_has_work = 1;
        break;
      }
    }
  }

  // Other Charon strategies have their own pulse/geometry eligibility logic.
  // Do not incorrectly suppress them based on RIOT's qfield threshold.
  if (!qfield && source_allowed) local_has_work = 1;

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &local_has_work, 1, MPI_INT,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  return local_has_work != 0;
}

void ClearInactiveStepFields(Mesh *pm) {
  namespace fj = field::charon;

  const int num_partitions = pm->DefaultNumPartitions();
  for (int p = 0; p < num_partitions; ++p) {
    auto &base = pm->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    const int nblocks = md->NumBlocks();
    if (nblocks == 0) continue;

    const auto &ib = md->GetBoundsI(IndexDomain::interior);
    const auto &jb = md->GetBoundsJ(IndexDomain::interior);
    const auto &kb = md->GetBoundsK(IndexDomain::interior);
    // Dispatch the clear through Charon so only allocated tally fields are touched.
    charon::UpdateDerivedTransportFields(md, 0.0);
    if (pm->packages.Get("charon")->Param<bool>("cpt_diagnostics")) {
      auto vdiag = riot::MakePack<fj::source_num_per_cell, fj::source_ew_per_cell,
                                 fj::delta_num_per_cell, fj::active_num_per_cell,
                                 fj::active_ew_per_cell>(md);
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "CPT::ClearInactiveDiagnostics",
          DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            vdiag(b, fj::source_num_per_cell(), k, j, i) = 0.0;
            vdiag(b, fj::source_ew_per_cell(), k, j, i) = 0.0;
            vdiag(b, fj::delta_num_per_cell(), k, j, i) = 0.0;
            vdiag(b, fj::active_num_per_cell(), k, j, i) = 0.0;
            vdiag(b, fj::active_ew_per_cell(), k, j, i) = 0.0;
          });
    }
  }
  Kokkos::fence();
}

template <bool Compact>
TaskStatus ApplyDeposition(MeshData<Real> *md) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace fj = field::charon;

  auto pm = md->GetMeshPointer();
  auto &charon_pkg = pm->packages.Get("charon");
  if (!charon_pkg->Param<bool>("do_feedback")) return TaskStatus::complete;

  // RIOT feeds back energy only; material densities and alpha mass tallies are
  // not needed to apply deposition or refresh the thermodynamic state.
  auto v = [&]() {
    if constexpr (Compact) {
      return riot::MakePack<ccbulk::total_material_energy,
                            fj::deposited_energy_step>(md);
    } else {
      return riot::MakePack<ccbulk::total_material_energy, fj::energy_tally_step,
                            fj::energy_delta_step>(md);
    }
  }();
  const int nblocks = v.GetNBlocks();
  if (nblocks == 0) return TaskStatus::complete;

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);

  int changed_cells = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "CPT::ApplyDeposition", DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    int &changed) {
        auto &coords = v.GetCoordinates(b);
        const Real deposited_energy = [&]() {
          if constexpr (Compact) {
            return v(b, fj::deposited_energy_step(), k, j, i);
          } else {
            return v(b, fj::energy_tally_step(), k, j, i) +
                   v(b, fj::energy_delta_step(), k, j, i);
          }
        }();
        if (deposited_energy != 0.0) {
          v(b, ccbulk::total_material_energy(), k, j, i) +=
              deposited_energy / coords.CellVolume(k, j, i);
          ++changed;
        }
      }, Kokkos::Sum<int>(changed_cells));

  if (changed_cells > 0) {
    *charon_pkg->MutableParam<bool>("feedback_changed") = true;
  }

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
    auto interior =
        tl.AddTask(none, PreCommFillDerived<MeshData<Real>>, md);
    auto boundaries =
        parthenon::AddBoundaryExchangeTasks(interior, tl, base, pm->multilevel);
    tl.AddTask(boundaries, FillDerived<MeshData<Real>>, md);
  }
  return tc;
}

} // namespace

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;

  // Keep the historical cutoff for existing decks that do not supply this knob.
  // The bangtime generator explicitly supplies its selected cutoff in kelvin.
  const Real source_temperature_threshold =
      pin->GetOrAddReal("charon", "source_temperature_threshold", 1.16045e7);
  PARTHENON_REQUIRE(std::isfinite(source_temperature_threshold) &&
                        source_temperature_threshold > 0.0,
                    "charon/source_temperature_threshold must be finite and positive [K].")

  // RIOT currently feeds back energy only. Other Charon hosts retain the full
  // tally default, and RIOT users can explicitly request mass/momentum tallies.
  bool output_mass_momentum = false;
  for (int n = 0; n < 100; ++n) {
    const std::string block = "parthenon/output" + std::to_string(n);
    if (!pin->DoesParameterExist(block, "variables")) continue;
    for (const auto &name : pin->GetVector<std::string>(block, "variables")) {
      if (name == "field.charon.mass_delta" ||
          name == "field.charon.mass_delta_step" ||
          name == "field.charon.momentum_delta" ||
          name == "field.charon.momentum_delta_step") output_mass_momentum = true;
    }
  }
  if (!pin->DoesParameterExist("charon", "tally_mass_momentum")) {
    pin->GetOrAddBoolean("charon", "tally_mass_momentum", output_mass_momentum);
  }
  auto pkg = charon::Initialize(pin, "charon", true);
  pkg->AddParam<Real>("source_temperature_threshold", source_temperature_threshold);
  pkg->AddParam<bool>("inactive_fields_dirty", true,
                      parthenon::Params::Mutability::Mutable);
  pkg->AddParam<bool>("feedback_changed", false,
                      parthenon::Params::Mutability::Mutable);
  Metadata source_metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField<ccbulk::cpt_alpha_source>(source_metadata);
  return pkg;
}

TaskListStatus Step(Mesh *pm, SimTime &tm, const Real dt) {
  auto &pkg = pm->packages.Get("charon");
  if (!HasCPTWork(pm, dt)) {
    auto scratch = pkg->Param<std::shared_ptr<charon::TransportScratch>>(
        "transport_scratch");
    bool same_blocks = scratch->inactive_blocks.size() == pm->block_list.size();
    for (std::size_t b = 0; same_blocks && b < pm->block_list.size(); ++b) {
      same_blocks = scratch->inactive_blocks[b].lock() == pm->block_list[b];
    }
    if (!same_blocks) *pkg->MutableParam<bool>("inactive_fields_dirty") = true;
    if (pkg->Param<bool>("inactive_fields_dirty")) {
      ClearInactiveStepFields(pm);
      scratch->inactive_blocks.clear();
      for (const auto &block : pm->block_list) scratch->inactive_blocks.push_back(block);
      *pkg->MutableParam<bool>("inactive_fields_dirty") = false;
    }
    return TaskListStatus::complete;
  }

  *pkg->MutableParam<bool>("inactive_fields_dirty") = true;
  auto status = charon::RadiationStep(pm, tm.time, dt).Execute();
  if (status != TaskListStatus::complete) return status;

  if (!pkg->Param<bool>("do_feedback")) return status;
  *pkg->MutableParam<bool>("feedback_changed") = false;
  for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
    auto *md = pm->mesh_data.GetOrAdd("base", p).get();
    if (pkg->Param<bool>("compact_tallies")) {
      ApplyDeposition<true>(md);
    } else {
      ApplyDeposition<false>(md);
    }
  }
  int changed = pkg->Param<bool>("feedback_changed") ? 1 : 0;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  // Every rank must participate in the boundary exchange if any rank deposited
  // energy. Keep the full thermodynamic refresh before subsequent physics.
  if (changed == 0) return status;
  return DepositionTasks(pm).Execute();
}

} // namespace CPT
