//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.

#include "cpt/cpt.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

#include <parthenon_mpi.hpp>

#include "charon.hpp"
#include "transport_utils.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "variables.hpp"
#include "microphysics/eos_riot.hpp"
#include "riot_utils/riot_utils.hpp"
#include "tnburn/ratelib/tn_data_types.hpp"

using namespace parthenon::driver::prelude;

namespace CPT {

namespace {

// TN background, source-only production, and explicit plasma feedback.
namespace fj = field::charon;
namespace cb = cell_variables::cell_averaged::bulk;
namespace cm = cell_variables::material_averaged;
namespace mat = cell_variables::cell_averaged::mat;

void InitializeTNPlasma(ParameterInput *pin, StateDescriptor *pkg,
                        StateDescriptor *materials) {
  if (!pkg->Param<bool>("tn_plasma")) return;
  PARTHENON_REQUIRE(pkg->Param<std::string>("stopping_power_model") == "LiPetrasso" ||
                        pkg->Param<std::string>("stopping_power_model") == "constant",
                    "tn_plasma supports LiPetrasso or constant stopping");
  PARTHENON_REQUIRE(pin->GetBoolean("physics", "tn") &&
                        pin->GetBoolean("physics", "ionization"),
                    "tn_plasma requires TN and ionization");
  PARTHENON_REQUIRE(pin->GetOrAddBoolean("tnburn", "source_only", false),
                    "tn_plasma currently requires tnburn/source_only=true");
  PARTHENON_REQUIRE(
      !pin->GetOrAddBoolean("ionization", "advect_electron_entropy", false),
      "tn_plasma currently requires electron-energy advection, not entropy advection");
  PARTHENON_REQUIRE(
      !pkg->Param<bool>("do_population_control") &&
          !pkg->Param<bool>("tally_mass_momentum") &&
          pkg->Param<charon::SourceStrategy>("source_strategy") ==
              charon::SourceStrategy::qfield &&
          pkg->Param<std::string>("alpha_source_mode") == "total",
      "tn_plasma requires qfield/total, no population or mass/momentum feedback");
  PARTHENON_REQUIRE(pkg->Param<Real>("alpha_source_physical_cap") ==
                        std::numeric_limits<Real>::max(),
                    "TN source accounting requires no alpha_source_physical_cap");
  const auto names = materials->Param<std::vector<std::vector<int>>>("Isotope Zaids");
  const auto masses = materials->Param<std::vector<std::vector<Real>>>("Isotope Masses");
  const auto charges = materials->Param<std::vector<std::vector<int>>>("Isotope Charges");
  const auto eos = materials->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  const auto eos_map = materials->Param<std::vector<int>>("h.EOS_from_matid");
  std::vector<int> starts, counts;
  std::vector<Real> species_masses, species_charges;
  for (int m = 0; m < names.size(); ++m) {
    starts.push_back(species_masses.size());
    counts.push_back(names[m].size());
    if (names[m].empty()) {
      species_masses.push_back(eos[eos_map[m]].MeanAtomicMass() * amu);
      species_charges.push_back(eos[eos_map[m]].MeanAtomicNumber());
    } else {
      for (int s = 0; s < names[m].size(); ++s) {
        PARTHENON_REQUIRE(masses[m][s] > 0.0 && charges[m][s] >= 0,
                          "Invalid isotope metadata");
        species_masses.push_back(masses[m][s] * amu);
        species_charges.push_back(charges[m][s]);
      }
    }
  }
  pkg->AddParam("plasma_species_start",
                RiotUtils::VectorToDevice(starts, "plasma species start"));
  pkg->AddParam("plasma_material_isotopes",
                RiotUtils::VectorToDevice(counts, "plasma isotope count"));
  pkg->AddParam("plasma_species_mass",
                RiotUtils::VectorToDevice(species_masses, "plasma species masses"));
  pkg->AddParam("plasma_species_nuclear_charge",
                RiotUtils::VectorToDevice(species_charges, "plasma nuclear charges"));
  pkg->AddParam("plasma_num_species", static_cast<int>(species_masses.size()));
  const std::string upper =
      pin->GetOrAddString("charon", "tn_upper_temperature", "error");
  PARTHENON_REQUIRE(upper == "error" || upper == "clamp",
                    "tn_upper_temperature must be error or explicit clamp");
  pkg->AddParam("tn_clamp_upper", upper == "clamp");
  Metadata work({Metadata::Cell, Metadata::OneCopy});
  Metadata persistent(
      {Metadata::Cell, Metadata::OneCopy, Metadata::Restart, Metadata::ForceRemeshComm});
  Metadata vector({Metadata::Cell, Metadata::OneCopy}, std::vector<int>{3});
  Metadata ions({Metadata::Cell, Metadata::OneCopy},
                std::vector<int>{static_cast<int>(species_masses.size())});
  pkg->AddField(fj::plasma_velocity::name(), vector);
  pkg->AddField(fj::plasma_ion_density::name(), ions);
  pkg->AddField(fj::plasma_ion_charge::name(), ions);
  for (auto name :
       {fj::plasma_electron_density::name(), fj::plasma_electron_temperature::name(),
        fj::plasma_ion_temperature::name(), fj::plasma_screening::name(),
        fj::electron_heat_step::name(), fj::ion_heat_step::name()})
    pkg->AddField(name, work);
  // Extensive physical inventory per cell, persisted independently of output cadence.
  for (auto name :
       {fj::pending_alphas::name(), fj::physical_alphas::name(),
        fj::sampled_alphas::name(), fj::electron_heat::name(), fj::ion_heat::name(),
        fj::escaped_alphas::name(), fj::escaped_energy::name()})
    pkg->AddField(name, persistent);
}

void PrepareTNPlasma(MeshData<Real> *md, Real dt) {
  auto pm = md->GetMeshPointer();
  const auto pkg = pm->packages.Get("charon");
  const auto tn = pm->packages.Get("TNBurn");
  const auto materials = pm->packages.Get("materials");
  const auto starts = pkg->Param<ParArray1D<int>>("plasma_species_start");
  const auto niso = pkg->Param<ParArray1D<int>>("plasma_material_isotopes");
  const auto masses = pkg->Param<ParArray1D<Real>>("plasma_species_mass");
  const auto charges = pkg->Param<ParArray1D<Real>>("plasma_species_nuclear_charge");
  const int ns = pkg->Param<int>("plasma_num_species");
  const Real weight = pkg->Param<Real>("alpha_weight");
  PARTHENON_REQUIRE(std::isfinite(weight) && weight > 0, "Invalid physical alpha weight");
  const bool clamp = pkg->Param<bool>("tn_clamp_upper");
  const auto nreactions = tn->Param<ParArray1D<int>>("num_reactions_per_mat");
  const auto reactions = tn->Param<ParArray2D<int>>("reaction_list_per_mat");
  const auto reactants = tn->Param<ParArray3D<int>>("reactant_indices_per_mat_reaction");
  const auto data = tn->Param<ParArray1D<ReactionData>>("reaction_data");
  const auto nphase = materials->Param<ParArray1D<int>>("d.nphase");
  const Real vfmin = pm->packages.Get("hydro")->Param<Real>("vol_frac_thresh");
  auto v = riot::MakePack<
      cb::temperature, cb::electron_temperature, cb::electron_number_density,
      cb::velocity, cb::cpt_alpha_source, mat::rho, mat::iso, mat::volume_fraction,
      cm::phase_fraction, cm::ionization_zbar, fj::plasma_velocity,
      fj::plasma_electron_density, fj::plasma_electron_temperature,
      fj::plasma_ion_temperature, fj::plasma_screening, fj::plasma_ion_density,
      fj::plasma_ion_charge, fj::pending_alphas, fj::physical_alphas, fj::sampled_alphas,
      fj::electron_heat_step, fj::ion_heat_step>(md);
  if (v.GetNBlocks() == 0) return;
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "PrepareTNPlasma", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        Real rho = 0;
        for (int m = 0; m < v.GetSize(b, mat::rho()); ++m)
          rho += v(b, mat::rho(m), k, j, i);
        // Positive vacuum sentinels belong to this adapter, not RIOT's state.
        const Real ti = rho > 0 ? v(b, cb::temperature(), k, j, i) : Real(1);
        const Real te = rho > 0 ? v(b, cb::electron_temperature(), k, j, i) : Real(1);
        const Real ne = v(b, cb::electron_number_density(), k, j, i);
        PARTHENON_REQUIRE(std::isfinite(ti) && ti > 0 && std::isfinite(te) && te > 0 &&
                              std::isfinite(ne) && ne >= 0,
                          "Invalid RIOT plasma state");
        v(b, fj::plasma_ion_temperature(), k, j, i) = ti;
        v(b, fj::plasma_electron_temperature(), k, j, i) = te;
        v(b, fj::plasma_electron_density(), k, j, i) = ne;
        for (int d = 0; d < 3; ++d)
          v(b, fj::plasma_velocity(d), k, j, i) =
              rho > 0 ? v(b, cb::velocity(d), k, j, i) : Real(0);
        for (int s = 0; s < ns; ++s) {
          v(b, fj::plasma_ion_density(s), k, j, i) = 0;
          v(b, fj::plasma_ion_charge(s), k, j, i) = 0;
        }
        Real rate = 0;
        int iso_offset = 0;
        for (int m = 0; m < v.GetSize(b, mat::rho()); ++m) {
          const int id = v(b, mat::rho(m)).sparse_id;
          // This adapter supports arbitrary sparse material IDs; multiphase chemistry
          // requires an explicit stopping closure rather than duplicate isotope density.
          PARTHENON_REQUIRE(nphase(id) == 1,
                            "tn_plasma currently requires single-phase materials");
          const int first = starts(id);
          const Real zbar = v(b, cm::ionization_zbar(m), k, j, i);
          if (niso(id) == 0) {
            v(b, fj::plasma_ion_density(first), k, j, i) =
                v(b, mat::rho(m), k, j, i) / masses(first);
            v(b, fj::plasma_ion_charge(first), k, j, i) = zbar;
          } else {
            for (int s = 0; s < niso(id); ++s) {
              v(b, fj::plasma_ion_density(first + s), k, j, i) =
                  v(b, mat::iso(iso_offset + s), k, j, i) / masses(first + s);
              // Explicit isotope populations in this mode are fully ionized fuel.
              v(b, fj::plasma_ion_charge(first + s), k, j, i) = charges(first + s);
            }
            const Real vf = v(b, mat::volume_fraction(m), k, j, i);
            for (int r = 0; r < nreactions(id); ++r) {
              const auto rd = data(reactions(id, r));
              PARTHENON_REQUIRE(
                  rd.num_reactants == 2 &&
                      ((rd.reactants[0] == 1002 && rd.reactants[1] == 1003) ||
                       (rd.reactants[0] == 1003 && rd.reactants[1] == 1002)),
                  "Source-only CPT supports DT only");
              const Real lo =
                  std::max(rd.Temperatures(0), Real(1.602176565e-9 / 1.3806488e-16));
              const Real hi = rd.Temperatures(rd.num_temp - 1);
              if (vf > vfmin && ti >= lo) {
                PARTHENON_REQUIRE(clamp || ti <= hi,
                                  "DT source above table maximum; extend table or "
                                  "explicitly select clamp");
                const Real sigma = rd.SigmaVBar.interpToReal(std::log(std::min(ti, hi)));
                const Real phase = v(b, cm::phase_fraction(m), k, j, i);
                rate += v(b, mat::iso(iso_offset + reactants(id, r, 0)), k, j, i) *
                        v(b, mat::iso(iso_offset + reactants(id, r, 1)), k, j, i) *
                        sigma * phase * phase / vf;
              }
            }
          }
          iso_offset += niso(id);
        }
        Real screening = charon::PlasmaStopping::Screening(ne, 1, te);
        for (int s = 0; s < ns; ++s) {
          const Real n = v(b, fj::plasma_ion_density(s), k, j, i);
          const Real z = v(b, fj::plasma_ion_charge(s), k, j, i);
          PARTHENON_REQUIRE(std::isfinite(n) && n >= 0 && std::isfinite(z) && z >= 0,
                            "Invalid ion density/charge in plasma adapter");
          screening += charon::PlasmaStopping::Screening(n, z, ti);
        }
        v(b, fj::plasma_screening(), k, j, i) = screening;
        const Real produced = rate * v.GetCoordinates(b).CellVolume(k, j, i) * dt;
        PARTHENON_REQUIRE(std::isfinite(produced) && produced >= 0,
                          "Invalid physical alpha production");
        auto &pending = v(b, fj::pending_alphas(), k, j, i);
        const Real count = charon::SourceHistoryCount(produced, weight, pending);
        const int ncells = (ib.e - ib.s + 1) * (jb.e - jb.s + 1) * (kb.e - kb.s + 1);
        PARTHENON_REQUIRE(count < INT_MAX / ncells,
                          "Too many histories per block: increase alpha_weight");
        v(b, cb::cpt_alpha_source(), k, j, i) = count;
        v(b, fj::physical_alphas(), k, j, i) += produced;
        v(b, fj::sampled_alphas(), k, j, i) += count * weight;
        v(b, fj::electron_heat_step(), k, j, i) = 0;
        v(b, fj::ion_heat_step(), k, j, i) = 0;
      });
}

TaskStatus ApplyPlasmaDeposition(MeshData<Real> *md) {
  const auto pkg = md->GetMeshPointer()->packages.Get("charon");
  auto v = riot::MakePack<cb::total_material_energy, cb::electron_internal_energy,
                          fj::electron_heat_step, fj::ion_heat_step>(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  int changed = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ApplyPlasmaHeat", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &n) {
        const Real qe = v(b, fj::electron_heat_step(), k, j, i);
        const Real qi = v(b, fj::ion_heat_step(), k, j, i);
        const Real volume = v.GetCoordinates(b).CellVolume(k, j, i);
        v(b, cb::total_material_energy(), k, j, i) += (qe + qi) / volume;
        v(b, cb::electron_internal_energy(), k, j, i) += qe / volume;
        if (qe + qi != 0) ++n;
      },
      Kokkos::Sum<int>(changed));
  if (changed) *pkg->MutableParam<bool>("feedback_changed") = true;
  return TaskStatus::complete;
}

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

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, StateDescriptor *materials) {
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
  InitializeTNPlasma(pin, pkg.get(), materials);
  return pkg;
}

TaskListStatus Step(Mesh *pm, SimTime &tm, const Real dt) {
  auto &pkg = pm->packages.Get("charon");
  if (pkg->Param<bool>("tn_plasma")) {
    for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
      PrepareTNPlasma(pm->mesh_data.GetOrAdd("base", p).get(), dt);
    }
  }
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
    if (pkg->Param<bool>("tn_plasma")) {
      ApplyPlasmaDeposition(md);
    } else if (pkg->Param<bool>("compact_tallies")) {
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
