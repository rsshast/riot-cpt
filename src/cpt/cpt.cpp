//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.

#include "cpt/cpt.hpp"
#include "cpt/spherical.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

#include <parthenon_mpi.hpp>

#include "charon.hpp"
#include "transport_utils.hpp"
#include "multiphysics/fill_shared_derived.hpp"
#include "variables.hpp"
#include "microphysics/eos_riot.hpp"
#include "materials/materials.hpp"
#include "riot_utils/riot_utils.hpp"
#include "tnburn/ratelib/tn_data_types.hpp"

using namespace parthenon::driver::prelude;

namespace CPT {

namespace {

// Exact isothermal two-distinct-reactant update, dn1/dt = dn2/dt = -k*n1*n2.
// Number densities and k include the material volume-fraction correction.
// Keep the device implementation here with its caller; the public host wrapper
// below lets the unit tests exercise the same calculation.
KOKKOS_INLINE_FUNCTION
Real AcceptedDTReactionsImpl(Real n1, Real n2, Real k, Real dt) {
  if (n1 <= 0 || n2 <= 0 || k <= 0 || dt <= 0) return 0;
  const Real lo = std::min(n1, n2), hi = std::max(n1, n2);
  const Real difference = hi - lo;
  const Real exposure = k * dt;
  if (difference == 0) {
    const Real burn = lo * exposure;
    return lo * (burn / (1 + burn));
  }
  // expm1 retains small burns and nearly equimolar fuel accurately.
  const Real removed = -std::expm1(-difference * exposure);
  return std::min(lo, std::max(Real(0),
      lo * (hi * removed / (difference + lo * removed))));
}

// TN burn/source accounting, stopping background, and conservative plasma feedback.
namespace fj = field::charon;
namespace cb = cell_variables::cell_averaged::bulk;
namespace cm = cell_variables::material_averaged;
namespace mat = cell_variables::cell_averaged::mat;

void FillHeliumMassFraction(MeshData<Real> *md) {
  auto v = riot::MakePack<mat::rho, mat::cpt_helium_rho,
                          cm::cpt_helium_mass_fraction>(md);
  if (v.GetNBlocks() == 0) return;
  auto ib = md->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "FillCPTHelium", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        for (int m = 0; m < v.GetSize(b, mat::rho()); ++m) {
          const Real rho = v(b, mat::rho(m), k, j, i);
          v(b, cm::cpt_helium_mass_fraction(m), k, j, i) =
              rho > 0 ? v(b, mat::cpt_helium_rho(m), k, j, i) / rho : Real(0);
        }
      });
}

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
                    "tn_plasma owns the burn update; set tnburn/source_only=true");
  const bool depletion = pin->GetOrAddBoolean("charon", "tn_depletion", true);
  PARTHENON_REQUIRE(!depletion || pkg->Param<bool>("do_feedback"),
                    "tn_depletion requires do_feedback=true; disable depletion for "
                    "a prescribed-background transport study");
  pkg->AddParam("tn_depletion", depletion);
  PARTHENON_REQUIRE(
      !pin->GetOrAddBoolean("ionization", "advect_electron_entropy", false),
      "tn_plasma currently requires electron-energy advection, not entropy advection");
  PARTHENON_REQUIRE(
      !pkg->Param<bool>("do_population_control") &&
          !pkg->Param<bool>("tally_mass_momentum") &&
          pkg->Param<charon::SourceStrategy>("source_strategy") ==
              charon::SourceStrategy::qfield &&
          pkg->Param<std::string>("alpha_source_mode") == "total",
      "tn_plasma requires qfield/total, no population control or legacy "
      "tally_mass_momentum; plasma uses its own conservative deposition fields");
  PARTHENON_REQUIRE(pkg->Param<Real>("alpha_source_physical_cap") ==
                        std::numeric_limits<Real>::max(),
                    "TN source accounting requires no alpha_source_physical_cap");
  const auto names = materials->Param<std::vector<std::vector<int>>>("Isotope Zaids");
  const auto masses = materials->Param<std::vector<std::vector<Real>>>("Isotope Masses");
  const auto charges = materials->Param<std::vector<std::vector<int>>>("Isotope Charges");
  const auto eos = materials->Param<std::vector<RiotEOS::EOS>>("h.h.EOS");
  const auto eos_map = materials->Param<std::vector<int>>("h.EOS_from_matid");
  std::vector<int> starts, counts, helium;
  std::vector<Real> species_masses, species_charges;
  for (int m = 0; m < names.size(); ++m) {
    starts.push_back(species_masses.size());
    counts.push_back(names[m].size());
    int helium_index = -1;
    for (int s = 0; s < names[m].size(); ++s)
      if (names[m][s] == 2004) helium_index = s;
    helium.push_back(helium_index);
    PARTHENON_REQUIRE(names[m].empty() || helium_index >= 0,
                      "TN plasma materials with isotopes must include helium-4 (2004)");
    if (names[m].empty()) {
      species_masses.push_back(eos[eos_map[m]].MeanAtomicMass() * amu);
      species_charges.push_back(eos[eos_map[m]].MeanAtomicNumber());
      // A distinct helium channel lets the pusher retain its surrogate species.
      species_masses.push_back(charon::units::MALPHA);
      species_charges.push_back(2);
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
  pkg->AddParam("plasma_helium_index",
                RiotUtils::VectorToDevice(helium, "plasma helium index"));
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
        fj::electron_heat_step::name(), fj::ion_heat_step::name(),
        fj::deposited_mass_step::name(), fj::deposited_lab_energy_step::name(),
        fj::source_alpha_energy::name()})
    pkg->AddField(name, work);
  pkg->AddField(fj::deposited_momentum_step::name(), vector);
  // Extensive physical inventory per cell, persisted independently of output cadence.
  for (auto name :
       {fj::pending_alphas::name(), fj::physical_alphas::name(),
        fj::sampled_alphas::name(), fj::electron_heat::name(), fj::ion_heat::name(),
        fj::escaped_alphas::name(), fj::escaped_energy::name(),
        fj::thermalized_alphas::name(), fj::neutron_mass::name(),
        fj::neutron_energy::name(), fj::nuclear_mass_defect::name(),
        fj::pending_alpha_energy::name()})
    pkg->AddField(name, persistent);
  Metadata ash({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Sparse, Metadata::FillGhost,
                Metadata::Advected, Metadata::WithFluxes});
  ash.Associate(cm::cpt_helium_mass_fraction::name());
  auto ash_pool = SparsePool::Make<mat::cpt_helium_rho>(ash, mat::rho::name());
  Metadata fraction({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                     Metadata::Sparse});
  auto fraction_pool = SparsePool::Make<cm::cpt_helium_mass_fraction>(
      fraction, mat::rho::name());
  const auto matids = materials->Param<std::vector<int>>("matids");
  for (int id : matids) Materials::AddMatId(id, {1}, ash_pool, fraction_pool);
  Materials::AddPools(pkg, ash_pool, fraction_pool);
  pkg->FillDerivedMesh = FillHeliumMassFraction;
  const auto before_loop = pkg->UserWorkBeforeLoopMesh;
  pkg->UserWorkBeforeLoopMesh = [before_loop](Mesh *pm, ParameterInput *input,
                                             SimTime &tm) {
    before_loop(pm, input, tm);
    if (tm.ncycle != 0) return; // retain restart-backed ash
    for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
      auto md = pm->mesh_data.GetOrAdd("base", p);
      auto v = riot::MakePack<mat::cpt_helium_rho,
                              cm::cpt_helium_mass_fraction>(md.get());
      if (v.GetNBlocks() == 0) continue;
      auto ib = md->GetBoundsI(IndexDomain::entire);
      auto jb = md->GetBoundsJ(IndexDomain::entire);
      auto kb = md->GetBoundsK(IndexDomain::entire);
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "InitializeCPTHelium", DevExecSpace(),
          0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(int b, int k, int j, int i) {
            for (int m = 0; m < v.GetSize(b, mat::cpt_helium_rho()); ++m) {
              v(b, mat::cpt_helium_rho(m), k, j, i) = 0;
              v(b, cm::cpt_helium_mass_fraction(m), k, j, i) = 0;
            }
          });
    }
  };
}

// One operator owns physical reaction counts, depletion and MC source inventory.
// The second (burn=false) pass refreshes the stopping background after the EOS.
void PrepareTNPlasma(MeshData<Real> *md, Real dt, bool burn = true) {
  auto pm = md->GetMeshPointer();
  const auto pkg = pm->packages.Get("charon");
  const auto tn = pm->packages.Get("TNBurn");
  const auto materials = pm->packages.Get("materials");
  const auto starts = pkg->Param<ParArray1D<int>>("plasma_species_start");
  const auto niso = pkg->Param<ParArray1D<int>>("plasma_material_isotopes");
  const auto masses = pkg->Param<ParArray1D<Real>>("plasma_species_mass");
  const auto charges = pkg->Param<ParArray1D<Real>>("plasma_species_nuclear_charge");
  const int ns = pkg->Param<int>("plasma_num_species");
  const bool depletion = pkg->Param<bool>("tn_depletion");
  const Real weight = pkg->Param<Real>("alpha_weight");
  PARTHENON_REQUIRE(std::isfinite(weight) && weight > 0, "Invalid physical alpha weight");
  const bool clamp = pkg->Param<bool>("tn_clamp_upper");
  const auto nreactions = tn->Param<ParArray1D<int>>("num_reactions_per_mat");
  const auto reactions = tn->Param<ParArray2D<int>>("reaction_list_per_mat");
  const auto reactants = tn->Param<ParArray3D<int>>("reactant_indices_per_mat_reaction");
  const auto data = tn->Param<ParArray1D<ReactionData>>("reaction_data");
  const auto nphase = materials->Param<ParArray1D<int>>("d.nphase");
  const int num_reactions = tn->Param<int>("num_reactions");
  const auto zaids = tn->Param<std::vector<int>>("all_isotopes");
  const auto isotope_masses = tn->Param<std::vector<Real>>("isotope_masses");
  Real neutron_mass = 0;
  for (int s = 0; s < zaids.size(); ++s)
    if (zaids[s] == 1) neutron_mass = isotope_masses[s] * amu;
  PARTHENON_REQUIRE(neutron_mass > 0, "DT burn requires neutron mass metadata");
  const bool one_temperature = pm->packages.Get("riot")->Param<bool>("one_temperature");
  const Real vfmin = pm->packages.Get("hydro")->Param<Real>("vol_frac_thresh");
  auto v = riot::MakePack<
      cb::temperature, cb::electron_temperature, cb::electron_number_density,
      cb::velocity, cb::cpt_alpha_source, cb::momentum, cb::total_material_energy,
      cb::internal_energy, cb::electron_internal_energy, mat::rho, mat::iso,
      mat::cpt_helium_rho, mat::tn_reaction_density, mat::volume_fraction,
      mat::ionization_zbar,
      cm::phase_fraction, cm::ionization_zbar, fj::plasma_velocity,
      fj::plasma_electron_density, fj::plasma_electron_temperature,
      fj::plasma_ion_temperature, fj::plasma_screening, fj::plasma_ion_density,
      fj::plasma_ion_charge, fj::pending_alphas, fj::physical_alphas, fj::sampled_alphas,
      fj::electron_heat_step, fj::ion_heat_step, fj::deposited_mass_step,
      fj::deposited_momentum_step, fj::deposited_lab_energy_step,
      fj::thermalized_alphas, fj::neutron_mass, fj::neutron_energy,
      fj::nuclear_mass_defect, fj::pending_alpha_energy, fj::source_alpha_energy>(md);
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
        const Real te = one_temperature ? ti :
            (rho > 0 ? v(b, cb::electron_temperature(), k, j, i) : Real(1));
        Real ne = 0;
        PARTHENON_REQUIRE(std::isfinite(ti) && ti > 0 && std::isfinite(te) && te > 0 &&
                              std::isfinite(ne) && ne >= 0,
                          "Invalid RIOT plasma state");
        v(b, fj::plasma_ion_temperature(), k, j, i) = ti;
        v(b, fj::plasma_electron_temperature(), k, j, i) = te;
        v(b, fj::plasma_electron_density(), k, j, i) = ne;
        for (int d = 0; d < 3; ++d)
          v(b, fj::plasma_velocity(d), k, j, i) =
              rho > 0 ? v(b, cb::velocity(d), k, j, i) : Real(0);
        for (int d = 0; d < 3; ++d)
          v(b, fj::deposited_momentum_step(d), k, j, i) = 0;
        for (int s = 0; s < ns; ++s) {
          v(b, fj::plasma_ion_density(s), k, j, i) = 0;
          v(b, fj::plasma_ion_charge(s), k, j, i) = 0;
        }
        Real produced = 0, alpha_energy = 0, removed_mass = 0, consumed_energy = 0;
        const Real volume = v.GetCoordinates(b).CellVolume(k, j, i);
        Real ion_energy = std::max(Real(0), v(b, cb::internal_energy(), k, j, i) -
                                             (one_temperature ? Real(0) : v(b, cb::electron_internal_energy(), k, j, i)));
        int iso_offset = 0, reaction_offset = 0;
        for (int m = 0; m < v.GetSize(b, mat::rho()); ++m) {
          const int id = v(b, mat::rho(m)).sparse_id;
          // This adapter supports arbitrary sparse material IDs; multiphase chemistry
          // requires an explicit stopping closure rather than duplicate isotope density.
          PARTHENON_REQUIRE(nphase(id) == 1,
                            "tn_plasma currently requires single-phase materials");
          const int first = starts(id);
          const Real zbar = v(b, cm::ionization_zbar(m), k, j, i);
          if (niso(id) == 0) {
            const Real ash = std::max(Real(0), v(b, mat::cpt_helium_rho(m), k, j, i));
            v(b, fj::plasma_ion_density(first), k, j, i) =
                std::max(Real(0), v(b, mat::rho(m), k, j, i) - ash) / masses(first);
            v(b, fj::plasma_ion_charge(first), k, j, i) = zbar;
            v(b, fj::plasma_ion_density(first + 1), k, j, i) = ash / masses(first + 1);
            v(b, fj::plasma_ion_charge(first + 1), k, j, i) = 2;
          } else {
            for (int s = 0; s < niso(id); ++s) {
              v(b, fj::plasma_ion_density(first + s), k, j, i) =
                  v(b, mat::iso(iso_offset + s), k, j, i) / masses(first + s);
              // Explicit isotope populations in this mode are fully ionized fuel.
              v(b, fj::plasma_ion_charge(first + s), k, j, i) = charges(first + s);
            }
            const Real vf = v(b, mat::volume_fraction(m), k, j, i);
            PARTHENON_REQUIRE(nreactions(id) <= 1,
                              "TN plasma accepts one DT reaction per material");
            for (int r = 0; burn && r < nreactions(id); ++r) {
              const auto rd = data(reactions(id, r));
              PARTHENON_REQUIRE(
                  rd.num_reactants == 2 &&
                      ((rd.reactants[0] == 1002 && rd.reactants[1] == 1003) ||
                       (rd.reactants[0] == 1003 && rd.reactants[1] == 1002)),
                  "TN plasma burn supports DT only");
              const Real lo =
                  std::max(rd.Temperatures(0), Real(1.602176565e-9 / 1.3806488e-16));
              const Real hi = rd.Temperatures(rd.num_temp - 1);
              if (vf > vfmin && ti >= lo) {
                PARTHENON_REQUIRE(clamp || ti <= hi,
                                  "DT source above table maximum; extend table or "
                                  "explicitly select clamp");
                const Real sigma = rd.SigmaVBar.interpToReal(std::log(std::min(ti, hi)));
                const Real phase = v(b, cm::phase_fraction(m), k, j, i);
                const int s1 = reactants(id, r, 0), s2 = reactants(id, r, 1);
                const Real m1 = masses(first + s1), m2 = masses(first + s2);
                auto &rho1 = v(b, mat::iso(iso_offset + s1), k, j, i);
                auto &rho2 = v(b, mat::iso(iso_offset + s2), k, j, i);
                PARTHENON_REQUIRE(std::isfinite(rho1) && rho1 >= 0 &&
                                  std::isfinite(rho2) && rho2 >= 0 &&
                                  std::isfinite(sigma) && sigma >= 0,
                                  "Invalid DT inventory or reactivity");
                const Real coefficient = sigma * m1 * m2 * phase * phase / vf;
                Real accepted = depletion
                    ? AcceptedDTReactionsImpl(rho1 / m1, rho2 / m2, coefficient, dt)
                    : rho1 * rho2 * sigma * phase * phase / vf * dt;
                Real ea = 0, en = 0;
                for (int p = 0; p < rd.num_products; ++p) {
                  PARTHENON_REQUIRE(rd.product_multiplicities[p] == 1 &&
                      (rd.products[p] == 1 || rd.products[p] == 2004),
                      "DT plasma requires one neutron and one alpha");
                  const Real e = rd.EnergyOut[p].interpToReal(std::log(std::min(ti, hi)));
                  if (rd.products[p] == 2004) ea = e;
                  else en = e;
                }
                PARTHENON_REQUIRE(ea > 0 && en > 0, "Invalid DT product energies");
                // Tables include relative and pair-CM thermal energy. Remove
                // their excess over Q once, from the ion (2T) or total (1T) reservoir.
                const Real input = std::max(Real(0), ea + en - rd.Qval);
                if (depletion && input > 0)
                  accepted = std::min(accepted, Real(0.99) * ion_energy / input);
                PARTHENON_REQUIRE(std::isfinite(accepted) && accepted >= 0,
                                  "Invalid accepted DT reaction count");
                produced += accepted * volume;
                alpha_energy += accepted * volume * ea;
                v(b, mat::tn_reaction_density(reaction_offset + reactions(id, r)),
                  k, j, i) += accepted;
                if (depletion) {
                  const Real loss1 = std::min(rho1, accepted * m1);
                  const Real loss2 = std::min(rho2, accepted * m2);
                  rho1 -= loss1;
                  rho2 -= loss2;
                  const Real loss = loss1 + loss2;
                  auto &material_rho = v(b, mat::rho(m), k, j, i);
                  PARTHENON_REQUIRE(loss <= material_rho * (1 + 1.e-12),
                                    "DT isotope masses exceed material density");
                  const Real remaining = std::max(Real(0), material_rho - loss);
                  v(b, mat::ionization_zbar(m), k, j, i) *=
                      material_rho > 0 ? remaining / material_rho : Real(0);
                  material_rho = remaining;
                  removed_mass += loss;
                  consumed_energy += accepted * input;
                  ion_energy -= accepted * input;
                  v(b, fj::neutron_mass(), k, j, i) += accepted * volume * neutron_mass;
                  v(b, fj::neutron_energy(), k, j, i) += accepted * volume * en;
                  v(b, fj::nuclear_mass_defect(), k, j, i) += accepted * volume *
                      (m1 + m2 - charon::units::MALPHA - neutron_mass);
                }
              }
            }
          }
          iso_offset += niso(id);
          if (nreactions(id) > 0) reaction_offset += num_reactions;
          const int channels = niso(id) == 0 ? 2 : niso(id);
          for (int s = 0; s < channels; ++s)
            ne += v(b, fj::plasma_ion_density(first + s), k, j, i) *
                  v(b, fj::plasma_ion_charge(first + s), k, j, i);
        }
        v(b, fj::plasma_electron_density(), k, j, i) = ne;
        Real screening = charon::PlasmaStopping::Screening(ne, 1, te);
        for (int s = 0; s < ns; ++s) {
          const Real n = v(b, fj::plasma_ion_density(s), k, j, i);
          const Real z = v(b, fj::plasma_ion_charge(s), k, j, i);
          PARTHENON_REQUIRE(std::isfinite(n) && n >= 0 && std::isfinite(z) && z >= 0,
                            "Invalid ion density/charge in plasma adapter");
          screening += charon::PlasmaStopping::Screening(n, z, ti);
        }
        v(b, fj::plasma_screening(), k, j, i) = screening;
        if (burn && depletion) {
          Real speed2 = 0;
          for (int d = 0; d < 3; ++d) {
            const Real u = v(b, cb::velocity(d), k, j, i);
            speed2 += u * u;
            v(b, cb::momentum(d), k, j, i) -= removed_mass * u;
          }
          v(b, cb::total_material_energy(), k, j, i) -=
              consumed_energy + 0.5 * removed_mass * speed2;
        }
        PARTHENON_REQUIRE(std::isfinite(produced) && produced >= 0,
                          "Invalid physical alpha production");
        if (burn) {
          auto &pending = v(b, fj::pending_alphas(), k, j, i);
          auto &pending_energy = v(b, fj::pending_alpha_energy(), k, j, i);
          const Real available = pending + produced;
          PARTHENON_REQUIRE(pending == 0 || pending_energy > 0,
                            "Pending alpha inventory has no energy; start a fresh "
                            "run instead of a pre-burnup checkpoint");
          const Real birth_energy = available > 0
              ? (pending_energy + alpha_energy) / available : charon::units::E0;
          PARTHENON_REQUIRE(std::isfinite(birth_energy) && birth_energy > 0,
                            "Invalid alpha birth energy");
          const Real count = charon::SourceHistoryCount(produced, weight, pending);
          pending_energy = pending * birth_energy;
          const int ncells = (ib.e - ib.s + 1) * (jb.e - jb.s + 1) * (kb.e - kb.s + 1);
          PARTHENON_REQUIRE(count < INT_MAX / ncells,
                            "Too many histories per block: increase alpha_weight");
          v(b, cb::cpt_alpha_source(), k, j, i) = count;
          v(b, fj::source_alpha_energy(), k, j, i) = birth_energy;
          v(b, fj::physical_alphas(), k, j, i) += produced;
          v(b, fj::sampled_alphas(), k, j, i) += count * weight;
        }
        v(b, fj::electron_heat_step(), k, j, i) = 0;
        v(b, fj::ion_heat_step(), k, j, i) = 0;
        v(b, fj::deposited_mass_step(), k, j, i) = 0;
        v(b, fj::deposited_lab_energy_step(), k, j, i) = 0;
      });
}

TaskStatus ApplyPlasmaDeposition(MeshData<Real> *md) {
  const auto pkg = md->GetMeshPointer()->packages.Get("charon");
  const bool one_temperature = md->GetMeshPointer()->packages.Get("riot")->Param<bool>("one_temperature");
  const auto niso = pkg->Param<ParArray1D<int>>("plasma_material_isotopes");
  const auto helium = pkg->Param<ParArray1D<int>>("plasma_helium_index");
  auto v = riot::MakePack<cb::total_material_energy, cb::electron_internal_energy,
      cb::momentum, mat::rho, mat::iso, mat::cpt_helium_rho, mat::ionization_zbar,
      fj::electron_heat_step, fj::ion_heat_step, fj::deposited_mass_step,
      fj::deposited_momentum_step, fj::deposited_lab_energy_step>(md);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  int changed = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ApplyPlasmaDeposition", DevExecSpace(), 0,
      v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &n) {
        const Real qe = v(b, fj::electron_heat_step(), k, j, i);
        const Real volume = v.GetCoordinates(b).CellVolume(k, j, i);
        const Real mass = v(b, fj::deposited_mass_step(), k, j, i);
        if (mass > 0) {
          int winner = -1, winner_id = INT_MAX, offset = 0, winner_offset = 0;
          Real largest = 0;
          for (int m = 0; m < v.GetSize(b, mat::rho()); ++m) {
            const int id = v(b, mat::rho(m)).sparse_id;
            const Real rho = v(b, mat::rho(m), k, j, i);
            // Cell mass fractions have a common denominator. Select from the
            // pre-deposition state; exact ties go to the smallest material ID.
            if (rho > largest || (rho == largest && rho > 0 && id < winner_id)) {
              largest = rho;
              winner = m;
              winner_id = id;
              winner_offset = offset;
            }
            offset += niso(id);
          }
          PARTHENON_REQUIRE(winner >= 0, "Cannot deposit alpha mass in a vacuum cell");
          // Preserve the host EOS charge guess; its ionization solve refreshes
          // this surrogate. CPT's explicit helium stopping channel uses Z=2.
          v(b, mat::ionization_zbar(winner), k, j, i) *= 1 + mass / (volume * largest);
          v(b, mat::rho(winner), k, j, i) += mass / volume;
          v(b, mat::cpt_helium_rho(winner), k, j, i) += mass / volume;
          if (niso(winner_id) > 0)
            v(b, mat::iso(winner_offset + helium(winner_id)), k, j, i) += mass / volume;
        }
        const Real energy = v(b, fj::deposited_lab_energy_step(), k, j, i);
        v(b, cb::total_material_energy(), k, j, i) += energy / volume;
        if (!one_temperature)
          v(b, cb::electron_internal_energy(), k, j, i) += qe / volume;
        bool momentum_changed = false;
        for (int d = 0; d < 3; ++d) {
          const Real dp = v(b, fj::deposited_momentum_step(d), k, j, i);
          v(b, cb::momentum(d), k, j, i) += dp / volume;
          momentum_changed = momentum_changed || dp != 0;
        }
        if (mass != 0 || energy != 0 || qe != 0 || momentum_changed) ++n;
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

Real AcceptedDTReactions(Real n1, Real n2, Real k, Real dt) {
  return AcceptedDTReactionsImpl(n1, n2, k, dt);
}

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
  if (pkg->Param<bool>("spherical_transport")) spherical::Initialize(pin, pkg.get());
  return pkg;
}

TaskListStatus Step(Mesh *pm, SimTime &tm, const Real dt) {
  auto &pkg = pm->packages.Get("charon");
  if (pkg->Param<bool>("tn_plasma")) {
    for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
      PrepareTNPlasma(pm->mesh_data.GetOrAdd("base", p).get(), dt);
    }
    // Burn changes the fluid even when the source remainder is too small to
    // create a history. Refresh on every rank before the no-work early return.
    if (pkg->Param<bool>("tn_depletion")) {
      auto status = DepositionTasks(pm).Execute();
      if (status != TaskListStatus::complete) return status;
      for (int p = 0; p < pm->DefaultNumPartitions(); ++p)
        PrepareTNPlasma(pm->mesh_data.GetOrAdd("base", p).get(), 0, false);
    }
  }
  if (pkg->Param<bool>("spherical_transport")) {
    spherical::Step(pm, tm.time, dt, tm.ncycle);
  } else if (!HasCPTWork(pm, dt)) {
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
  auto status = pkg->Param<bool>("spherical_transport")
                    ? TaskListStatus::complete
                    : charon::RadiationStep(pm, tm.time, dt).Execute();
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
