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

// C++ includes
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <parthenon_mpi.hpp>

// Charon includes
#include "charon.hpp"
#include "charon_utils.hpp"
#include "transport_utils.hpp"

namespace charon {

using namespace units;
using parthenon::TaskQualifier;
using TQ = TaskQualifier;

// TODO(BRR) Move these methods to Parthenon
TaskStatus MeshResetCommunication(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->ResetCommunication();
  }

  return TaskStatus::complete;
}

TaskStatus MeshSend(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->Send(BoundaryCommSubset::all);
  }

  return TaskStatus::complete;
}

TaskStatus MeshReceive(MeshData<Real> *md) {
  TaskStatus status = TaskStatus::complete;
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    auto local_status = sc->Receive(BoundaryCommSubset::all);
    if (local_status == TaskStatus::incomplete) {
      status = TaskStatus::incomplete;
    }
  }

  return status;
}
namespace {

void InitializePlasmaInventory(Mesh *pm, const SimTime &tm) {
  namespace fj = field::charon;
  const auto pkg = pm->packages.Get("charon");
  if (!pkg->Param<bool>("tn_plasma")) return;
  if (tm.ncycle != 0) {
    // Parameters are reconstructed at startup; recover the logging counter from
    // the restart-backed cell inventory rather than resetting it to zero.
    Real sampled = 0;
    for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
      auto md = pm->mesh_data.GetOrAdd("base", p);
      auto desc = MakePackDescriptor<fj::sampled_alphas>(pm->resolved_packages.get());
      auto v = desc.GetPack(md.get());
      if (v.GetNBlocks() == 0) continue;
      auto ib = md->GetBoundsI(IndexDomain::interior);
      auto jb = md->GetBoundsJ(IndexDomain::interior);
      auto kb = md->GetBoundsK(IndexDomain::interior);
      Real part = 0;
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "RestoreSampledAlphaInventory",
          DevExecSpace(), 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(int b, int k, int j, int i, Real &sum) {
            sum += v(b, fj::sampled_alphas(), k, j, i);
          },
          Kokkos::Sum<Real>(part));
      sampled += part;
    }
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &sampled, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif
    pkg->UpdateParam<Real>("cumulative_alphas_sourced", sampled);
    return;
  }
  // Restarted cycles retain the extensive inventory, including fractional births.
  for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
    auto md = pm->mesh_data.GetOrAdd("base", p);
    auto desc =
        MakePackDescriptor<fj::pending_alphas, fj::physical_alphas, fj::sampled_alphas,
                           fj::electron_heat, fj::ion_heat, fj::escaped_alphas,
                           fj::escaped_energy>(pm->resolved_packages.get());
    auto v = desc.GetPack(md.get());
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    if (v.GetNBlocks() == 0) continue;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "InitializePlasmaInventory", DevExecSpace(), 0,
        v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(int b, int k, int j, int i) {
          v(b, fj::pending_alphas(), k, j, i) = 0;
          v(b, fj::physical_alphas(), k, j, i) = 0;
          v(b, fj::sampled_alphas(), k, j, i) = 0;
          v(b, fj::electron_heat(), k, j, i) = 0;
          v(b, fj::ion_heat(), k, j, i) = 0;
          v(b, fj::escaped_alphas(), k, j, i) = 0;
          v(b, fj::escaped_energy(), k, j, i) = 0;
        });
  }
}

template <typename T>
struct MinMeanMax {
  T min;
  Real mean;
  T max;
};

#ifdef MPI_PARALLEL
template <typename T>
MPI_Datatype DiagnosticsMpiType();
template <>
MPI_Datatype DiagnosticsMpiType<int>() {
  return MPI_INT;
}
template <>
MPI_Datatype DiagnosticsMpiType<long long>() {
  return MPI_LONG_LONG;
}
template <>
MPI_Datatype DiagnosticsMpiType<Real>() {
  return MPI_PARTHENON_REAL;
}
#endif

template <typename T>
MinMeanMax<T> GlobalMinMeanMax(T local_value) {
  T global_min = local_value;
  T global_sum = local_value;
  T global_max = local_value;
#ifdef MPI_PARALLEL
  const auto mpi_type = DiagnosticsMpiType<T>();
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_min, 1, mpi_type, MPI_MIN,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_sum, 1, mpi_type, MPI_SUM,
                                    MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, mpi_type, MPI_MAX,
                                    MPI_COMM_WORLD));
#endif
  return {global_min, static_cast<Real>(global_sum) /
                          static_cast<Real>(parthenon::Globals::nranks),
          global_max};
}

std::string RankLocalDiagnosticsFilename(const std::string &basename) {
  std::ostringstream os;
  os << basename << ".rank" << std::setfill('0') << std::setw(6)
     << parthenon::Globals::my_rank << ".csv";
  return os.str();
}

} // namespace

// Fold the step tallies into persistent totals once per cell, avoiding a second
// contended atomic update for every particle segment. Totals survive data output.
TaskStatus AccumulateTransportTallies(MeshData<Real> *md) {
  namespace fj = field::charon;
  auto &pkg = md->GetMeshPointer()->packages.Get("charon");
  const bool tally_mass_momentum = pkg->Param<bool>("tally_mass_momentum");
  auto desc = MakePackDescriptor<fj::energy_tally, fj::energy_delta, fj::mass_delta,
      fj::momentum_delta, fj::energy_tally_step, fj::energy_delta_step,
      fj::mass_delta_step, fj::momentum_delta_step>(
          md->GetMeshPointer()->resolved_packages.get());
  auto v = desc.GetPack(md);
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  if (v.GetNBlocks() == 0) return TaskStatus::complete;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AccumulateTransportTallies", DevExecSpace(),
      0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        v(b, fj::energy_tally(), k, j, i) += v(b, fj::energy_tally_step(), k, j, i);
        v(b, fj::energy_delta(), k, j, i) += v(b, fj::energy_delta_step(), k, j, i);
        if (tally_mass_momentum) {
          v(b, fj::mass_delta(), k, j, i) += v(b, fj::mass_delta_step(), k, j, i);
          for (int d = 0; d < 3; ++d) {
            v(b, fj::momentum_delta(d), k, j, i) +=
                v(b, fj::momentum_delta_step(d), k, j, i);
          }
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection charon::RadiationStep
//! \brief Construct the collection of tasks that contribute to a complete radiation cycle
//!        from t to t + dt, including setting up derived quantities, sourcing particles,
//!        transporting particles, and communicating particles.
//template <FrequencyType FT, ParticleType PT>
TaskCollection RadiationStep(Mesh *pmesh, const Real t_start, const Real dt) {
  namespace fj = field::charon;

  auto &cpt_pkg = pmesh->packages.Get("charon");
  const auto &max_transport_iterations =
      cpt_pkg->template Param<int>("max_transport_iterations");
  const bool &transport_alphas =
      cpt_pkg->template Param<bool>("transport_alphas");

  TaskCollection tc;
  TaskID none(0);

  auto &timing_region0 = tc.AddRegion(1);
  {
    auto &tl = timing_region0[0];
    tl.AddTask(none, []() {
      Kokkos::Profiling::pushRegion("Charon::Timestep");
      return TaskStatus::complete;
    });
  }

  const int num_partitions = pmesh->DefaultNumPartitions();
  PARTHENON_REQUIRE(
      num_partitions == 1,
      "Iterative tasking may not support multiple partitions per rank as of 2024/5/14")
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    // Get base register for particles and DDMC fields register (if applicable)
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    //auto &md_ddmc = pmesh->mesh_data.GetOrAdd("ddmc_reg", i);

    // prepare for iterative transport loop
    auto derived = tl.AddTask(none, UpdateDerivedTransportFields, base.get(), dt);
    auto source = derived;
    // source particles
    source = tl.AddTask(derived, charon::SourceAlphas<MeshData<Real>>, base.get(), t_start, dt);

    auto transport_done = source;
    if (transport_alphas) {
      auto bcs = source;

      // keep pushing particles until there are none left
      auto [itl, push] = tl.AddSublist(bcs, {1, max_transport_iterations});
      auto time_start = itl.AddTask(none, []() {
        Kokkos::Profiling::pushRegion("Charon::TransportLoop");
        return TaskStatus::complete;
      });
      auto transport = time_start;

      transport = itl.AddTask(time_start, TransportAlphas, base.get(), t_start, dt);

      auto reset_comms = itl.AddTask(transport, MeshResetCommunication, base.get());
      auto send = itl.AddTask(reset_comms, MeshSend, base.get());
      auto receive = itl.AddTask(transport | send, MeshReceive, base.get());
      auto time_stop = itl.AddTask(receive, []() {
        Kokkos::Profiling::popRegion(/*Charon::TransportLoop*/);
        return TaskStatus::complete;
      });
      auto complete = itl.AddTask(TQ::once_per_region | TQ::global_sync | TQ::completion,
                                  time_stop, CheckCompletion, base.get(), t_start + dt);
      transport_done = push;
    }

    // Update radiation fields
    auto eval_rad =
        tl.AddTask(transport_done, charon::EvaluateRadiationEnergy<MeshData<Real>>,
                   base.get());

    // Control particle population
    auto control_pop = tl.AddTask(eval_rad, charon::ControlPopulation, base.get());
    if (!cpt_pkg->Param<bool>("compact_tallies")) {
      tl.AddTask(control_pop, AccumulateTransportTallies, base.get());
    }

    // TODO: Defrag particles? Verify parth swarm defrag mechanics before uncommenting
    // auto defrag_pop = tl.AddTask(control_pop, charon::DefragParticles, base.get());
  }

  auto &timing_region1 = tc.AddRegion(1);
  {
    auto &tl = timing_region1[0];
    tl.AddTask(none, []() {
      Kokkos::Profiling::popRegion(/* Charon::Timestep */);
      return TaskStatus::complete;
    });
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Charon::Initialize_impl
//! \brief Initialize the Charon physics package. This function defines and sets the
//! parameters associated with Charon, and enrolls the data variables associated with
//! this physics package, for everything not related to frequency discretization.
std::shared_ptr<StateDescriptor>
Initialize_impl(ParameterInput *pin, std::string block_name,
                const bool allow_compact_tallies) {
  auto pkg = std::make_shared<StateDescriptor>("charon");
  pkg->AddParam("tn_plasma", pin->GetOrAddBoolean(block_name, "tn_plasma", false));
  if (pkg->Param<bool>("tn_plasma")) {
    const Real cutoff = pin->GetOrAddReal(block_name, "thermalization_mev", 0.2);
    const Real tolerance = pin->GetOrAddReal(block_name, "plasma_velocity_tolerance", 1.e-4);
    const Real fraction = pin->GetOrAddReal(block_name, "plasma_fractional_step", 0.02);
    const Real ion_fraction = pin->GetOrAddReal(block_name, "constant_ion_fraction", 0.5);
    const int limit = pin->GetOrAddInteger(block_name, "plasma_max_substeps", 100000);
    PARTHENON_REQUIRE(std::isfinite(cutoff) && cutoff > 0 && cutoff < 3.54 &&
        std::isfinite(tolerance) && tolerance > 0 && tolerance < 1 &&
        std::isfinite(fraction) && fraction > 0 && fraction <= 0.1 &&
        ion_fraction >= 0 && ion_fraction <= 1 && limit > 0,
        "Invalid plasma transport controls");
    PARTHENON_REQUIRE(parthenon::IsCoord<parthenon::UniformCartesian>(),
                      "TN plasma transport currently requires Cartesian coordinates");
    pkg->AddParam("plasma_thermalization_energy", cutoff * units::MeV_to_ERG);
    pkg->AddParam("plasma_velocity_tolerance", tolerance);
    pkg->AddParam("plasma_fractional_step", fraction);
    pkg->AddParam("plasma_max_substeps", limit);
    pkg->AddParam("constant_ion_fraction", ion_fraction);
  }
  const std::string default_stopping_power_model =
      pin->DoesParameterExist("charon", "stopping_power_model")
          ? pin->GetString("charon", "stopping_power_model")
          : "constant";
  pkg->AddParam<std::string>(
      "stopping_power_model",
      pin->GetOrAddString(block_name, "stopping_power_model",
                          default_stopping_power_model));
  std::string spower_model = pkg->Param<std::string>("stopping_power_model");
  PARTHENON_REQUIRE(
      spower_model == "LiPetrasso" || spower_model == "constant" ||
          spower_model == "quadratic" || spower_model == "density_linear" ||
          spower_model == "temperature_linear",
      "Stopping Power Model Not Supported! Please Choose LiPetrasso, quadratic, "
      "constant, density_linear, or temperature_linear");
  const Real default_stopping_power_value =
      pin->DoesParameterExist("charon", "stopping_power_value")
          ? pin->GetReal("charon", "stopping_power_value")
          : 1.0;
  pkg->AddParam<Real>(
      "stopping_power_value",
      pin->GetOrAddReal(block_name, "stopping_power_value",
                        default_stopping_power_value));
  const Real default_stopping_power_density_reference =
      pin->DoesParameterExist("charon", "stopping_power_density_reference")
          ? pin->GetReal("charon", "stopping_power_density_reference")
          : 1.0;
  pkg->AddParam<Real>(
      "stopping_power_density_reference",
      pin->GetOrAddReal(block_name, "stopping_power_density_reference",
                        default_stopping_power_density_reference));
  const Real default_stopping_power_density_slope =
      pin->DoesParameterExist("charon", "stopping_power_density_slope")
          ? pin->GetReal("charon", "stopping_power_density_slope")
          : 1.0;
  pkg->AddParam<Real>(
      "stopping_power_density_slope",
      pin->GetOrAddReal(block_name, "stopping_power_density_slope",
                        default_stopping_power_density_slope));
  const Real default_stopping_power_temperature_reference =
      pin->DoesParameterExist("charon", "stopping_power_temperature_reference")
          ? pin->GetReal("charon", "stopping_power_temperature_reference")
          : (pin->DoesParameterExist(block_name, "initial_temperature")
                 ? pin->GetReal(block_name, "initial_temperature")
                 : (pin->DoesParameterExist("mcblock", "initial_temperature")
                        ? pin->GetReal("mcblock", "initial_temperature")
                        : 1.0));
  pkg->AddParam<Real>(
      "stopping_power_temperature_reference",
      pin->GetOrAddReal(block_name, "stopping_power_temperature_reference",
                        default_stopping_power_temperature_reference));
  const Real default_stopping_power_temperature_slope =
      pin->DoesParameterExist("charon", "stopping_power_temperature_slope")
          ? pin->GetReal("charon", "stopping_power_temperature_slope")
          : 1.0;
  pkg->AddParam<Real>(
      "stopping_power_temperature_slope",
      pin->GetOrAddReal(block_name, "stopping_power_temperature_slope",
                        default_stopping_power_temperature_slope));

  // Total number of particles. SourceStrategy::qfield can be driven entirely by the
  // host Q field, so this defaults to zero for downstream hosts such as Artemis.
  int num_particles = pin->GetOrAddInteger(block_name, "num_particles", 0);
  pkg->AddParam<>("num_particles", num_particles);
  int num_particles_per_pulse = pin->GetOrAddInteger(block_name, "num_particles_per_pulse", num_particles);
  pkg->AddParam<>("num_particles_per_pulse", num_particles_per_pulse);
  const std::string variable_field =
      pin->GetOrAddString(block_name, "variable_field", "constant");
  PARTHENON_REQUIRE(variable_field == "constant" || variable_field == "none" ||
                        variable_field == "sine_test",
                    "variable_field must be 'constant', 'none', or 'sine_test'");
  pkg->AddParam<std::string>("variable_field", variable_field);
  Real dnpc_min = pin->GetOrAddReal(block_name, "dnpc_min", 20.0);
  PARTHENON_REQUIRE(dnpc_min >= 1.0, "dnpc_min must be at least 1");
  pkg->AddParam<>("dnpc_min", dnpc_min);
  Real dt = pin->GetOrAddReal(block_name, "dt", std::numeric_limits<Real>::max());
  pkg->AddParam<>("dt", dt);
  Real alpha_weight = pin->GetOrAddReal(block_name, "alpha_weight", 1.0);
  PARTHENON_REQUIRE(alpha_weight > 0.0, "alpha_weight must be positive");
  pkg->AddParam<>("alpha_weight", alpha_weight);
  const Real alpha_source_physical_cap = pin->GetOrAddReal(
      block_name, "alpha_source_physical_cap", 1.0e20);
      //block_name, "alpha_source_physical_cap", std::numeric_limits<Real>::max());
  PARTHENON_REQUIRE(alpha_source_physical_cap > 0.0,
                    "alpha_source_physical_cap must be positive");
  pkg->AddParam<Real>("alpha_source_physical_cap", alpha_source_physical_cap);
  pkg->AddParam<Real>("cumulative_alphas_sourced", 0.0,
                      parthenon::Params::Mutability::Mutable);

  // sourcing frequency
  const int default_pulses = pin->DoesParameterExist("charon", "pulses") ? pin->GetInteger("charon", "pulses"): 1;
  const int pulses = pin->GetOrAddInteger(block_name, "pulses", default_pulses);
  pkg->AddParam<int>("pulses", pulses);
  int source_time_pulses = pin->GetOrAddInteger(block_name, "source_time_pulses", 1);
  PARTHENON_REQUIRE(source_time_pulses > 0, "source_time_pulses must be positive");
  pkg->AddParam<int>("source_time_pulses", source_time_pulses);
  const bool line_source_from_qfield =
      pin->GetOrAddBoolean(block_name, "line_source_from_qfield", false);
  pkg->AddParam<bool>("line_source_from_qfield", line_source_from_qfield);
  Real default_fuel_density_threshold = 1.0;
  if (pin->DoesParameterExist("problem", "fuel_density_threshold")) {
    default_fuel_density_threshold = pin->GetReal("problem", "fuel_density_threshold");
  } else if (pin->DoesParameterExist("problem", "fuel_density") &&
             pin->DoesParameterExist("problem", "ambient_density")) {
    default_fuel_density_threshold =
        0.5 * (pin->GetReal("problem", "fuel_density") +
               pin->GetReal("problem", "ambient_density"));
  } else if (pin->DoesParameterExist("problem", "fuel_density")) {
    default_fuel_density_threshold = 0.5 * pin->GetReal("problem", "fuel_density");
  } else if (pin->DoesParameterExist("problem", "amr_density_threshold")) {
    default_fuel_density_threshold = pin->GetReal("problem", "amr_density_threshold");
  }
  const Real fuel_density_threshold = pin->GetOrAddReal(
      block_name, "fuel_density_threshold", default_fuel_density_threshold);
  pkg->AddParam<Real>("fuel_density_threshold", fuel_density_threshold);
  pkg->AddParam<int>("transport_tally_reset_file_count", 0,
                     parthenon::Params::Mutability::Mutable);
  pkg->AddParam<int>("weak_scaling_diagnostics_output_count", 0,
                     parthenon::Params::Mutability::Mutable);
  pkg->AddParam<Real>("transport_walltime_interval", 0.0,
                      parthenon::Params::Mutability::Mutable);
  const bool weak_scaling_diagnostics =
      pin->GetOrAddBoolean(block_name, "weak_scaling_diagnostics", true);
  pkg->AddParam<bool>("weak_scaling_diagnostics", weak_scaling_diagnostics);
  const std::string weak_scaling_diagnostics_basename =
      pin->GetOrAddString(block_name, "weak_scaling_diagnostics_basename",
                          "weak_scaling_diagnostics");
  pkg->AddParam<std::string>("weak_scaling_diagnostics_basename",
                             weak_scaling_diagnostics_basename);
  const bool do_population_control =
      pin->GetOrAddBoolean(block_name, "do_population_control", true);
  pkg->AddParam<bool>("do_population_control", do_population_control);
  int nlim = pin->GetOrAddInteger("parthenon/time", "nlim", -1);
  pkg->AddParam<int>("nlim", nlim);
  Real tlim = pin->GetOrAddReal(
    "parthenon/time", "tlim", std::numeric_limits<Real>::infinity(),
    "physical time at which to end the simulation");
  //Real tlim = pin->GetOrAddReal("parthenon/time", "tlim", -1.0);
  pkg->AddParam<Real>("tlim", tlim);

  // minimum occupancy of swarm below which defragmentation is triggered.
  Real min_swarm_occupancy = pin->GetOrAddReal(block_name, "min_swarm_occupancy", 0.);
  PARTHENON_REQUIRE(min_swarm_occupancy >= 0 && min_swarm_occupancy < 1.0,
                    "Minimum allowable swarm occupancy must be >= 0 and less than 1");
  pkg->AddParam<>("min_swarm_occupancy", min_swarm_occupancy);

  // RNG
  bool unique_rank_seeds = pin->GetOrAddBoolean(block_name, "unique_rank_seeds", true);
  pkg->AddParam<>("unique_rank_seeds", unique_rank_seeds);
  int seed = pin->GetOrAddInteger(block_name, "seed", 123);
  pkg->AddParam<>("seed", unique_rank_seeds ? seed + Globals::my_rank : seed);
  RngPool rng_pool(seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  // Transport numerics
  int max_transport_iterations =
      pin->GetOrAddInteger(block_name, "max_transport_iterations", 10000);
  pkg->AddParam<>("max_transport_iterations", max_transport_iterations);
  bool transport_alphas = pin->GetOrAddBoolean(block_name, "transport_alphas", true);
  pkg->AddParam<bool>("transport_alphas", transport_alphas);
  bool do_feedback = pin->GetOrAddBoolean(block_name, "do_feedback", true);
  pkg->AddParam<bool>("do_feedback", do_feedback);
  bool verbose = pin->GetOrAddBoolean(block_name, "verbose", false);
  pkg->AddParam<bool>("verbose", verbose);

  // Sourcing strategy
  const std::string default_source_strategy =
      pin->DoesParameterExist("mcblock", "source_strategy")
          ? pin->GetString("mcblock", "source_strategy")
          : "isotropic";
  std::string strategy =
      pin->GetOrAddString(block_name, "source_strategy", default_source_strategy);
  pkg->AddParam<std::string>("source_strategy_str", strategy);

  // mcblock sourcing
  const std::string default_alpha_source_file =
      pin->DoesParameterExist("mcblock", "alpha_source_file")
          ? pin->GetString("mcblock", "alpha_source_file")
          : "none";
  std::string alpha_source_file =
      pin->GetOrAddString(block_name, "alpha_source_file", default_alpha_source_file);
  pkg->AddParam<std::string>("alpha_source_file", alpha_source_file);
  const std::string default_alpha_source_mode =
      pin->DoesParameterExist("mcblock", "alpha_source_mode")
          ? pin->GetString("mcblock", "alpha_source_mode")
          : "total";
  std::string alpha_source_mode =
      pin->GetOrAddString(block_name, "alpha_source_mode", default_alpha_source_mode);
  pkg->AddParam<std::string>("alpha_source_mode", alpha_source_mode);

  // pick the source strategy
  SourceStrategy source_strategy;
  if (strategy == "t0") { source_strategy = SourceStrategy::t0;
  } else if (strategy == "isotropic") { source_strategy = SourceStrategy::isotropic;
  } else if (strategy == "center") { source_strategy = SourceStrategy::center;
  } else if (strategy == "point") { source_strategy = SourceStrategy::point;
  } else if (strategy == "xhat") { source_strategy = SourceStrategy::xhat;
  } else if (strategy == "line") { source_strategy = SourceStrategy::line;
  } else if (strategy == "plane") { source_strategy = SourceStrategy::plane;
  } else if (strategy == "fuel") { source_strategy = SourceStrategy::fuel;
  } else if (strategy == "qfield" || strategy == "Q") {
      source_strategy = SourceStrategy::qfield;
  } else {
    PARTHENON_FAIL(
        "Unsupported source_strategy! Choose: isotropic, center, point, xhat, line, "
        "plane, fuel, t0, or qfield.");
  }
  pkg->AddParam<SourceStrategy>("source_strategy", source_strategy);

  const bool cpt_diagnostics =
      pin->GetOrAddBoolean(block_name, "cpt_diagnostics", false);
  pkg->AddParam<bool>("cpt_diagnostics", cpt_diagnostics);
  // RIOT can supply an energy-only default before initializing Charon. Read an
  // existing value without registering a second, conflicting input default.
  const bool tally_mass_momentum =
      pin->DoesParameterExist(block_name, "tally_mass_momentum")
          ? pin->GetBoolean(block_name, "tally_mass_momentum")
          : pin->GetOrAddBoolean(block_name, "tally_mass_momentum", false);
  pkg->AddParam<bool>("tally_mass_momentum", tally_mass_momentum);
  // Only hosts that consume the combined deposition field opt into this path.
  // Explicit legacy tally outputs retain their cumulative/step semantics even
  // when the optional transport diagnostics are disabled.
  const std::vector<std::string> legacy_tallies = {
      field::charon::energy_tally::name(), field::charon::energy_delta::name(),
      field::charon::mass_delta::name(), field::charon::momentum_delta::name(),
      field::charon::energy_tally_step::name(), field::charon::energy_delta_step::name(),
      field::charon::mass_delta_step::name(), field::charon::momentum_delta_step::name()};
  bool legacy_tally_output = false;
  for (int n = 0; n < 100; ++n) {
    const std::string output = "parthenon/output" + std::to_string(n);
    if (!pin->DoesParameterExist(output, "variables")) continue;
    for (const auto &variable : pin->GetVector<std::string>(output, "variables")) {
      if (variable == "all" || variable.find('*') != std::string::npos ||
          std::find(legacy_tallies.begin(), legacy_tallies.end(), variable) !=
              legacy_tallies.end()) {
        legacy_tally_output = true;
      }
    }
  }
  const bool compact_tallies = allow_compact_tallies && !cpt_diagnostics &&
                                !tally_mass_momentum && !legacy_tally_output;
  pkg->AddParam<bool>("compact_tallies", compact_tallies);
  pkg->AddParam<std::shared_ptr<TransportScratch>>(
      "transport_scratch", std::make_shared<TransportScratch>());

  // Fail early when an output explicitly requests a disabled diagnostic field.
  const std::vector<std::string> diagnostic_fields = {
      field::charon::source_ew_per_cell::name(),
      field::charon::source_num_per_cell::name(),
      field::charon::delta_num_per_cell::name(),
      field::charon::active_ew_per_cell::name(),
      field::charon::active_num_per_cell::name(),
      field::charon::old_active_ew_per_cell::name(),
      field::charon::active_ke_tally::name(), field::charon::Phi::name(),
      field::charon::Phi_uq::name(), field::charon::ave_energy::name(),
      field::charon::count_tally::name(), field::charon::residence_tally::name(),
      field::charon::spower_tally::name(), field::charon::spower_sum::name(),
      field::charon::spower_count::name()};
  if (!cpt_diagnostics) {
    for (int n = 0; n < 100; ++n) {
      const std::string output = "parthenon/output" + std::to_string(n);
      if (!pin->DoesParameterExist(output, "variables")) continue;
      const auto variables = pin->GetVector<std::string>(output, "variables");
      for (const auto &variable : variables) {
        if (std::find(diagnostic_fields.begin(), diagnostic_fields.end(), variable) !=
            diagnostic_fields.end()) {
          PARTHENON_FAIL(output + " requests " + variable +
                         "; set <charon>/cpt_diagnostics=true or remove this variable.");
        }
      }
    }
  }

  // Swarm and swarm variables
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(alphas_swarm_name, swarm_metadata);

  Metadata mreal({Metadata::Real, Metadata::Independent});
  pkg->AddSwarmValue(particle::alphas::time::name(), alphas_swarm_name, mreal);
  pkg->AddSwarmValue(particle::alphas::weight::name(), alphas_swarm_name, mreal);
  pkg->AddSwarmValue(particle::alphas::energy::name(), alphas_swarm_name, mreal);

  Metadata mrealv({Metadata::Real, Metadata::Vector, Metadata::Independent}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::alphas::v::name(), alphas_swarm_name, mrealv);
  Metadata mintv({Metadata::Integer, Metadata::Vector, Metadata::Independent}, std::vector<int>{3});
  pkg->AddSwarmValue(particle::alphas::ijk::name(), alphas_swarm_name, mintv);

  // Sourcing fields
  // Tallies are local cell accumulators; particle migration carries the particle
  // state, not ghost tally values. Persistent totals still transfer on remeshing.
  Metadata m_onecopy({Metadata::Cell, Metadata::OneCopy});
  Metadata m_accum({Metadata::Cell, Metadata::OneCopy, Metadata::Restart,
                    Metadata::ForceRemeshComm});
  Metadata m_vector_accum({Metadata::Cell, Metadata::Vector, Metadata::OneCopy,
                          Metadata::Restart, Metadata::ForceRemeshComm},
                         std::vector<int>{3});
  Metadata m_vector_onecopy({Metadata::Cell, Metadata::Vector, Metadata::OneCopy},
                           std::vector<int>{3});
  if (cpt_diagnostics) {
    pkg->AddField(field::charon::source_ew_per_cell::name(), m_onecopy);
    pkg->AddField(field::charon::source_num_per_cell::name(), m_onecopy);
    pkg->AddField(field::charon::delta_num_per_cell::name(), m_onecopy);
  }
  if (compact_tallies) {
    // Per-step feedback workspace only: no cumulative output, restart, remesh
    // communication, or mass/momentum allocation on the energy-only path.
    pkg->AddField(field::charon::deposited_energy_step::name(), m_onecopy);
  } else {
    pkg->AddField(field::charon::energy_delta::name(), m_accum);
    pkg->AddField(field::charon::mass_delta::name(), m_accum);
    pkg->AddField(field::charon::momentum_delta::name(), m_vector_accum);
    pkg->AddField(field::charon::energy_tally::name(), m_accum);
    pkg->AddField(field::charon::energy_delta_step::name(), m_onecopy);
    pkg->AddField(field::charon::mass_delta_step::name(), m_onecopy);
    pkg->AddField(field::charon::momentum_delta_step::name(), m_vector_onecopy);
    pkg->AddField(field::charon::energy_tally_step::name(), m_onecopy);
  }

  // Population control fields
  if (cpt_diagnostics) {
    pkg->AddField(field::charon::active_ew_per_cell::name(), m_onecopy);
    pkg->AddField(field::charon::active_num_per_cell::name(), m_onecopy);
    pkg->AddField(field::charon::old_active_ew_per_cell::name(), m_onecopy);
    pkg->AddField(field::charon::active_ke_tally::name(), m_onecopy);
  }

  // Radiation timestep
  pkg->EstimateTimestepMesh = EstimateTimestepMesh;
  pkg->UserWorkBeforeLoopMesh = [](Mesh *pm, ParameterInput *input, SimTime &tm) {
    InitializePlasmaInventory(pm, tm);
    if (tm.ncycle == 0) ResetTransportOutputTallies(pm, input, tm);
  };
  pkg->UserWorkBeforeOutputMesh = PrepareTransportOutput;

  // Scalar flux in transport loop
  if (cpt_diagnostics) {
    pkg->AddField(field::charon::Phi::name(), m_accum);
    pkg->AddField(field::charon::Phi_uq::name(), m_accum);

    // average energy
    pkg->AddField(field::charon::ave_energy::name(),  m_accum);
    pkg->AddField(field::charon::count_tally::name(), m_accum);
    pkg->AddField(field::charon::residence_tally::name(), m_accum);
    pkg->AddField(field::charon::spower_tally::name(), m_onecopy);
    pkg->AddField(field::charon::spower_sum::name(), m_accum);
    pkg->AddField(field::charon::spower_count::name(), m_accum);
  }
  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Charon::Initialize
//! \brief Initialize the Charon physics package. This function defines and sets the
//! parameters associated with Charon, and enrolls the data variables associated with
//! this physics package, specific to the gray frequency discretization.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, std::string block_name,
                                            const bool allow_compact_tallies) {
  auto pkg = Initialize_impl(pin, block_name, allow_compact_tallies);
  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Charon::EstimateTimestepMesh
//! \brief Compute radiation timestep
Real EstimateTimestepMesh(MeshData<Real> *md) {
  return md->GetParentPointer()->packages.Get("charon")->template Param<Real>("dt");
}

//----------------------------------------------------------------------------------------
//! \fn  void PrepareTransportOutput
//! \brief Fill instantaneous output-only alpha diagnostics immediately before output. Purely diagnostic
void PrepareTransportOutput(Mesh *pmesh, [[maybe_unused]] ParameterInput *pin,
                            [[maybe_unused]] SimTime &tm) {
  namespace fj = field::charon;
  namespace pa = particle::alphas;
  namespace sp = swarm_position;

  if (!pmesh->packages.Get("charon")->Param<bool>("cpt_diagnostics")) return;

  const int num_partitions = pmesh->DefaultNumPartitions();
  for (int p = 0; p < num_partitions; ++p) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    auto &resolved_pkgs = pmesh->resolved_packages;

    static auto desc = MakePackDescriptor<fj::active_ke_tally, fj::active_num_per_cell,
        fj::active_ew_per_cell, fj::spower_tally, fj::spower_sum,
        fj::spower_count>(resolved_pkgs.get());
    auto vmesh = desc.GetPack(md);

    const auto &ib = md->GetBoundsI(IndexDomain::interior);
    const auto &jb = md->GetBoundsJ(IndexDomain::interior);
    const auto &kb = md->GetBoundsK(IndexDomain::interior);
    const int nblocks = vmesh.GetNBlocks();

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PrepareTransportOutput::zero-active-ke",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          vmesh(b, fj::active_ke_tally(), k, j, i) = 0.0;
          vmesh(b, fj::active_num_per_cell(), k, j, i) = 0.0;
          vmesh(b, fj::active_ew_per_cell(), k, j, i) = 0.0;
        });

    static auto pdesc_r = MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, pa::energy,
                                                  pa::weight>(alphas_swarm_name);
    static auto pdesc_i = MakeSwarmPackDescriptor<pa::ijk>(alphas_swarm_name);
    auto ppack_r = pdesc_r.GetPack(md);
    auto ppack_i = pdesc_i.GetPack(md);
    const int nparticles_per_pack = ppack_r.GetMaxFlatIndex();

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PrepareTransportOutput::active-ke", DevExecSpace(), 0,
        nparticles_per_pack, KOKKOS_LAMBDA(const int idx) {
          auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
          const auto &swarm_d = ppack_r.GetContext(b);
          if (swarm_d.IsActive(n)) {
            const Real &x = ppack_r(b, sp::x(), n);
            const Real &y = ppack_r(b, sp::y(), n);
            const Real &z = ppack_r(b, sp::z(), n);
            int &ip = ppack_i(b, pa::ijk(0), n);
            int &jp = ppack_i(b, pa::ijk(1), n);
            int &kp = ppack_i(b, pa::ijk(2), n);
            // A particle can reach census immediately after crossing a meshblock
            // boundary. In that case its migrated ijk values still refer to the
            // source block, so refresh them from its position before output tallying.
            swarm_d.Xtoijk(x, y, z, ip, jp, kp);
            const Real &ee = ppack_r(b, pa::energy(), n);
            const Real &ww = ppack_r(b, pa::weight(), n);
            Kokkos::atomic_add(&vmesh(b, fj::active_ke_tally(), kp, jp, ip), ee * ww);
            Kokkos::atomic_add(&vmesh(b, fj::active_num_per_cell(), kp, jp, ip), 1.0);
            Kokkos::atomic_add(&vmesh(b, fj::active_ew_per_cell(), kp, jp, ip), ww);
          }
        });
    parthenon::par_for(
      DEFAULT_LOOP_PATTERN,
      "PrepareTransportOutput::mean-spower",
      parthenon::DevExecSpace(),
      0, nblocks - 1,
      kb.s, kb.e,
      jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(
        const int &b,
        const int &k,
        const int &j,
        const int &i) {
      const Real count =
        vmesh(b, fj::spower_count(), k, j, i);
  
        if (count > 0.0) {
          vmesh(b, fj::spower_tally(), k, j, i) =
              vmesh(b, fj::spower_sum(), k, j, i) / count;
        } else {
          vmesh(b, fj::spower_tally(), k, j, i) = 0.0;
        }
      });
    }
  }

//----------------------------------------------------------------------------------------
//! \fn  void ResetTransportOutputTallies
//! \brief Initialize persistent cumulative tallies for a new simulation.
void ResetTransportOutputTallies(Mesh *pmesh, [[maybe_unused]] ParameterInput *pin,
                                 [[maybe_unused]] SimTime &tm) {
  namespace fj = field::charon;

  const int num_partitions = pmesh->DefaultNumPartitions();
  if (pmesh->packages.Get("charon")->Param<bool>("compact_tallies")) {
    for (int p = 0; p < num_partitions; ++p) {
      UpdateDerivedTransportFields(pmesh->mesh_data.GetOrAdd("base", p).get(), 0.0);
    }
    return;
  }
  for (int p = 0; p < num_partitions; ++p) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    auto &resolved_pkgs = pmesh->resolved_packages;

    static auto desc =
        MakePackDescriptor<fj::energy_tally, fj::energy_delta, fj::mass_delta,
                           fj::momentum_delta>(resolved_pkgs.get());
    auto vmesh = desc.GetPack(md);

    const auto &ib = md->GetBoundsI(IndexDomain::interior);
    const auto &jb = md->GetBoundsJ(IndexDomain::interior);
    const auto &kb = md->GetBoundsK(IndexDomain::interior);
    const int nblocks = vmesh.GetNBlocks();

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ResetTransportOutputTallies", parthenon::DevExecSpace(),
        0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          vmesh(b, fj::energy_tally(), k, j, i) = 0.0;
          vmesh(b, fj::energy_delta(), k, j, i) = 0.0;
          vmesh(b, fj::mass_delta(), k, j, i) = 0.0;
          for (int d = 0; d < 3; ++d) {
            vmesh(b, fj::momentum_delta(d), k, j, i) = 0.0;
          }
        });
    if (pmesh->packages.Get("charon")->Param<bool>("cpt_diagnostics")) {
      auto diagnostic_desc = MakePackDescriptor<
          fj::Phi, fj::Phi_uq, fj::ave_energy, fj::count_tally, fj::residence_tally,
          fj::active_ke_tally, fj::spower_tally, fj::spower_sum,
          fj::spower_count>(resolved_pkgs.get());
      auto vdiag = diagnostic_desc.GetPack(md);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "ResetTransportDiagnostics", DevExecSpace(),
          0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            vdiag(b, fj::Phi(), k, j, i) = 0.0;
            vdiag(b, fj::Phi_uq(), k, j, i) = 0.0;
            vdiag(b, fj::ave_energy(), k, j, i) = 0.0;
            vdiag(b, fj::count_tally(), k, j, i) = 0.0;
            vdiag(b, fj::residence_tally(), k, j, i) = 0.0;
            vdiag(b, fj::active_ke_tally(), k, j, i) = 0.0;
            vdiag(b, fj::spower_tally(), k, j, i) = 0.0;
            vdiag(b, fj::spower_sum(), k, j, i) = 0.0;
            vdiag(b, fj::spower_count(), k, j, i) = 0.0;
          });
    }
  }
}

//! \fn  void WriteWeakScalingDiagnostics
//! \brief Write cumulative tallies and interval timing to rank-local diagnostics.
void WriteWeakScalingDiagnostics(Mesh *pmesh, [[maybe_unused]] ParameterInput *pin,
                                 SimTime &tm) {
  namespace fj = field::charon;

  auto &cpt_pkg = pmesh->packages.Get("charon");
  if (!cpt_pkg->Param<bool>("weak_scaling_diagnostics")) return;
  const bool diagnostics = cpt_pkg->Param<bool>("cpt_diagnostics");

  long long active_particles_local = 0;
  Real count_tally_local = 0.0;
  Real residence_tally_local = 0.0;

  const int num_partitions = pmesh->DefaultNumPartitions();
  for (int p = 0; p < num_partitions; ++p) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", p);
    auto *md = base.get();
    auto &resolved_pkgs = pmesh->resolved_packages;

    for (int b = 0; b < md->NumBlocks(); ++b) {
      active_particles_local +=
          md->GetSwarmData(b)->Get(alphas_swarm_name)->GetNumActive();
    }

    if (diagnostics) {
      static auto desc =
          MakePackDescriptor<fj::count_tally, fj::residence_tally>(resolved_pkgs.get());
      auto vmesh = desc.GetPack(md);

      const auto &ib = md->GetBoundsI(IndexDomain::interior);
      const auto &jb = md->GetBoundsJ(IndexDomain::interior);
      const auto &kb = md->GetBoundsK(IndexDomain::interior);
      const int nblocks = vmesh.GetNBlocks();

      Real count_tally_partition = 0.0;
      Real residence_tally_partition = 0.0;
      parthenon::par_reduce(
          DEFAULT_LOOP_PATTERN, "WriteWeakScalingDiagnostics::count-tallies",
          parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
          ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                        Real &count_sum, Real &residence_sum) {
            count_sum += vmesh(b, fj::count_tally(), k, j, i);
            residence_sum += vmesh(b, fj::residence_tally(), k, j, i);
          },
          Kokkos::Sum<Real>(count_tally_partition),
          Kokkos::Sum<Real>(residence_tally_partition));

      count_tally_local += count_tally_partition;
      residence_tally_local += residence_tally_partition;
    }
  }

  const Real transport_walltime_local =
      cpt_pkg->Param<Real>("transport_walltime_interval");
  const int meshblocks_local = pmesh->GetNumMeshBlocksThisRank();

  auto active_stats = GlobalMinMeanMax<long long>(active_particles_local);
  auto transport_stats = GlobalMinMeanMax<Real>(transport_walltime_local);
  auto meshblock_stats = GlobalMinMeanMax<int>(meshblocks_local);

  Real count_tally_total = count_tally_local;
  Real residence_tally_total = residence_tally_local;
#ifdef MPI_PARALLEL
  if (diagnostics) {
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &count_tally_total, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM,
                                      MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &residence_tally_total, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM,
                                      MPI_COMM_WORLD));
  }
#endif

  auto *diagnostic_output_count =
      cpt_pkg->MutableParam<int>("weak_scaling_diagnostics_output_count");
  const auto filename =
      RankLocalDiagnosticsFilename(cpt_pkg->Param<std::string>(
          "weak_scaling_diagnostics_basename"));
  std::ofstream ofs(filename,
                    (*diagnostic_output_count == 0) ? std::ios::out
                                                    : (std::ios::out | std::ios::app));
  PARTHENON_REQUIRE(ofs.is_open(),
                    "Could not open weak-scaling diagnostics output file.");

  if (*diagnostic_output_count == 0) {
    ofs << "ncycle,time,dt,"
        << "active_particles_local,active_particles_global_min,"
        << "active_particles_global_mean,active_particles_global_max,"
        << "transport_walltime_local,transport_walltime_global_min,"
        << "transport_walltime_global_mean,transport_walltime_global_max,"
        << "meshblocks_local,meshblocks_global_min,meshblocks_global_mean,"
        << "meshblocks_global_max";
    if (diagnostics) ofs << ",count_tally_total,residence_tally_total";
    ofs << '\n';
  }

  ofs << std::setprecision(17)
      << tm.ncycle << ',' << tm.time << ',' << tm.dt << ','
      << active_particles_local << ',' << active_stats.min << ','
      << active_stats.mean << ',' << active_stats.max << ','
      << transport_walltime_local << ',' << transport_stats.min << ','
      << transport_stats.mean << ',' << transport_stats.max << ','
      << meshblocks_local << ',' << meshblock_stats.min << ','
      << meshblock_stats.mean << ',' << meshblock_stats.max;
  if (diagnostics) ofs << ',' << count_tally_total << ',' << residence_tally_total;
  ofs << '\n';

  *diagnostic_output_count += 1;
  *cpt_pkg->MutableParam<Real>("transport_walltime_interval") = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn  void ResetTransportOutputTalliesIfOutputAdvanced
//! \brief Record output diagnostics without resetting cumulative tallies.
void ResetTransportOutputTalliesIfOutputAdvanced(Mesh *pmesh, ParameterInput *pin,
                                                 SimTime &tm) {
  auto &cpt_pkg = pmesh->packages.Get("charon");
  auto &out_pkg = pmesh->packages.Get("Outputs");

  int output_file_count = 0;
  for (int n = 0; n < 100; ++n) {
    const std::string outn = std::to_string(n);
    const std::string block = "parthenon/output" + outn;

    if (!pin->DoesParameterExist(block, "file_type")) continue;

    const std::string file_type = pin->GetString(block, "file_type");
    if (file_type == "hst" || file_type == "rst") continue;

    output_file_count += out_pkg->Param<int>(outn + "/file_number");
  }

  auto *last_reset_file_count =
      cpt_pkg->MutableParam<int>("transport_tally_reset_file_count");
  if (output_file_count > *last_reset_file_count) {
    if (tm.ncycle > 0) WriteWeakScalingDiagnostics(pmesh, pin, tm);
    // Persistent tallies are cumulative. Output must never clear physics or
    // diagnostic accumulators; mean stopping power uses a separate raw sum.
    *last_reset_file_count = output_file_count;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateDerivedTransportFieldsImpl
//! \brief Reset per-step transport tallies before sourcing and transport.
TaskStatus UpdateDerivedTransportFieldsImpl(
    MeshData<Real> *md, [[maybe_unused]] const Real dt) {
  namespace fj = field::charon;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);

  if (pm->packages.Get("charon")->Param<bool>("compact_tallies")) {
    auto desc = MakePackDescriptor<fj::deposited_energy_step>(resolved_pkgs.get());
    auto vmesh = desc.GetPack(md);
    if (vmesh.GetNBlocks() == 0) return TaskStatus::complete;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ZeroDepositedEnergyStep", DevExecSpace(),
        0, vmesh.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          vmesh(b, fj::deposited_energy_step(), k, j, i) = 0.0;
        });
    return TaskStatus::complete;
  }

  static auto desc =
      MakePackDescriptor<fj::energy_tally_step, fj::energy_delta_step,
                         fj::mass_delta_step,
                         fj::momentum_delta_step>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ZeroTransportTallies", parthenon::DevExecSpace(), 0,
                    md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          vmesh(b, fj::energy_tally_step(), k, j, i) = 0.0;
          vmesh(b, fj::energy_delta_step(), k, j, i) = 0.0;
          vmesh(b, fj::mass_delta_step(), k, j, i) = 0.0;
          for (int d = 0; d < 3; ++d) {
            vmesh(b, fj::momentum_delta_step(d), k, j, i) = 0.0;
          }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus UpdateDerivedTransportFieldsImpl
//! \brief  call update in IMC
TaskStatus UpdateDerivedTransportFields(MeshData<Real> *md, const Real dt) {
  return UpdateDerivedTransportFieldsImpl(md, dt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus TaskStatus DefragParticles(MeshBlock *pmb)
//! populate a minimum number of particles in each cell
TaskStatus DefragParticles(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &cpt_pkg = pm->packages.Get("charon");
  auto &min_swarm_occupancy = cpt_pkg->template Param<Real>("min_swarm_occupancy");
  const int nblocks = md->NumBlocks();

  for (int b = 0; b <= nblocks - 1; ++b) {
    auto &swarm = md->GetSwarmData(b)->Get(alphas_swarm_name);
    if (swarm->GetNumActive() > 0) {
      if (swarm->GetPackingEfficiency() < min_swarm_occupancy) {
        swarm->Defrag();
      }
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus EvaluateRadiationEnergy
//! \brief
template <typename T>
TaskStatus EvaluateRadiationEnergy(T *md) {
  return TaskStatus::complete;
  namespace fj = field::charon;
  namespace pa = particle::alphas;

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &cpt_pkg = pm->packages.Get("charon");

  // Create SparsePack
  static auto desc = MakePackDescriptor<fj::energy_tally,
                                        fj::energy_delta
                                        >(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r = MakeSwarmPackDescriptor<pa::weight>(alphas_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<pa::ijk>(alphas_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ZeroEnergyTally", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        vmesh(b, fj::energy_tally(), k, j, i) = 0.0;
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FillEnergyTally", DevExecSpace(), 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          auto &coords = vmesh.GetCoordinates(b);
          const int &ip = ppack_i(b, pa::ijk(0), n);
          const int &jp = ppack_i(b, pa::ijk(1), n);
          const int &kp = ppack_i(b, pa::ijk(2), n);
          const Real &dv = coords.CellVolume(kp, jp, ip);
          Kokkos::atomic_add(&vmesh(b, fj::energy_tally(), kp, jp, ip),
                             ppack_r(b, pa::weight(), n) / dv);
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void InitializeRadiation
//! \brief Initialize radiation based on material temperature and either thermal or
//!        zero initial radiation.
void InitializeRadiation(MeshBlockData<Real> *mbd) {
  auto *pmb = mbd->GetBlockPointer();
  auto &cpt_pkg = mbd->GetBlockPointer()->packages.Get("charon");
  charon::SourceAlphas(mbd, 0.0, 0.0);

  // Call this so radiation field variables are properly initialized for outputs
  charon::EvaluateRadiationEnergy<MeshBlockData<Real>>(mbd);
}

//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus EvaluateRadiationEnergy<MeshBlockData<Real>>(MeshBlockData<Real> *md);

} // namespace charon
