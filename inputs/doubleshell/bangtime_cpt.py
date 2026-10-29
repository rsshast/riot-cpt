#!/usr/bin/env python3
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
# ========================================================================================
"""Generate the reduced double-shell bang-time hydro/CPT problem.

The initial state is a spherical reduction of the 10.0025 ns IdealGas double-shell
output.  The DT R99 mass radius is 0.02095 cm at that time.  Density and temperature
profiles below are compact, 10-micron radial averages of the uploaded output; the
velocity profile is inferred from motion of fixed DT-mass surfaces between the 10.0,
10.2, and 10.4 ns outputs because velocity was not written to those files.

The pusher is an initial condition, not a maintained boundary.  The calculation uses a
uniform Cartesian mesh and only the hydro and optional CPT physics packages.
"""

import argparse

import numpy as np


GAMMA = 5.0 / 3.0
CV = 1.0e8

# Geometry and timing.
FUEL_RADIUS = 0.02095
PROFILE_RADIUS = 0.035
DOMAIN_RADIUS = 0.040
FINAL_TIME = 2.0e-9
TIMESTEP = 2.0e-12
SOURCE_TEMPERATURE_THRESHOLD = 1.11e7
OUTPUT_DT = 2.0e-11

# FillInteriorDerived only sources CPT histories where rho > 100 g/cc and
# T > 1.16045e7 K.  The uploaded 10 ns state is below that joint threshold.  A
# deliberately over-pressured source seed remains available for CPT plumbing tests,
# but it is disabled by default because it drives an explosion rather than an implosion.
SOURCE_SEED_RADIUS = 0.0
SOURCE_SEED_DENSITY = 101.0
SOURCE_SEED_TEMPERATURE = 1.0e6

# Initial outer-pusher state.  With the chosen IdealGas EOS this gives T=1.282e5 K.
PUSHER_PRESSURE = 1.0e14
PUSHER_DENSITY = 11.7

# Radial averages of the DT-bearing cells at 10.0025 ns.  The cold inner fuel is
# uniform; the closely spaced outer anchors retain the incoming compressed layer.
FUEL_R = np.array(
    [0.0, 0.0135, 0.0165, 0.0175, 0.0185, 0.0195, 0.0205, FUEL_RADIUS]
)
FUEL_RHO = np.array(
    [0.2, 0.2, 0.2000083, 0.2057873, 0.4447740, 5.514811, 33.60257, 45.0]
)
FUEL_T = np.array(
    [37.5, 37.5, 37.58934, 1.583805e3, 3.416783e4, 5.902909e4,
     9.201909e4, 1.30e5]
)

# Bulk radial averages outside the DT R99 surface.  They preserve the already-compressed
# inner pusher, then join smoothly to the requested 1e14 erg/cc outer-pusher state.
PUSHER_R = np.array([FUEL_RADIUS, 0.0225, 0.0255, 0.0305, 0.0345, PROFILE_RADIUS])
PUSHER_RHO = np.array([40.0, 16.69499, 11.66766, 17.21504, 11.72037, PUSHER_DENSITY])
PUSHER_T = np.array([1.10e5, 1.069596e6, 1.327379e6, 1.449462e5,
                     1.334525e5, PUSHER_PRESSURE / ((GAMMA - 1.0) * PUSHER_DENSITY * CV)])

# Inferred initial radial velocity [cm/s].  The inner 25% of DT mass is stationary at
# 10 ns while the outer mass surfaces move inward at roughly 40--60 km/s.  The pusher
# velocity is tapered to zero in the outer padding to avoid a discontinuity at the box.
VELOCITY_R = np.array(
    [0.0, 0.0135, 0.0170514, 0.0192029, 0.0200686, 0.0204145,
     FUEL_RADIUS, 0.0215581, 0.025, 0.030, PROFILE_RADIUS, DOMAIN_RADIUS]
)
VELOCITY_UR = np.array(
    [0.0, 0.0, -8.817992e5, -4.793657e6, -5.887468e6, -5.523071e6,
     -4.376330e6, -3.288234e6, -5.0e6, -3.0e6, 0.0, 0.0]
)


def _positive_float(value):
    value = float(value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_float(value):
    value = float(value)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _case_tag(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stopping-power", type=_positive_float, default=1.0e-3,
                        help="Constant CPT stopping power [erg/cm]")
    parser.add_argument("--alpha-weight", type=_positive_float, default=1.0e10)
    parser.add_argument(
        "--source-temperature-threshold", type=_positive_float,
        default=SOURCE_TEMPERATURE_THRESHOLD,
        help="CPT source temperature cutoff [K]; cells must also have rho > 100 g/cc",
    )
    parser.add_argument("--no-cpt", action="store_true",
                        help="Generate the hydro-only comparison deck")
    parser.add_argument("--nx", type=_positive_int, default=288,
                        help="Uniform cells in each Cartesian direction")
    parser.add_argument("--meshblock-nx", type=_positive_int, default=36,
                        help="Cells per meshblock direction")
    parser.add_argument("--domain-radius", type=_positive_float, default=DOMAIN_RADIUS,
                        help="Cartesian half-width [cm]")
    parser.add_argument("--tlim", type=_positive_float, default=FINAL_TIME,
                        help="Duration of the reduced calculation [s]")
    parser.add_argument("--dt", type=_positive_float, default=TIMESTEP,
                        help="Fixed hydro/CPT timestep [s]")
    parser.add_argument("--output-dt", type=_positive_float, default=OUTPUT_DT,
                        help="HDF5 output interval [s]")
    parser.add_argument(
        "--source-seed-radius",
        type=_nonnegative_float,
        default=SOURCE_SEED_RADIUS,
        help=(
            "Radius of an optional over-pressured CPT diagnostic seed [cm]; "
            "zero disables it"
        ),
    )
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--output", default=None,
                        help="Generated .rin filename (default: <problem-id>.rin)")
    return parser


def resolve_args(args):
    if args.nx % args.meshblock_nx:
        raise SystemExit("--nx must be divisible by --meshblock-nx")
    if args.domain_radius <= PROFILE_RADIUS:
        raise SystemExit("--domain-radius must exceed the 0.035 cm profile radius")
    if args.source_seed_radius > 0.0:
        cell_width = 2.0 * args.domain_radius / args.nx
        if args.source_seed_radius <= np.sqrt(3.0) * 0.5 * cell_width:
            raise SystemExit(
                "--source-seed-radius is too small to contain a cell center on this grid"
            )
        if args.source_seed_radius >= FUEL_RADIUS:
            raise SystemExit("--source-seed-radius must be smaller than the fuel radius")
    if args.dt > args.tlim:
        raise SystemExit("--dt cannot exceed --tlim")
    if args.problem_id is None:
        args.problem_id = (
            "bangtime_no_cpt"
            if args.no_cpt
            else f"bangtime_cpt_C{_case_tag(args.stopping_power)}"
        )
    return args


def make_input(args):
    import riot

    do_cpt = not args.no_cpt
    riot.input(
        "riot",
        problem="region_pgen",
        sparse_init=False,
        sparse_dealloc=True,
        verbose=False,
        use_mpi_shared_memory=True,
    )
    riot.input("comment", problem="Reduced double-shell bang-time CPT study")
    riot.input("parthenon/job", problem_id=args.problem_id)

    output_variables = [
        "c.c.bulk.rho",
        "c.c.bulk.pressure",
        "c.c.bulk.temperature",
        "c.c.bulk.velocity",
        "c.c.mat.rho",
        "c.c.mat.volume_fraction",
    ]
    if do_cpt:
        output_variables += [
            #"c.c.bulk.cpt_alpha_source",
            #"field.charon.source_num_per_cell",
            #"field.charon.active_num_per_cell",
            #"field.charon.active_ke_tally",
            "field.charon.energy_tally",
            #"field.charon.energy_delta",
            #"field.charon.ave_energy",
            #"field.charon.spower_tally",
        ]
    riot.input("parthenon/output0", file_type="hst", dt=args.dt)
    riot.input(
        "parthenon/output1",
        variables=output_variables,
        file_type="hdf5",
        dt=args.output_dt,
        sparse_seed_nans=True,
    )
    riot.input("parthenon/output2", file_type="rst", dt=args.tlim)
    riot.input(
        "parthenon/time",
        #nlim=500,
        nlim=-1,
        tlim=args.tlim,
        integrator="rk1",
        #dt_force=args.dt,
        #dt_init=1e-12,
        dt_init=args.dt,
        #dt_max=args.dt,
        #dt_factor=1.01,
        ncycle_out=1,
    )

    mesh = dict(
        refinement="none",
        numlevel=1,
        nghost=2,
        nx1=args.nx,
        x1min=-args.domain_radius,
        x1max=args.domain_radius,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=args.nx,
        x2min=-args.domain_radius,
        x2max=args.domain_radius,
        ix2_bc="outflow",
        ox2_bc="outflow",
        nx3=args.nx,
        x3min=-args.domain_radius,
        x3max=args.domain_radius,
        ix3_bc="outflow",
        ox3_bc="outflow",
    )
    riot.input("parthenon/mesh", **mesh)
    riot.input(
        "parthenon/meshblock",
        nx1=args.meshblock_nx,
        nx2=args.meshblock_nx,
        nx3=args.meshblock_nx,
    )

    riot.input("materials", sparse_init=False, sparse_dealloc=True)
    for matid, label in ((0, "fuel"), (1, "pusher")):
        eos_name = f"{label.capitalize()}EOS"
        riot.input(
            f"material{matid}",
            label=label,
            eos=eos_name,
            max_bnd_level=0,
            max_mat_level=0,
        )
        riot.input(eos_name, eos_type="IdealGas", Gamma=GAMMA, Cv=CV)

    riot.input("regions", nlev_min=0, nlev_max=0)
    riot.input(
        "region0",
        name="PusherAmbient",
        mask_type="background",
        matid=1,
        c_m_rho=PUSHER_DENSITY,
        c_m_pressure=PUSHER_PRESSURE,
    )
    riot.input(
        "region1",
        name="PusherProfile",
        mask_type="inside_spherical_shell",
        inner_radius=FUEL_RADIUS,
        outer_radius=PROFILE_RADIUS,
        matid=1,
    )
    riot.input(
        "region2",
        name="FuelProfile",
        mask_type="inside_sphere",
        radius=FUEL_RADIUS,
        matid=0,
    )
    if args.source_seed_radius > 0.0:
        # Diagnostic only: this region is intentionally above both hard-coded CPT
        # thresholds and is not pressure matched to the physical implosion profile.
        riot.input(
            "region3",
            name="IgnitionSeed",
            mask_type="inside_sphere",
            radius=args.source_seed_radius,
            matid=0,
            c_m_rho=SOURCE_SEED_DENSITY,
            c_m_temperature=SOURCE_SEED_TEMPERATURE,
        )

    riot.input(
        "physics",
        hydro=True,
        cpt=do_cpt,
        sparse_physics=False,
        fixed_fluid=False,
    )
    riot.input(
        "hydro",
        recon="plm",
        riemann="chllc",
        cfl=0.3,
        amr_interface=False,
        mass_frac_thresh=1.0e-8,
    )

    if do_cpt:
        riot.input(
            "charon",
            source_strategy="qfield",
            alpha_source_mode="total",
            source_temperature_threshold=args.source_temperature_threshold,
            num_particles=0,
            num_particles_per_pulse=0,
            pulses=0,
            source_time_pulses=1,
            stopping_power_model="constant",
            stopping_power_value=args.stopping_power,
            alpha_weight=args.alpha_weight,
            alpha_source_physical_cap=1.0e16,
            do_feedback=True,
            do_population_control=True,
            #do_population_control=False,
            weak_scaling_diagnostics=False,
            verbose=False,
            max_transport_iterations=10000,
            seed=42,
        )


def _radius(pos, obj):
    return np.sqrt(
        pos[:, obj.x] ** 2 + pos[:, obj.y] ** 2 + pos[:, obj.z] ** 2
    )


def _set_radial_velocity(pos, vel, obj):
    radius = _radius(pos, obj)
    ur = np.interp(radius, VELOCITY_R, VELOCITY_UR)
    nonzero = radius > 0.0
    vel[:] = 0.0
    vel[nonzero, obj.x] = ur[nonzero] * pos[nonzero, obj.x] / radius[nonzero]
    vel[nonzero, obj.y] = ur[nonzero] * pos[nonzero, obj.y] / radius[nonzero]
    vel[nonzero, obj.z] = ur[nonzero] * pos[nonzero, obj.z] / radius[nonzero]


class PusherAmbient:
    def __init__(self):
        pass

    def c_c_bulk_velocity(self, pos, vel):
        _set_radial_velocity(pos, vel, self)


class PusherProfile:
    def __init__(self):
        pass

    def c_m_rho(self, pos, rho):
        rho[:] = np.interp(_radius(pos, self), PUSHER_R, PUSHER_RHO)

    def c_m_temperature(self, pos, temperature):
        temperature[:] = np.interp(_radius(pos, self), PUSHER_R, PUSHER_T)

    def c_c_bulk_velocity(self, pos, vel):
        _set_radial_velocity(pos, vel, self)


class FuelProfile:
    def __init__(self):
        pass

    def c_m_rho(self, pos, rho):
        rho[:] = np.interp(_radius(pos, self), FUEL_R, FUEL_RHO)

    def c_m_temperature(self, pos, temperature):
        temperature[:] = np.interp(_radius(pos, self), FUEL_R, FUEL_T)

    def c_c_bulk_velocity(self, pos, vel):
        _set_radial_velocity(pos, vel, self)


class IgnitionSeed:
    def __init__(self):
        pass

    def c_c_bulk_velocity(self, pos, vel):
        _set_radial_velocity(pos, vel, self)


if __name__ == "__main__":
    import riot

    arguments = resolve_args(build_parser().parse_args())
    make_input(arguments)
    output_name = arguments.output or f"{arguments.problem_id}.rin"
    riot.input.generate_input(output_name)
