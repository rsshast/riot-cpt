#!/usr/bin/env python3
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
# ========================================================================================
# This file was made in part with generative AI.

import math

import riot


# timing
NANOSECOND = 1.0e-9
PICOSECOND = 1.0e-12
FINAL_TIME = 2.0 * NANOSECOND           # tf
TIMESTEP = 200.0 * PICOSECOND             # dt CPT
OUTPUT_DT = 100.0 * PICOSECOND          # phdf output frequency

# sourcing
PULSES = 1000                      # distribution of particles post t0
N0 = 10000                           # initial population
NPP = 100                                # num particles per pulse
PARTICLE_WEIGHT = 1.0e10                # weight per MC particle
DO_SINE = True
if DO_SINE: SINE_AMPLITUDE = 10000

# material
RHO0 = 1.0                              # density, g/cc
T0 = 293.0                              # initial temperature, K
CV = 1.0e8                              # specific heat

# CPT
STOPPING_POWER_VALUE = 1.0e-5           # stopping power value, erg/cm


# make input
def make_input():
    riot.input("riot", problem="cpt_0d")

    riot.input("parthenon/job", problem_id="cpt_0d")

    riot.input(
        "parthenon/output1",
        variables=[
            "c.c.bulk.total_material_energy",
            #"c.c.bulk.internal_energy",
            "c.c.bulk.temperature",
            "c.c.bulk.rho",
            "c.c.mat.rho",
            "c.c.bulk.cpt_alpha_source",
            "field.charon.active_ke_tally",
            "field.charon.source_num_per_cell",
            "field.charon.active_num_per_cell",
            "field.charon.energy_tally",
            "field.charon.energy_delta",
            "field.charon.mass_delta",
            "field.charon.ave_energy",
            #"field.charon.Phi",
            "field.charon.spower_tally",
        ],
        file_type="hdf5",
        dt=OUTPUT_DT,
    )

    riot.input(
        "parthenon/time",
        nlim=-1,
        #nlim=SOURCE_PULSES,
        tlim=FINAL_TIME,
        integrator="rk1",
        dt_force=TIMESTEP,
        ncycle_out=1,
    )

    riot.input(
        "parthenon/mesh",
        nghost=2,
        refinement="none",
        numlevel=1,
        nx1=189,
        x1min=-0.6,
        x1max=0.6,
        ix1_bc="outflow",
        ox1_bc="outflow",
        nx2=189,
        x2min=-0.6,
        x2max=0.6,
        ix2_bc="outflow",
        ox2_bc="outflow",
        nx3=189,
        x3min=-0.6,
        x3max=0.6,
        ix3_bc="outflow",
        ox3_bc="outflow",
    )

    riot.input("parthenon/meshblock", 
                    nx1=27, 
                    nx2=27, 
                    nx3=27)

    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=CV)

    riot.input(
        "physics",
        hydro=True,
        cpt=True,
        sparse_physics=False,
        freeze_hydro=True,
    )

    riot.input("hydro", recon="plm", cfl=0.3)

    riot.input("problem", density=RHO0, temperature=T0)

    riot.input(
        "charon",
        source_strategy="point",
        num_particles=N0,
        variable_field="sine_test" if DO_SINE else "constant",
        num_particles_per_pulse=SINE_AMPLITUDE if DO_SINE else NPP,
        pulses=PULSES,
        source_time_pulses=1,
        stopping_power_model="constant",
        stopping_power_value=1.0e-5,
        alpha_weight=PARTICLE_WEIGHT,
        dt=TIMESTEP,
        do_feedback=True,
        do_population_control=False,
        weak_scaling_diagnostics=False,
        verbose=False,
        max_transport_iterations=10000,
        seed=42,
    )


if __name__ == "__main__":
    make_input()
    riot.input.generate_input()
