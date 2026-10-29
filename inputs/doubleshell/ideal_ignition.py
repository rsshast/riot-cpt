#!/usr/bin/env python3
# (C) 2026. Triad National Security, LLC. All rights reserved.
# This file was made in part with generative AI.
"""TN source-only/LP ignition study initialized from the reduced bangtime profiles.

Generate DT data separately (never overwritten here):
  python script/dt_reactions.py 1 30 -n 181 -o dt_isotope_data.hdf5
Run with the same PYTHONPATH/runtime setup as bangtime_cpt.py. All times are seconds.
The tungsten surrogate is an approximation to the collapsed outer material stack.
"""
from pathlib import Path
import argparse
import numpy as np
import bangtime_cpt as bangtime

AMU_G = 1.660538921e-24
KB = 1.3806488e-16

def build_parser():
    parser = bangtime.build_parser()
    parser.description = __doc__
    parser.set_defaults(tlim=4.0e-9, alpha_weight=1.0e8)
    parser.add_argument("--tn-data", type=Path, default=Path("dt_isotope_data.hdf5"))
    parser.add_argument("--stopping-model", choices=["LiPetrasso", "constant"], default="LiPetrasso")
    parser.add_argument("--pusher-mass", type=bangtime._positive_float, default=184.0)
    parser.add_argument("--pusher-z", type=bangtime._positive_float, default=74.0)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--electron-ion-coupling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tn-upper-temperature", choices=["error", "clamp"], default="error")
    parser.add_argument("--constant-ion-fraction", type=float, default=0.5)
    parser.add_argument("--plasma-velocity-tolerance", type=bangtime._positive_float, default=1.e-4)
    return parser


def read_fuel_metadata(path):
    """Validate the PR #40 code-unit tables without rewriting them."""
    import h5py
    with h5py.File(path, "r") as f:
        r = f["reactions/d+t->n+a"]
        t = r["Temperatures/data"][:]
        sigma = r["SigmaVBar/data"][:]
        if len(t) < 2 or t.shape != sigma.shape or not np.all(np.isfinite(t)):
            raise ValueError("Invalid DT table shape/temperature values")
        if not np.all(t > 0) or not np.all(np.diff(t) > 0):
            raise ValueError("DT temperatures must be positive and increasing [K]")
        if not np.allclose(np.diff(np.log(t)), np.diff(np.log(t))[0], rtol=1.e-6):
            raise ValueError("DT table must be uniformly spaced in ln(T/K)")
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Invalid mass-normalized DT reactivity")
        if t[-1] < 1.602176565e-9 / KB:
            raise ValueError("DT table does not reach the generator's 1 keV cutoff")
        metadata = dict(zip(f["Masses"].attrs["zaids"], f["Masses"].attrs["masses"]))
        return float(metadata[1002]), float(metadata[1003])


def make_input(args):
    import riot
    if args.no_cpt or args.source_seed_radius != 0:
        raise ValueError("This study requires CPT and disables the diagnostic source seed")
    if not 0 <= args.constant_ion_fraction <= 1:
        raise ValueError("--constant-ion-fraction must be in [0,1]")
    if args.pusher_z > args.pusher_mass:
        raise ValueError("Pusher nuclear charge cannot exceed its mass number")
    if args.problem_id is None:
        args.problem_id = "ideal_ignition_" + args.stopping_model
    args = bangtime.resolve_args(args)
    data = args.tn_data.expanduser().resolve()
    md, mt = read_fuel_metadata(data)
    # Reuse the existing geometry, Python region callbacks, mesh and velocity.
    bangtime.make_input(args)
    riot.input("comment", problem="Ideal-gas DT source-only LP ignition study")
    riot.input("physics", tn=True, ionization=True)
    riot.input("isotope_data", filename=str(data))
    riot.input("tnburn", reaction0="d+t->n+a", source_only=True,
               deposit_locally_1=False, deposit_locally_2004=False)
    riot.input("material0", isotope0=1002, isotope1=1003, isotope2=2004,
               isotope0_mfrac=md/(md+mt), isotope1_mfrac=mt/(md+mt),
               isotope2_mfrac=0.0, electron_eos="FuelElectrons")
    riot.input("material1", electron_eos="PusherElectrons")
    # Monatomic ions and ideal electrons. IdealElectrons supplies zbar-dependent
    # heat capacity; the ion IdealGas contains ION heat capacity only.
    fuel_mass = 0.5*(md+mt)
    for label, mass, z in (("Fuel", fuel_mass, 1.0),
                            ("Pusher", args.pusher_mass, args.pusher_z)):
        riot.input(label+"EOS", eos_type="IdealGas", Gamma=5.0/3.0,
                   Cv=1.5*KB/(mass*AMU_G), mean_atomic_mass=mass,
                   mean_atomic_number=z, zsplit=False)
        riot.input(label+"Electrons", eos_type="IdealElectrons",
                   mean_atomic_mass=mass, mean_atomic_number=z)
    # Keep the original ambient temperature; do not pressure-match the old EOS.
    riot.input.blocks["region0"].pop("c_m_pressure", None)
    riot.input("region0", c_m_temperature=float(bangtime.PUSHER_T[-1]))
    riot.input("ionization", fully_ionized=False, fully_ionize_hydrogen=True,
               electron_ion_coupling=args.electron_ion_coupling,
               electron_ion_coupling_model="landau_spitzer",
               electron_thermal_conduction=False, ion_thermal_conduction=False,
               advect_electron_entropy=False, plasma_viscosity=False)
    riot.input("charon", tn_plasma=True, stopping_power_model=args.stopping_model,
               alpha_weight=args.alpha_weight,
               thermalization_mev=0.2, tn_upper_temperature=args.tn_upper_temperature,
               constant_ion_fraction=args.constant_ion_fraction,
               plasma_velocity_tolerance=args.plasma_velocity_tolerance,
               do_feedback=args.feedback, tally_mass_momentum=False,
               do_population_control=False, cpt_diagnostics=args.diagnostics)
    riot.input.blocks["charon"].pop("source_temperature_threshold", None)
    riot.input.blocks["charon"].pop("alpha_source_physical_cap", None)
    fields = ["c.c.bulk.rho", "c.c.bulk.pressure", "c.c.bulk.temperature",
              "c.c.bulk.electron_temperature", "c.c.bulk.electron_number_density",
              "c.c.bulk.velocity", "c.c.mat.rho", "c.c.mat.iso", "c.m.ionization_zbar",
              "c.c.mat.volume_fraction", "c.c.bulk.cpt_alpha_source"]
    fields += ["field.charon."+name for name in (
        "physical_alphas", "sampled_alphas", "pending_alphas", "electron_heat",
        "ion_heat", "escaped_alphas", "escaped_energy")]
    if args.diagnostics:
        fields += ["field.charon."+name for name in (
            "source_num_per_cell", "active_num_per_cell", "active_ke_tally",
            "energy_tally", "energy_delta", "spower_tally")]
    riot.input("parthenon/output1", variables=fields)
    # Frequent checkpoints allow source-remainder/census restart verification.
    riot.input("parthenon/output2", dt=2.0e-10)
    #riot.input("parthenon/output2", dt=min(args.output_dt, args.tlim))
    return args


if __name__ == "__main__":
    import riot
    args = make_input(build_parser().parse_args())
    riot.input.generate_input(args.output or args.problem_id+".rin")
