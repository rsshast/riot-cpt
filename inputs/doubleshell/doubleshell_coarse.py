#!/usr/bin/env python3
"""One-node smoke-test configuration for the double-shell CPT problem.

Defaults to a 72^3 Cartesian base grid (64 meshblocks), no AMR refinement,
0.5 ns of simulated time, and one HDF5 output interval. The coarse settings are
applied directly so this wrapper also works with an older doubleshell_cpt.py parser.
"""

import sys

import doubleshell_cpt


BASE_NX1 = 36
MESHBLOCK_NX = 18
TLIM = 1.0e-8
OUTPUT_DT = 5.0e-10
STOPPING_POWER = 1.0e-5
SMOKE_GAMMA = 5.0 / 3.0
SMOKE_CV = 1.0e8


class CoarseInput:
    """Forward RIOT input calls after applying the smoke-test overrides."""

    def __init__(self, base):
        self.base = base

    def __call__(self, section, **kwargs):
        if section == "parthenon/job":
            kwargs["problem_id"] = "doubleshell_coarse"
        elif section == "parthenon/output1":
            kwargs["dt"] = OUTPUT_DT
        elif section == "parthenon/time" and "tlim" in kwargs:
            kwargs["tlim"] = TLIM
        elif section == "parthenon/mesh":
            kwargs.update(nx1=2 * BASE_NX1, nx2=2 * BASE_NX1, nx3=2 * BASE_NX1)
            kwargs["numlevel"] = 1
        elif section == "parthenon/meshblock":
            kwargs.update(nx1=MESHBLOCK_NX, nx2=MESHBLOCK_NX, nx3=MESHBLOCK_NX)
        elif section == "physics":
            kwargs.update(cpt=True, tn=False)
        elif section == "charon":
            kwargs.update(
                stopping_power_model="constant",
                stopping_power_value=STOPPING_POWER,
            )
        elif kwargs.get("eos_type") == "SpinerEOSDependsRhoT":
            # The tabular materials.sp5 file is not distributed with RIOT. Ideal-gas
            # surrogates keep this one-node deck self-contained as a plumbing smoke test.
            kwargs = {
                "eos_type": "IdealGas",
                "Gamma": SMOKE_GAMMA,
                "Cv": SMOKE_CV,
            }
        return self.base(section, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


def main():
    defaults = [
        "--dim",
        "3",
        "--rad",
        "diffusion",
        "--do_cpt",
        "--nlevels",
        "1",
        "--no_gap",
    ]
    args = doubleshell_cpt.resolve_args(
        doubleshell_cpt.build_parser().parse_args(defaults + sys.argv[1:])
    )
    # Newer versions of doubleshell_cpt.py consume these attributes directly. The
    # CoarseInput adapter below applies the same settings for older versions.
    args.base_nx1 = BASE_NX1
    args.meshblock_nx = MESHBLOCK_NX
    args.tlim = TLIM
    args.output_dt = OUTPUT_DT
    args.cpt_stopping_power = STOPPING_POWER
    args.problem_id = "doubleshell_coarse"

    import riot

    riot.input = CoarseInput(riot.input)
    doubleshell_cpt.make_input(args)
    riot.input.generate_input()


if __name__ == "__main__":
    main()
