"""Command-line workflows. Run ``python -m lmc_auriga --help``."""

import argparse
import json
from pathlib import Path

from .analysis import analyze_snapshot
from .benchmark import run_benchmark
from .io import read_metadata
from .synthetic import write_synthetic


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validated halo kinematics and reproducible CPU benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate a labeled toy snapshot and analyze it")
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--particles", type=int, default=20000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--figures", action="store_true")
    analyze = commands.add_parser("analyze", help="Analyze an explicitly documented schema-v1 snapshot")
    analyze.add_argument("snapshot", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--shell", type=float, nargs=2, default=(6., 10.), metavar=("INNER", "OUTER"))
    analyze.add_argument("--cone-axis", type=float, nargs=3)
    analyze.add_argument("--half-angle", type=float, default=45.)
    analyze.add_argument("--vc", type=float, default=220.)
    analyze.add_argument("--basis", type=float, nargs=9, help="Row-major local-to-simulation orthonormal basis")
    analyze.add_argument("--phases", type=int, default=12)
    analyze.add_argument("--chunk-size", type=int, default=100000)
    analyze.add_argument("--speed-max", type=float, default=1000.)
    analyze.add_argument("--bin-width", type=float, default=20.)
    analyze.add_argument("--backend", choices=("numpy", "numba", "python"), default="numpy")
    analyze.add_argument("--figures", action="store_true")
    inspect = commands.add_parser("inspect", help="Validate and print processed snapshot metadata")
    inspect.add_argument("snapshot", type=Path)
    legacy = commands.add_parser("import-legacy", help="Import processed legacy arrays after explicit frame confirmation")
    legacy.add_argument("source", type=Path)
    legacy.add_argument("destination", type=Path)
    legacy.add_argument("--provenance", required=True, help="Source and preprocessing evidence for this dataset")
    legacy.add_argument("--confirm-processed-frame", action="store_true",
                        help="Assert positions are host-centered physical kpc and velocities host-rest-frame km/s")
    figures = commands.add_parser("figures", help="Render existing analysis tables")
    figures.add_argument("directory", type=Path)
    benchmark = commands.add_parser("benchmark", help="Measure validated observer-speed kernels")
    benchmark.add_argument("--output", type=Path, default=Path("results/benchmark"))
    benchmark.add_argument("--counts", type=int, nargs="+", default=[1000, 10000, 100000])
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--backends", nargs="+", choices=("numpy", "numba", "python"), default=["numpy", "numba"])
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            snapshot = args.output / "synthetic.hdf5"
            write_synthetic(snapshot, n_particles=args.particles, seed=args.seed)
            report = analyze_snapshot(snapshot, args.output)
            print(f"Synthetic demonstration: {report['selected_count']} selected particles; {args.output}")
        elif args.command == "analyze":
            import numpy as np
            basis = None if args.basis is None else np.asarray(args.basis).reshape(3, 3)
            report = analyze_snapshot(args.snapshot, args.output, shell=args.shell, cone_axis=args.cone_axis,
                half_angle_deg=args.half_angle, vc=args.vc, basis=basis, phase_count=args.phases,
                chunk_size=args.chunk_size, speed_max=args.speed_max, bin_width=args.bin_width, backend=args.backend)
            print(f"Analyzed {report['selected_count']} selected particles; {args.output}")
        elif args.command == "inspect":
            print(json.dumps(read_metadata(args.snapshot), indent=2, allow_nan=False))
        elif args.command == "import-legacy":
            from .importing import import_legacy_snapshot
            metadata = import_legacy_snapshot(args.source, args.destination, provenance=args.provenance,
                confirmed_processed_frame=args.confirm_processed_frame)
            print(json.dumps(metadata, indent=2, allow_nan=False))
        elif args.command == "benchmark":
            result = run_benchmark(args.output, counts=args.counts, repeats=args.repeats, seed=args.seed, backends=args.backends)
            print(json.dumps({"status": result["metadata"]["status"],
                "skipped_backends": result["metadata"]["skipped_backends"],
                "summary": result["summary"], "paths": result["paths"]}, indent=2))
            if result["metadata"]["status"] == "no_available_backends":
                return 2
        if args.command == "figures" or getattr(args, "figures", False):
            from .plotting import make_figures
            for path in make_figures(args.directory if args.command == "figures" else args.output):
                print(path)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0
