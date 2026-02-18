# File: cli.py
# CLI entry point for generating the parametric profiling signal
# Usage: python -m nam.signals.cli --output profiling_signal.wav

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate NAM parametric amp profiling signal"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output WAV file path (24-bit, 48kHz, mono)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    from .profiling_signal import ProfilingSignalGenerator

    generator = ProfilingSignalGenerator(seed=args.seed)
    print(f"Generating profiling signal ({generator.total_duration:.1f}s)...")
    generator.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
