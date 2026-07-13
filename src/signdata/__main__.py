"""Entry point: python -m signdata run|experiment <config.yaml>"""

import logging
import sys

# Ensure registrations happen on import
import signdata.datasets  # noqa: F401
import signdata.processors  # noqa: F401

from signdata.cli import parse_args
from signdata.config.loader import load_config
from signdata.pipeline.runner import PipelineRunner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    if args.command is None:
        print("Usage: python -m signdata <command> <config.yaml> [options]")
        print()
        print("Commands:")
        print("  run         Run a single preprocessing job")
        print("  experiment  Run a multi-job experiment")
        print()
        print("Run 'python -m signdata <command> --help' for details.")
        sys.exit(1)

    if args.command == "run":
        # Handle --list-presets early exit (no config file needed)
        if args.list_presets:
            from signdata.processors.pose import list_presets
            for name, desc in sorted(list_presets().items()):
                print(f"  {name:30s} {desc}")
            return

        if not args.config:
            print("Error: config file is required (unless using --list-presets)")
            sys.exit(1)

        overrides = args.override or []

        if args.run_name:
            overrides.append(f"run_name={args.run_name}")

        config = load_config(args.config, overrides=overrides or None)

        runner = PipelineRunner(
            config,
            force_all=args.force_all,
        )
        runner.run()

    elif args.command == "experiment":
        from signdata.config.experiment import load_experiment
        from signdata.pipeline.experiment import ExperimentRunner

        experiment = load_experiment(args.config)
        runner = ExperimentRunner(
            experiment, force_all=args.force_all,
        )
        results = runner.run()

        # Exit with error code if any job failed
        if any(result.status == "failed" for result in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
