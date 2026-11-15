from pathlib import Path
import pytimeloop.timeloopfe.v4 as tl


ROOT = Path(__file__).resolve().parents[1]

ARCH_PATH = ROOT / "timeloop/architectures/reference_chiplet.yaml"
COMPONENTS_PATH = ROOT / "timeloop/components/components.yaml"
MAPPER_PATH = ROOT / "timeloop/mappings/mapper.yaml"
VARIABLES_PATH = ROOT / "timeloop/variables.yaml"

PROBLEM_DIR = ROOT / "timeloop/problems/resnet50"
OUTPUT_DIR = ROOT / "timeloop/outputs/resnet50"


def run_layer(problem_file):

    problem_path = PROBLEM_DIR / problem_file
    output_path = OUTPUT_DIR / problem_path.stem
    stats_file = output_path / "timeloop-mapper.stats.txt"

    # ---------------------------------------------------------
    # Skip already completed layers
    # ---------------------------------------------------------

    if stats_file.exists():
        print(
            f"[SKIP] {problem_file} "
            f"(existing stats found)"
        )
        return "skipped"

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print(f"[RUN] {problem_file}")
    print(f"Output: {output_path}")
    print("=" * 70)

    try:

        spec = tl.Specification.from_yaml_files(
            str(ARCH_PATH),
            str(COMPONENTS_PATH),
            str(MAPPER_PATH),
            str(problem_path),
            str(VARIABLES_PATH),
        )

        tl.call_mapper(
            spec,
            output_dir=str(output_path),
        )

        # Verify successful output
        if stats_file.exists():
            print(
                f"[DONE] {problem_file}"
            )
            return "completed"

        print(
            f"[WARNING] {problem_file} "
            f"finished but stats file was not found."
        )

        return "failed"

    except Exception as e:

        print(
            f"[ERROR] {problem_file}: {e}"
        )

        return "failed"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    problem_files = sorted(
        PROBLEM_DIR.glob("*.yaml")
    )

    print("=" * 70)
    print("P04 — ResNet-50 Timeloop Batch Run")
    print("=" * 70)

    print(
        f"Problem files found: {len(problem_files)}"
    )

    if len(problem_files) != 53:
        print(
            "WARNING: Expected 53 problem files."
        )

    completed = 0
    skipped = 0
    failed = 0

    for index, problem_path in enumerate(
        problem_files,
        start=1,
    ):

        print(
            f"\nProgress: "
            f"{index}/{len(problem_files)}"
        )

        status = run_layer(
            problem_path.name
        )

        if status == "completed":
            completed += 1

        elif status == "skipped":
            skipped += 1

        else:
            failed += 1

    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)

    print(f"Completed : {completed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Total     : {len(problem_files)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
