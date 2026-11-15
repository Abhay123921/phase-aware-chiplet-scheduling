from pathlib import Path
import sys

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

    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Running: {problem_file}")
    print(f"Output : {output_path}")
    print("=" * 70)

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

    print(f"\nCompleted: {problem_file}")


def main():
    layers = [
        "01_conv1.yaml",
        "03_layer1_0_conv2.yaml",
        "30_layer3_1_conv2.yaml",
        "49_layer4_1_conv2.yaml",
    ]

    for layer in layers:
        run_layer(layer)

    print("\nAll selected layer runs completed.")


if __name__ == "__main__":
    main()
