from pathlib import Path
import pytimeloop.timeloopfe.v4 as tl


ROOT = Path(__file__).resolve().parents[1]

ARCH_PATH = ROOT / "timeloop/architectures/reference_chiplet.yaml"
COMPONENTS_PATH = ROOT / "timeloop/components/components.yaml"
MAPPER_PATH = ROOT / "timeloop/mappings/mapper.yaml"
PROBLEM_PATH = ROOT / "timeloop/problems/resnet50_conv1.yaml"
VARIABLES_PATH = ROOT / "timeloop/variables.yaml"
OUTPUT_DIR = ROOT / "timeloop/outputs/conv1_reference"


def main():
    print("=== P04 Timeloop Conv1 Test ===")
    print(f"Architecture : {ARCH_PATH}")
    print(f"Components   : {COMPONENTS_PATH}")
    print(f"Mapper       : {MAPPER_PATH}")
    print(f"Problem      : {PROBLEM_PATH}")
    print(f"Variables    : {VARIABLES_PATH}")
    print(f"Output       : {OUTPUT_DIR}")

    spec = tl.Specification.from_yaml_files(
        str(ARCH_PATH),
        str(COMPONENTS_PATH),
        str(MAPPER_PATH),
        str(PROBLEM_PATH),
        str(VARIABLES_PATH),
    )

    tl.call_mapper(
        spec,
        output_dir=str(OUTPUT_DIR),
    )

    print("\nTimeloop run completed.")


if __name__ == "__main__":
    main()
