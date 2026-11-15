from pathlib import Path
import pytimeloop.timeloopfe.v4 as tl

ROOT = Path(__file__).resolve().parents[1]

ARCH_PATH = ROOT / "timeloop/architectures/reference_chiplet.yaml"
COMPONENTS_PATH = ROOT / "timeloop/components/components.yaml"
MAPPER_PATH = ROOT / "timeloop/mappings/mapper.yaml"
VARIABLES_PATH = ROOT / "timeloop/variables.yaml"

PROBLEM_PATH = ROOT / "timeloop/problems/resnet50/12_layer2_0_conv1.yaml"
OUTPUT_PATH = ROOT / "timeloop/outputs/resnet50/12_layer2_0_conv1"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Running: 12_layer2_0_conv1.yaml")
print("=" * 70)

spec = tl.Specification.from_yaml_files(
    str(ARCH_PATH),
    str(COMPONENTS_PATH),
    str(MAPPER_PATH),
    str(PROBLEM_PATH),
    str(VARIABLES_PATH),
)

tl.call_mapper(
    spec,
    output_dir=str(OUTPUT_PATH),
)

print("\nRun completed.")
