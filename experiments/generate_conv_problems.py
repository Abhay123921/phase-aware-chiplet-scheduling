from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "timeloop/problems/resnet50"


def tensor_shape(tensor):
    return list(tensor.shape)


def register_hooks(model):
    records = []
    hooks = []

    def hook_fn(name):
        def hook(module, inputs, output):
            input_tensor = inputs[0]

            records.append({
                "name": name,
                "type": module.__class__.__name__,
                "module": module,
                "input_shape": tensor_shape(input_tensor),
                "output_shape": tensor_shape(output),
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(
                module.register_forward_hook(hook_fn(name))
            )

    return records, hooks


def sanitize_name(name):
    return name.replace(".", "_")


def make_conv_problem(record):
    module = record["module"]

    input_shape = record["input_shape"]
    output_shape = record["output_shape"]

    # N, C, H, W
    N, C, H, W = input_shape

    # N, M, P, Q
    _, M, P, Q = output_shape

    R, S = module.kernel_size
    stride_h, stride_w = module.stride
    pad_h, pad_w = module.padding
    dilation_h, dilation_w = module.dilation
    groups = module.groups

    # ResNet-50 convolution layers are regular convolutions.
    # Keep G explicit so the problem remains compatible with
    # Timeloop's convolution formulation.
    G = groups

    problem = {
        "problem": {
            "version": 0.4,

            "instance": {
                "C": C,
                "G": G,
                "H": H,
                "Hdilation": dilation_h,
                "Hpad": pad_h,
                "Hstride": stride_h,
                "M": M,
                "N": N,
                "P": P,
                "Q": Q,
                "R": R,
                "S": S,
                "W": W,
                "Wdilation": dilation_w,
                "Wpad": pad_w,
                "Wstride": stride_w,
            },

            "shape": {
                "coefficients": [
                    {"name": "Wstride", "default": 1},
                    {"name": "Hstride", "default": 1},
                    {"name": "Wdilation", "default": 1},
                    {"name": "Hdilation", "default": 1},
                ],

                "data_spaces": [
                    {
                        "name": "Weights",
                        "projection": [
                            [["C"]],
                            [["M"]],
                            [["R"]],
                            [["S"]],
                            [["G"]],
                        ],
                    },

                    {
                        "name": "Inputs",
                        "projection": [
                            [["N"]],
                            [["C"]],
                            [["R", "Wdilation"], ["P", "Wstride"]],
                            [["S", "Hdilation"], ["Q", "Hstride"]],
                            [["G"]],
                        ],
                    },

                    {
                        "name": "Outputs",
                        "projection": [
                            [["N"]],
                            [["M"]],
                            [["Q"]],
                            [["P"]],
                            [["G"]],
                        ],
                        "read_write": True,
                    },
                ],

                "dimensions": [
                    "C",
                    "M",
                    "R",
                    "S",
                    "N",
                    "P",
                    "Q",
                    "G",
                ],
            },
        }
    }

    return problem


def main():
    print("Loading ResNet-50...")

    # We only need the model structure to obtain layer shapes.
    # Pretrained weights are unnecessary for this step.
    model = resnet50(weights=None)
    model.eval()

    records, hooks = register_hooks(model)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(records)} Conv2D layers.")

    for index, record in enumerate(records, start=1):
        filename = (
            f"{index:02d}_{sanitize_name(record['name'])}.yaml"
        )

        output_path = OUTPUT_DIR / filename

        problem = make_conv_problem(record)

        with open(output_path, "w") as f:
            yaml.safe_dump(
                problem,
                f,
                sort_keys=False,
            )

        print(
            f"{index:02d}. {record['name']:25s} "
            f"{record['input_shape']} -> {record['output_shape']} "
            f"-> {output_path.name}"
        )

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
