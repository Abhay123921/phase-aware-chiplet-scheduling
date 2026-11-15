from dataclasses import dataclass
from pathlib import Path
import yaml


REFERENCE_MAC_INSTANCES = 16
REFERENCE_SRAM_WORDS = 4096


@dataclass(frozen=True)
class Chiplet:
    name: str
    kind: str
    compute_capacity: float
    memory_capacity: float
    memory_bandwidth: float
    sram_capacity_words: int


@dataclass(frozen=True)
class HardwareSystem:
    name: str
    chiplets: dict[str, Chiplet]
    nop_topology: str
    nop_bandwidth: float
    nop_hop_latency: float


def load_hardware(config_path=None):
    root = Path(__file__).resolve().parents[2]

    if config_path is None:
        config_path = root / "configs/hardware/baseline.yaml"
    else:
        config_path = Path(config_path)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    system_cfg = config["system"]
    chiplet_cfg = config["chiplets"]
    nop_cfg = config["nop"]

    if len(chiplet_cfg) != system_cfg["num_chiplets"]:
        raise ValueError(
            "Configured chiplet count does not match num_chiplets"
        )

    chiplets = {}

    for name, item in chiplet_cfg.items():
        compute = float(item["compute_capacity"])
        memory = float(item["memory_capacity"])
        bandwidth = float(item["memory_bandwidth"])

        if min(compute, memory, bandwidth) <= 0:
            raise ValueError(f"Non-positive capacity in {name}")

        # Explicit simulator convention:
        # memory_capacity=1.0 represents 4096 local SRAM words.
        # Other values scale this nominal capacity proportionally.
        sram_words = round(memory * REFERENCE_SRAM_WORDS)

        chiplets[name] = Chiplet(
            name=name,
            kind=item["type"],
            compute_capacity=compute,
            memory_capacity=memory,
            memory_bandwidth=bandwidth,
            sram_capacity_words=sram_words,
        )

    return HardwareSystem(
        name=system_cfg["name"],
        chiplets=chiplets,
        nop_topology=nop_cfg["topology"],
        nop_bandwidth=float(nop_cfg["bandwidth"]),
        nop_hop_latency=float(nop_cfg["hop_latency"]),
    )


if __name__ == "__main__":
    hw = load_hardware()

    print(f"System: {hw.name}")
    print(f"Chiplets: {len(hw.chiplets)}")
    print(f"Reference MAC instances: {REFERENCE_MAC_INSTANCES}")
    print(f"Reference SRAM: {REFERENCE_SRAM_WORDS} words")
    print()

    for chiplet in hw.chiplets.values():
        print(
            f"{chiplet.name}: type={chiplet.kind}, "
            f"compute={chiplet.compute_capacity:.2f}x, "
            f"SRAM={chiplet.sram_capacity_words} words, "
            f"bandwidth_scale={chiplet.memory_bandwidth:.2f}x"
        )

    print(
        f"\nNoP: topology={hw.nop_topology}, "
        f"bandwidth_scale={hw.nop_bandwidth:.2f}, "
        f"hop_latency_scale={hw.nop_hop_latency:.2f}"
    )
