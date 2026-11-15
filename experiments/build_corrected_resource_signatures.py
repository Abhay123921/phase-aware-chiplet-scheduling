import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/workspace/P04")

inp = ROOT / "results/resnet50_timeloop_stats.csv"
out = ROOT / "results/resnet50_resource_signatures.csv"

df = pd.read_csv(inp)

# ---------------------------------------------------------
# Basic intrinsic resource features
# ---------------------------------------------------------

# Compute intensity:
# total operations / total memory traffic in bytes
compute_intensity = (
    df["optimal_op_per_byte"]
)

# Memory-access intensity:
# memory accesses / operations
memory_access_per_op = (
    df["total_memory_accesses"] /
    df["total_ops"]
)

# Aggregate buffer bandwidth.
# Read + write bandwidth captures total buffer pressure.
buffer_bandwidth = (
    df["inputs_read_bw"] +
    df["inputs_write_bw"] +
    df["outputs_read_bw"] +
    df["outputs_write_bw"] +
    df["weights_read_bw"] +
    df["weights_write_bw"]
)

# Aggregate DRAM bandwidth.
dram_bandwidth = (
    df["dram_inputs_read_bw"] +
    df["dram_inputs_write_bw"] +
    df["dram_outputs_read_bw"] +
    df["dram_outputs_write_bw"] +
    df["dram_weights_read_bw"] +
    df["dram_weights_write_bw"]
)

# DRAM energy fraction.
dram_energy_fraction = (
    df["dram_fj_per_compute"] /
    df["total_fj_per_compute"]
)

# ---------------------------------------------------------
# Correct MAC energy fraction
# ---------------------------------------------------------
#
# mac_energy_pj is TOTAL MAC energy in pJ.
# Divide by computes -> pJ/compute.
# Convert pJ -> fJ.
#
# mac_energy_pj / computes * 1000
#
mac_energy_per_compute_fj = (
    df["mac_energy_pj"] /
    df["computes"] *
    1000.0
)

mac_energy_fraction = (
    mac_energy_per_compute_fj /
    df["total_fj_per_compute"]
)

# ---------------------------------------------------------
# Secondary features
# ---------------------------------------------------------

mac_utilization = df["mac_utilization"]

cycle_intensity = (
    df["cycles"] /
    df["computes"]
)

# ---------------------------------------------------------
# Construct output
# ---------------------------------------------------------

sig = pd.DataFrame({
    "layer_index": np.arange(1, len(df) + 1),
    "layer_id": df["layer_id"],

    # Primary phase/resource signature
    "compute_intensity": compute_intensity,
    "memory_access_per_op": memory_access_per_op,
    "buffer_bandwidth": buffer_bandwidth,
    "dram_bandwidth": dram_bandwidth,
    "dram_energy_fraction": dram_energy_fraction,

    # Secondary characterization
    "mac_utilization": mac_utilization,
    "cycle_intensity": cycle_intensity,
    "mac_energy_fraction": mac_energy_fraction,

    # Useful reference quantities
    "total_fj_per_compute": df["total_fj_per_compute"],
    "summary_cycles": df["summary_cycles"],
    "total_energy_uJ": df["total_energy_uJ"],
})

sig.to_csv(out, index=False)

print("=" * 70)
print("Corrected resource signature")
print("=" * 70)
print(f"Input : {inp}")
print(f"Output: {out}")
print()
print("Shape:", sig.shape)
print()
print("Columns:")
for c in sig.columns:
    print(" ", c)

print()
print("First 10 rows:")
print(sig.head(10).to_string(index=False))

print()
print("Feature ranges:")
for c in [
    "compute_intensity",
    "memory_access_per_op",
    "buffer_bandwidth",
    "dram_bandwidth",
    "dram_energy_fraction",
    "mac_energy_fraction",
    "mac_utilization",
    "cycle_intensity",
]:
    print(
        f"{c:25s}: "
        f"min={sig[c].min():.6f}, "
        f"mean={sig[c].mean():.6f}, "
        f"max={sig[c].max():.6f}"
    )

print()
print("Energy fraction sanity check:")
print(
    "MAC fraction + DRAM fraction + buffer fraction "
    "should approximately explain total energy."
)

buffer_fraction = (
    df["buffer_fj_per_compute"] /
    df["total_fj_per_compute"]
)

energy_check = (
    mac_energy_fraction +
    buffer_fraction +
    dram_energy_fraction
)

print(
    f"Energy-fraction sum: "
    f"min={energy_check.min():.4f}, "
    f"mean={energy_check.mean():.4f}, "
    f"max={energy_check.max():.4f}"
)

print()
print("Saved successfully.")
