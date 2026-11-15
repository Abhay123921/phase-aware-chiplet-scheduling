from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "resnet50_timeloop_stats.csv"
OUTPUT = ROOT / "results" / "resnet50_resource_signatures.csv"

df = pd.read_csv(INPUT)

print("=" * 80)
print("BUILDING INTRINSIC RESOURCE SIGNATURES")
print("=" * 80)

print(f"\nInput shape: {df.shape}")


# ============================================================
# 1. Basic identifiers
# ============================================================

df["layer_index"] = np.arange(1, len(df) + 1)


# ============================================================
# 2. Intrinsic compute / memory characteristics
# ============================================================

# Timeloop's optimal operational intensity.
#
# Higher value:
#     more computation per byte
#     relatively compute-intensive
#
# Lower value:
#     more data movement per computation
#     relatively memory-intensive
#
df["compute_intensity"] = df["optimal_op_per_byte"]


# Total memory accesses per operation.
#
# Lower:
#     computation dominates
#
# Higher:
#     more memory traffic relative to computation
#
df["memory_access_per_op"] = (
    df["total_memory_accesses"] /
    df["total_ops"]
)


# Inverse operational intensity.
# Useful because it behaves like a memory-pressure indicator.
df["bytes_per_op"] = (
    1.0 /
    df["compute_intensity"]
)


# ============================================================
# 3. Energy composition
# ============================================================

# Energy values:
#
# DRAM and buffer are already represented as fJ/compute.
# MAC energy is reported in pJ, so convert to fJ.
#
df["mac_energy_fj_per_compute"] = (
    df["mac_energy_pj"] * 1000.0
)


total_energy = df["total_fj_per_compute"].replace(
    0,
    np.nan
)

df["mac_energy_fraction"] = (
    df["mac_energy_fj_per_compute"] /
    total_energy
)

df["buffer_energy_fraction"] = (
    df["buffer_fj_per_compute"] /
    total_energy
)

df["dram_energy_fraction"] = (
    df["dram_fj_per_compute"] /
    total_energy
)


# ============================================================
# 4. Bandwidth composition
# ============================================================

# Keep read and write traffic conceptually separate.
#
# These are normalized later, so absolute Timeloop bandwidth
# values do not dominate the signature simply because they
# use different numerical scales.

buffer_bw_columns = [
    "inputs_shared_bw",
    "weights_shared_bw",
    "outputs_shared_bw",
]

dram_bw_columns = [
    "dram_inputs_shared_bw",
    "dram_weights_shared_bw",
    "dram_outputs_shared_bw",
]


# Verify columns before proceeding.
required = (
    buffer_bw_columns
    + dram_bw_columns
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    print("\nERROR: Missing expected columns:")
    for c in missing:
        print(" ", c)

    print("\nAvailable columns:")
    for c in df.columns:
        print(" ", c)

    raise SystemExit(1)


df["buffer_bandwidth"] = df[
    buffer_bw_columns
].sum(axis=1)

df["dram_bandwidth"] = df[
    dram_bw_columns
].sum(axis=1)


# Ratio of DRAM traffic to total represented bandwidth.
df["dram_bandwidth_fraction"] = (
    df["dram_bandwidth"] /
    (
        df["buffer_bandwidth"]
        + df["dram_bandwidth"]
        + 1e-12
    )
)


# ============================================================
# 5. Execution / utilization characteristics
# ============================================================

# Keep these as secondary features.
df["mac_utilization_signature"] = (
    df["mac_utilization"]
)

df["cycle_intensity"] = (
    df["summary_cycles"] /
    df["summary_computes"]
)


# ============================================================
# 6. Construct compact intrinsic signature
# ============================================================

signature_cols = [
    "layer_index",
    "layer_id",

    # Compute vs memory behavior
    "compute_intensity",
    "bytes_per_op",
    "memory_access_per_op",

    # Energy composition
    "mac_energy_fraction",
    "buffer_energy_fraction",
    "dram_energy_fraction",

    # Data movement
    "buffer_bandwidth",
    "dram_bandwidth",
    "dram_bandwidth_fraction",

    # Secondary execution characteristics
    "mac_utilization_signature",
    "cycle_intensity",
]


signatures = df[signature_cols].copy()


# ============================================================
# 7. Clean numerical values
# ============================================================

signatures = signatures.replace(
    [np.inf, -np.inf],
    np.nan
)

signatures = signatures.fillna(0.0)


# ============================================================
# 8. Print signatures
# ============================================================

print("\n" + "=" * 80)
print("RESOURCE SIGNATURES")
print("=" * 80)

print(
    signatures.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 9. Summary statistics
# ============================================================

numeric_cols = [
    c for c in signatures.columns
    if c not in ["layer_index", "layer_id"]
]

print("\n" + "=" * 80)
print("SIGNATURE SUMMARY")
print("=" * 80)

print(
    signatures[numeric_cols]
    .describe()
    .T
    .to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 10. Save
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

signatures.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(f"\nSaved:")
print(f"  {OUTPUT}")

print(f"\nShape:")
print(f"  {signatures.shape}")

print("\nResource signature construction complete.")
