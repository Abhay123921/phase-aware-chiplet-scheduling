from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "resnet50_timeloop_stats.csv"
OUTPUT = ROOT / "results" / "resnet50_phase_features.csv"

df = pd.read_csv(INPUT)

# Preserve execution order.
df["layer_index"] = range(1, len(df) + 1)

# ---------------------------------------------------------
# 1. Compute demand
# ---------------------------------------------------------
df["compute_demand"] = df["compute_per_cycle"]

# ---------------------------------------------------------
# 2. On-chip buffer bandwidth demand
# ---------------------------------------------------------
df["buffer_read_bw"] = (
    df["inputs_shared_bw"]
    + df["weights_shared_bw"]
    + df["outputs_shared_bw"]
)

# ---------------------------------------------------------
# 3. DRAM bandwidth demand
# ---------------------------------------------------------
df["dram_read_bw"] = (
    df["dram_inputs_shared_bw"]
    + df["dram_weights_shared_bw"]
    + df["dram_outputs_shared_bw"]
)

# Include writes separately for future analysis.
df["buffer_write_bw"] = (
    df["inputs_write_bw"]
    + df["weights_write_bw"]
    + df["outputs_write_bw"]
)

df["dram_write_bw"] = (
    df["dram_inputs_write_bw"]
    + df["dram_weights_write_bw"]
    + df["dram_outputs_write_bw"]
)

# ---------------------------------------------------------
# 4. Energy intensity
# ---------------------------------------------------------
df["energy_per_compute"] = df["total_fj_per_compute"]

# ---------------------------------------------------------
# 5. Useful additional indicators
# ---------------------------------------------------------
df["mac_util"] = df["mac_utilization"]

df["memory_access_intensity"] = (
    df["total_memory_accesses"] / df["summary_computes"]
)

# ---------------------------------------------------------
# Select compact phase-analysis representation
# ---------------------------------------------------------
features = df[
    [
        "layer_index",
        "layer_id",
        "compute_demand",
        "mac_util",
        "buffer_read_bw",
        "buffer_write_bw",
        "dram_read_bw",
        "dram_write_bw",
        "energy_per_compute",
        "memory_access_intensity",
        "summary_cycles",
        "total_energy_uJ",
    ]
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT, index=False)

print("Created:", OUTPUT)
print("Shape:", features.shape)

print("\nPhase-analysis features:")
print(features.to_string(index=False))
