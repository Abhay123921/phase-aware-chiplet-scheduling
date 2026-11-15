import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results/resnet50_timeloop_stats.csv"
OUTPUT_FILE = ROOT / "results/resnet50_memory_profiles.csv"


# ============================================================
# 1. Load Timeloop statistics
# ============================================================

df = pd.read_csv(INPUT_FILE)

required = [
    "layer_id",

    "inputs_utilized_capacity",
    "outputs_utilized_capacity",
    "weights_utilized_capacity",

    "dram_inputs_utilized_capacity",
    "dram_outputs_utilized_capacity",
    "dram_weights_utilized_capacity",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 2. Construct memory profile
# ============================================================

out = df[required].copy()

local_cols = [
    "inputs_utilized_capacity",
    "outputs_utilized_capacity",
    "weights_utilized_capacity",
]

dram_cols = [
    "dram_inputs_utilized_capacity",
    "dram_outputs_utilized_capacity",
    "dram_weights_utilized_capacity",
]


# Peak individual local-buffer requirement
out["local_capacity_peak"] = out[local_cols].max(axis=1)

# Sum retained as an auxiliary statistic.
# We DO NOT assume this is the actual simultaneous footprint.
out["local_capacity_sum"] = out[local_cols].sum(axis=1)


# Peak individual DRAM footprint
out["dram_capacity_peak"] = out[dram_cols].max(axis=1)

# Auxiliary DRAM total
out["dram_capacity_sum"] = out[dram_cols].sum(axis=1)


# ============================================================
# 3. Print summary
# ============================================================

print("\n" + "=" * 80)
print("MEMORY PROFILE SUMMARY")
print("=" * 80)

summary_cols = [
    "inputs_utilized_capacity",
    "outputs_utilized_capacity",
    "weights_utilized_capacity",
    "local_capacity_peak",
    "local_capacity_sum",
    "dram_capacity_peak",
    "dram_capacity_sum",
]

print(
    out[summary_cols]
    .describe()
    .T
    .to_string()
)


print("\n" + "=" * 80)
print("FIRST 10 LAYERS")
print("=" * 80)

print(
    out[
        [
            "layer_id",
            "inputs_utilized_capacity",
            "outputs_utilized_capacity",
            "weights_utilized_capacity",
            "local_capacity_peak",
            "local_capacity_sum",
            "dram_capacity_peak",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 4. Save
# ============================================================

out.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:")
print(OUTPUT_FILE)
