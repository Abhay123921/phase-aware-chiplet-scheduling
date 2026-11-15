import pandas as pd
import yaml
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HW_FILE = ROOT / "configs/hardware/baseline.yaml"
REGIME_FILE = ROOT / "results/resnet50_resource_regimes.csv"
OUT_FILE = ROOT / "results/regime_chiplet_capability.csv"


# ============================================================
# 1. Load hardware configuration
# ============================================================

with open(HW_FILE, "r") as f:
    hw = yaml.safe_load(f)

chiplets = hw["chiplets"]

chiplet_rows = []

for chiplet_id, cfg in chiplets.items():
    chiplet_rows.append({
        "chiplet": chiplet_id,
        "compute_capacity": float(cfg["compute_capacity"]),
        "memory_capacity": float(cfg["memory_capacity"]),
        "memory_bandwidth": float(cfg["memory_bandwidth"]),
    })

chiplet_df = pd.DataFrame(chiplet_rows)


# ============================================================
# 2. Load discovered resource regimes
# ============================================================

df = pd.read_csv(REGIME_FILE)

regime_features = [
    "compute_intensity",
    "dram_bandwidth",
    "buffer_bandwidth",
]

regimes = (
    df.groupby("regime")[regime_features]
      .mean()
      .sort_index()
)

print("\n" + "=" * 70)
print("RESOURCE REGIMES")
print("=" * 70)
print(regimes)


# ============================================================
# 3. Normalize workload resource characteristics
#
# NOTE:
# This is NOT an execution-performance model.
# It only provides a relative resource-demand representation.
# ============================================================

def minmax(series):
    mn = series.min()
    mx = series.max()

    if np.isclose(mx, mn):
        return pd.Series(0.5, index=series.index)

    return (series - mn) / (mx - mn)


regimes["compute_demand_norm"] = minmax(
    regimes["compute_intensity"]
)

regimes["dram_bw_demand_norm"] = minmax(
    regimes["dram_bandwidth"]
)

regimes["buffer_bw_demand_norm"] = minmax(
    regimes["buffer_bandwidth"]
)


# ============================================================
# 4. Normalize chiplet capabilities
# ============================================================

chiplet_df["compute_cap_norm"] = (
    chiplet_df["compute_capacity"]
    / chiplet_df["compute_capacity"].max()
)

chiplet_df["memory_bw_cap_norm"] = (
    chiplet_df["memory_bandwidth"]
    / chiplet_df["memory_bandwidth"].max()
)


# ============================================================
# 5. Calculate analytical compatibility
#
# We compare:
#
#     workload compute demand  <-> chiplet compute capability
#     workload DRAM BW demand  <-> chiplet memory BW capability
#
# This is a relative compatibility score.
#
# It must NOT be interpreted as latency, throughput, or speedup.
# ============================================================

results = []

EPS = 1e-9

for regime_id, regime in regimes.iterrows():

    d_compute = regime["compute_demand_norm"]
    d_bw = regime["dram_bw_demand_norm"]

    for _, chip in chiplet_df.iterrows():

        c_compute = chip["compute_cap_norm"]
        c_bw = chip["memory_bw_cap_norm"]

        # Absolute mismatch in normalized resource space
        compute_mismatch = abs(d_compute - c_compute)
        bw_mismatch = abs(d_bw - c_bw)

        # Equal weighting for the first analytical model
        mismatch = (
            0.5 * compute_mismatch
            + 0.5 * bw_mismatch
        )

        compatibility = 1.0 / (1.0 + mismatch)

        results.append({
            "regime": int(regime_id),

            "chiplet": chip["chiplet"],

            "compute_demand_norm": d_compute,
            "dram_bw_demand_norm": d_bw,
            "buffer_bw_demand_norm":
                regime["buffer_bw_demand_norm"],

            "compute_cap_norm": c_compute,
            "memory_bw_cap_norm": c_bw,

            "compute_mismatch": compute_mismatch,
            "dram_bw_mismatch": bw_mismatch,

            "compatibility_score": compatibility,
        })


result_df = pd.DataFrame(results)


# ============================================================
# 6. Determine analytical best match
#
# Again: this is NOT an empirical performance result.
# ============================================================

best_matches = (
    result_df
    .sort_values(
        ["regime", "compatibility_score"],
        ascending=[True, False]
    )
    .groupby("regime")
    .first()
    .reset_index()
)


# ============================================================
# 7. Print results
# ============================================================

print("\n" + "=" * 70)
print("CHIPLET CAPABILITIES")
print("=" * 70)

print(
    chiplet_df[
        [
            "chiplet",
            "compute_capacity",
            "memory_capacity",
            "memory_bandwidth",
            "compute_cap_norm",
            "memory_bw_cap_norm",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("REGIME × CHIPLET COMPATIBILITY")
print("=" * 70)

pivot = result_df.pivot(
    index="regime",
    columns="chiplet",
    values="compatibility_score"
)

print(pivot.round(4).to_string())


print("\n" + "=" * 70)
print("ANALYTICAL BEST MATCH")
print("=" * 70)

print(
    best_matches[
        [
            "regime",
            "chiplet",
            "compatibility_score",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("IMPORTANT INTERPRETATION")
print("=" * 70)

print(
    "These scores represent normalized resource-profile compatibility "
    "only. They are NOT measured latency, throughput, speedup, or energy."
)


# ============================================================
# 8. Save
# ============================================================

result_df.to_csv(OUT_FILE, index=False)

print("\nSaved:")
print(OUT_FILE)
