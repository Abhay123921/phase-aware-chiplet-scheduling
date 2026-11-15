from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")

def read_csv(relative):
    path = RESULTS / relative
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)

def parse_counts(value):
    if isinstance(value, dict):
        return value
    return json.loads(value)

# F2: ResNet-50 layer-wise resource signatures
df = read_csv("resnet50_resource_signatures.csv").sort_values("layer_index")
x = df["layer_index"].to_numpy()
labels = df["layer_id"].astype(str).to_numpy()

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
axes[0].plot(x, df["compute_intensity"], marker="o", markersize=2.5, linewidth=1)
axes[0].set_ylabel("Compute intensity\n(ops/byte)")
axes[0].set_title("(a) Compute intensity")

axes[1].plot(x, df["buffer_bandwidth"], label="Local buffer", linewidth=1.5)
axes[1].plot(x, df["dram_bandwidth"], label="DRAM", linewidth=1.5)
axes[1].set_ylabel("Bandwidth\n(model units)")
axes[1].set_title("(b) Memory bandwidth characteristics")
axes[1].legend(frameon=False, ncol=2)

axes[2].plot(x, df["dram_energy_fraction"], label="DRAM energy fraction", linewidth=1.5)
axes[2].plot(x, df["mac_energy_fraction"], label="MAC energy fraction", linewidth=1.5)
axes[2].set_ylabel("Energy fraction")
axes[2].set_xlabel("ResNet-50 layer (ordered index)")
axes[2].set_title("(c) Energy contribution fractions")
axes[2].set_ylim(bottom=0)
axes[2].legend(frameon=False, ncol=2)

tick_positions = x[::4]
axes[2].set_xticks(tick_positions)
axes[2].set_xticklabels([labels[int(i)-1] for i in tick_positions],
                        rotation=55, ha="right")
fig.suptitle("ResNet-50 Layer-wise Resource Characteristics")
save(fig, "fig2_resnet50_resource_characteristics")

# F4: E1 policy comparison, separate panels for aggregation methods
e1 = read_csv("compact_group_rr_validation.csv")
policies = ["static", "static_rr", "reactive", "phase_aware"]
policy_labels = ["Static", "Static RR", "Reactive", "Phase-aware"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, aggregation in zip(axes, ["max", "mean"]):
    part = e1[e1["aggregation"] == aggregation].set_index("policy").reindex(policies)
    bars = ax.bar(policy_labels, part["makespan"])
    ax.set_title(f"{aggregation.capitalize()} aggregation")
    ax.set_ylabel("Modeled makespan (cycles)")
    ax.tick_params(axis="x", rotation=20)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in part["makespan"]],
                 padding=3, fontsize=7)
fig.suptitle("E1: Compact Execution-Group Scheduling Comparison")
save(fig, "fig4_e1_policy_comparison")

# F5: E2 calibration sensitivity, isolate Reactive policy
e2 = read_csv("calibration_ablation_e2/calibration_ablation_e2.csv")
e2 = e2[e2["policy"] == "reactive"].copy()
e2["profile_label"] = e2["profile_source"].map({
    "offline_timeloop_reference": "Offline Timeloop",
    "metadata_causal": "Metadata-causal",
}).fillna(e2["profile_source"])
e2["scale_label"] = e2["scale_source"].map({
    "offline_v2": "Offline v2 scales",
    "causal_v2": "Causal v2 scales",
}).fillna(e2["scale_source"])

fig, ax = plt.subplots(figsize=(8.5, 5))
profiles = ["Offline Timeloop", "Metadata-causal"]
scales = ["Offline v2 scales", "Causal v2 scales"]
width = 0.34
positions = np.arange(len(profiles))
for j, scale in enumerate(scales):
    values = []
    for profile in profiles:
        row = e2[(e2["profile_label"] == profile) &
                 (e2["scale_label"] == scale)]
        if len(row) != 1:
            raise ValueError(f"Expected one E2 Reactive row for {profile}, {scale}; found {len(row)}")
        values.append(float(row["makespan"].iloc[0]))
    bars = ax.bar(positions + (j - 0.5) * width, values,
                  width, label=scale)
    ax.bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in values],
                 padding=3, fontsize=8)
ax.set_xticks(positions)
ax.set_xticklabels(profiles)
ax.set_ylabel("Modeled makespan (cycles, log scale)")
ax.set_yscale("log")
ax.set_title("E2: Sensitivity of Reactive Scheduling to Calibration")
ax.legend(frameon=False)
save(fig, "fig5_e2_calibration_sensitivity")

# F6: E5 and E6 mixed-workload comparisons
e5 = read_csv("mixed_cnn_gemm_e5/mixed_cnn_gemm_e5.csv")
e6 = read_csv("phase_varying_mixed_e6/phase_varying_mixed_e6.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

for ax, data, title, order in [
    (axes[0], e5, "E5: Concurrent CNN + GEMM",
     ["static", "static_rr", "reactive"]),
    (axes[1], e6, "E6: Sequential CNN layers + concurrent GEMM",
     ["static", "static_rr", "reactive", "causal_phase_aware"]),
]:
    part = data.set_index("policy").reindex(order).dropna(subset=["makespan"])
    names = {
        "static": "Static",
        "static_rr": "Static RR",
        "reactive": "Reactive",
        "causal_phase_aware": "Causal phase-aware",
    }
    bars = ax.bar([names[p] for p in part.index], part["makespan"])
    ax.set_title(title)
    ax.set_ylabel("Modeled makespan (cycles)")
    ax.tick_params(axis="x", rotation=22)
    ax.bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in part["makespan"]],
                 padding=3, fontsize=8)
fig.suptitle("Mixed-Workload Scheduling Results")
save(fig, "fig6_e5_e6_mixed_workloads")

# F7: E7 controlled resource-phase experiment
e7 = read_csv("controlled_phase_e7/controlled_phase_e7.csv")
order = ["static_rr", "reactive", "causal_phase_aware"]
e7 = e7.set_index("policy").reindex(order).dropna(subset=["makespan"])
names = ["Static RR", "Reactive", "Causal phase-aware"]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.bar(names, e7["makespan"])
ax.set_ylabel("Modeled makespan (cycles)")
ax.set_title("E7: Controlled Compute- and Memory-Demand Workloads")
ax.bar_label(bars, labels=[f"{v:,.0f}" for v in e7["makespan"]],
             padding=3)
improvement = (1 - e7.loc["causal_phase_aware", "makespan"] /
               e7.loc["reactive", "makespan"]) * 100
ax.text(0.98, 0.95, f"Phase-aware vs Reactive: {improvement:.2f}% lower",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="gray", alpha=0.9))
save(fig, "fig7_e7_controlled_phase_experiment")

# Supplementary: E3/E4 policy ablation relative to Reactive
e3 = read_csv("causal_phase_aware_e3/causal_phase_aware_e3.csv")
e4 = read_csv("locality_aware_e4/locality_aware_e4.csv")
base = e3[e3["policy"] == "reactive"][[
    "profile_source", "scale_source", "makespan"
]].rename(columns={"makespan": "reactive_makespan"})

def relative_to_reactive(data, policies_to_plot):
    subset = data[data["policy"].isin(policies_to_plot)].copy()
    merged = subset.merge(base, on=["profile_source", "scale_source"],
                          validate="many_to_one")
    merged["change_percent"] = (
        (merged["makespan"] / merged["reactive_makespan"]) - 1
    ) * 100
    merged["condition"] = (
        merged["profile_source"].map({
            "offline_timeloop_reference": "Offline profile",
            "metadata_causal": "Metadata profile",
        }).fillna(merged["profile_source"]) + "\n" +
        merged["scale_source"]
    )
    return merged

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, policies_to_plot, title in [
    (axes[0], e3, ["causal_phase_aware", "oracle_phase_aware"],
     "E3: Phase-aware variants"),
    (axes[1], e4, ["locality_aware"],
     "E4: Locality-aware policy"),
]:
    rel = relative_to_reactive(data, policies_to_plot)
    variants = policies_to_plot
    conditions = list(dict.fromkeys(rel["condition"]))
    width = 0.8 / max(len(variants), 1)
    xloc = np.arange(len(conditions))
    for j, policy in enumerate(variants):
        vals = []
        for condition in conditions:
            row = rel[(rel["condition"] == condition) &
                      (rel["policy"] == policy)]
            vals.append(float(row["change_percent"].iloc[0]) if len(row) else np.nan)
        label = policy.replace("_", " ").title()
        ax.bar(xloc + (j - (len(variants)-1)/2)*width, vals,
               width, label=label)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(xloc)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylabel("Makespan change vs Reactive (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
fig.suptitle("E3–E4: Scheduler Ablations Relative to Reactive")
save(fig, "fig8_e3_e4_scheduler_ablations")

# Supplementary: chiplet placement distribution in E6
fig, ax = plt.subplots(figsize=(8.5, 5))
chiplets = ["C1", "C2", "C3", "C4"]
policies = ["static", "static_rr", "reactive", "causal_phase_aware"]
labels = ["Static", "Static RR", "Reactive", "Causal phase-aware"]
bottom = np.zeros(len(policies))
colors = None
e6_indexed = e6.set_index("policy")
for chiplet in chiplets:
    vals = []
    for policy in policies:
        counts = parse_counts(e6_indexed.loc[policy, "chiplet_group_counts"])
        vals.append(counts[chiplet])
    ax.bar(labels, vals, bottom=bottom, label=chiplet)
    bottom += np.array(vals)
ax.set_ylabel("Number of assigned logical groups")
ax.set_title("E6: Chiplet-wise Group Placement")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Chiplet", frameon=False, ncol=4)
save(fig, "fig9_e6_chiplet_placement")

print(f"\nAll figures saved under: {OUT}")
