from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "resnet50_phase_features.csv"
OUTPUT_CSV = ROOT / "results" / "resnet50_phase_sequence_analysis.csv"
OUTPUT_PLOT = ROOT / "results" / "resnet50_resource_sequence.png"


df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# 1. Resource-demand features
# ------------------------------------------------------------

feature_cols = [
    "compute_demand",
    "buffer_read_bw",
    "buffer_write_bw",
    "dram_read_bw",
    "dram_write_bw",
    "memory_access_intensity",
]

X = df[feature_cols].astype(float).values


# ------------------------------------------------------------
# 2. Standardize WITHOUT sklearn
# ------------------------------------------------------------

mean = X.mean(axis=0)
std = X.std(axis=0)

# Avoid division by zero.
std[std == 0] = 1.0

Z = (X - mean) / std


# ------------------------------------------------------------
# 3. Consecutive-layer resource distance
# ------------------------------------------------------------

consecutive_distance = np.zeros(len(df))

for i in range(1, len(df)):
    consecutive_distance[i] = np.linalg.norm(
        Z[i] - Z[i - 1]
    )

df["consecutive_distance"] = consecutive_distance


# ------------------------------------------------------------
# 4. Smoothed resource behavior
# ------------------------------------------------------------

for window in [2, 3, 4]:

    for feature in feature_cols:

        df[f"{feature}_smooth_{window}"] = (
            df[feature]
            .rolling(
                window=window,
                center=True,
                min_periods=1
            )
            .mean()
        )


# ------------------------------------------------------------
# 5. Simple exploratory clustering WITHOUT sklearn
#
# This is intentionally only exploratory.
# These clusters are NOT being called phases.
# ------------------------------------------------------------

def simple_kmeans(X, k, iterations=100, seed=42):

    rng = np.random.default_rng(seed)

    n = len(X)

    # Randomly choose initial centroids.
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(iterations):

        # Distance from every point to every centroid.
        distances = np.zeros((n, k))

        for j in range(k):
            distances[:, j] = np.linalg.norm(
                X - centroids[j],
                axis=1
            )

        new_labels = np.argmin(
            distances,
            axis=1
        )

        # Stop if assignments no longer change.
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        # Recalculate centroids.
        for j in range(k):

            members = X[labels == j]

            if len(members) > 0:
                centroids[j] = members.mean(axis=0)

    return labels


for k in [2, 3, 4]:

    df[f"cluster_k{k}"] = simple_kmeans(
        Z,
        k=k
    )


# ------------------------------------------------------------
# 6. Candidate boundaries
# ------------------------------------------------------------

threshold = np.percentile(
    consecutive_distance[1:],
    75
)

df["candidate_boundary"] = (
    df["consecutive_distance"] >= threshold
)

df.loc[0, "candidate_boundary"] = False


# ------------------------------------------------------------
# 7. Print analysis
# ------------------------------------------------------------

print("=" * 80)
print("RESNET-50 RESOURCE-PHASE SEQUENCE ANALYSIS")
print("=" * 80)

print(f"\nNumber of layers: {len(df)}")

print("\nResource features:")
for feature in feature_cols:
    print(f"  {feature}")


print("\n" + "-" * 80)
print("CONSECUTIVE RESOURCE-DISTANCE STATISTICS")
print("-" * 80)

distance_stats = pd.Series(
    consecutive_distance[1:]
).describe()

print(
    distance_stats.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n75th-percentile boundary threshold:")
print(f"  {threshold:.4f}")


print("\n" + "-" * 80)
print("CANDIDATE PHASE BOUNDARIES")
print("-" * 80)

boundaries = df[
    df["candidate_boundary"]
]

for _, row in boundaries.iterrows():

    idx = int(row["layer_index"])

    previous = df.iloc[idx - 2]["layer_id"]

    print(
        f"After layer {idx-1:02d} "
        f"({previous}) "
        f"-> layer {idx:02d} "
        f"({row['layer_id']}) "
        f"distance={row['consecutive_distance']:.4f}"
    )


# ------------------------------------------------------------
# 8. Cluster sequence
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RESOURCE CLUSTER SEQUENCES")
print("=" * 80)


for k in [2, 3, 4]:

    labels = df[f"cluster_k{k}"].tolist()

    print(f"\nK={k}")

    start = 0

    for i in range(1, len(labels) + 1):

        if (
            i == len(labels)
            or labels[i] != labels[start]
        ):

            print(
                f"  layers {start+1:02d}-{i:02d}: "
                f"cluster {labels[start]}"
            )

            start = i


# ------------------------------------------------------------
# 9. Stage-level resource summary
# ------------------------------------------------------------

def get_stage(layer_id):

    if layer_id.startswith("01_"):
        return "conv1"

    if "layer1_" in layer_id:
        return "layer1"

    if "layer2_" in layer_id:
        return "layer2"

    if "layer3_" in layer_id:
        return "layer3"

    if "layer4_" in layer_id:
        return "layer4"

    return "other"


df["stage"] = df["layer_id"].apply(
    get_stage
)


print("\n" + "=" * 80)
print("STAGE-LEVEL RESOURCE SUMMARY")
print("=" * 80)

stage_summary = (
    df.groupby("stage")[feature_cols]
    .mean()
)

print(
    stage_summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ------------------------------------------------------------
# 10. Save analysis table
# ------------------------------------------------------------

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ------------------------------------------------------------
# 11. Resource sequence plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    len(feature_cols),
    1,
    figsize=(14, 18),
    sharex=True
)

for ax, feature in zip(
    axes,
    feature_cols
):

    ax.plot(
        df["layer_index"],
        df[feature],
        marker="o",
        linewidth=1.2,
        markersize=3
    )

    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.25)

    boundary_indices = df.loc[
        df["candidate_boundary"],
        "layer_index"
    ]

    for boundary in boundary_indices:

        ax.axvline(
            boundary - 0.5,
            linestyle="--",
            alpha=0.35
        )


axes[-1].set_xlabel(
    "Execution layer index"
)

fig.suptitle(
    "ResNet-50 Resource-Demand Evolution Across Execution",
    fontsize=14
)

plt.tight_layout()

fig.savefig(
    OUTPUT_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 12. Final output
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nSequence analysis:\n  {OUTPUT_CSV}"
)

print(
    f"\nResource sequence plot:\n  {OUTPUT_PLOT}"
)

print("\nAnalysis complete.")
