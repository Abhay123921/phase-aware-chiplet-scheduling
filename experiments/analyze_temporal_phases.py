import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/workspace/P04")

inp = ROOT / "results/resnet50_resource_signatures.csv"

df = pd.read_csv(inp)

# ============================================================
# 1. Primary resource signature
# ============================================================

features = [
    "compute_intensity",
    "memory_access_per_op",
    "buffer_bandwidth",
    "dram_bandwidth",
    "dram_energy_fraction",
]

X = df[features].values.astype(float)

# Standardize each resource dimension.
mean = X.mean(axis=0)
std = X.std(axis=0)

Z = (X - mean) / std

# ============================================================
# 2. Consecutive resource distance
# ============================================================

distances = np.linalg.norm(Z[1:] - Z[:-1], axis=1)

df["transition_distance"] = np.r_[np.nan, distances]

print("=" * 80)
print("TEMPORAL RESOURCE-PHASE ANALYSIS")
print("=" * 80)

print()
print("Number of layers:", len(df))

print()
print("Primary features:")
for f in features:
    print("  ", f)

print()
print("Consecutive resource distance:")
print(f"  mean   = {distances.mean():.4f}")
print(f"  std    = {distances.std():.4f}")
print(f"  min    = {distances.min():.4f}")
print(f"  median = {np.median(distances):.4f}")
print(f"  75%    = {np.percentile(distances, 75):.4f}")
print(f"  max    = {distances.max():.4f}")

# ============================================================
# 3. Print transitions ranked by magnitude
# ============================================================

print()
print("=" * 80)
print("LARGEST RESOURCE TRANSITIONS")
print("=" * 80)

order = np.argsort(distances)[::-1]

for idx in order[:15]:
    left = idx + 1
    right = idx + 2

    print(
        f"  after layer {left:2d} -> layer {right:2d} "
        f"distance={distances[idx]:.4f}"
    )

# ============================================================
# 4. Simple candidate segmentation
#
# Use high transition distances as candidate boundaries.
# We intentionally do NOT call these final phases yet.
# ============================================================

thresholds = [
    np.percentile(distances, 80),
    np.percentile(distances, 85),
    np.percentile(distances, 90),
]

for threshold in thresholds:

    boundaries = np.where(distances >= threshold)[0] + 1

    print()
    print("=" * 80)
    print(
        f"CANDIDATE SEGMENTATION "
        f"(distance >= {threshold:.4f})"
    )
    print("=" * 80)

    print("Candidate boundaries:", boundaries.tolist())

    start = 0

    for phase_id, end in enumerate(
        list(boundaries) + [len(df)],
        start=1
    ):

        segment = Z[start:end]

        center = segment.mean(axis=0)

        within = np.mean(
            np.sum(
                (segment - center) ** 2,
                axis=1
            )
        )

        print(
            f"  Phase {phase_id}: "
            f"layers {start+1}-{end}, "
            f"length={end-start}, "
            f"within_var={within:.4f}"
        )

        start = end

# ============================================================
# 5. Candidate phase profile
#
# Use 3 and 4 coarse phases based on dynamic programming.
# We minimize within-segment squared error.
# ============================================================

def segment_cost(Z, i, j):
    """
    Cost of assigning layers [i:j] to one phase.
    """
    seg = Z[i:j]

    if len(seg) == 0:
        return 0.0

    center = seg.mean(axis=0)

    return np.sum((seg - center) ** 2)


def optimal_segmentation(Z, k):
    """
    Exact dynamic-programming segmentation into k
    contiguous phases.
    """

    n = len(Z)

    dp = np.full((k + 1, n + 1), np.inf)
    prev = np.full((k + 1, n + 1), -1, dtype=int)

    dp[0, 0] = 0.0

    # Minimum one layer per phase.
    for p in range(1, k + 1):

        for j in range(p, n + 1):

            for i in range(p - 1, j):

                cost = (
                    dp[p - 1, i]
                    + segment_cost(Z, i, j)
                )

                if cost < dp[p, j]:

                    dp[p, j] = cost
                    prev[p, j] = i

    # Recover boundaries.
    boundaries = []

    j = n

    for p in range(k, 0, -1):

        i = prev[p, j]

        boundaries.append(i)

        j = i

    boundaries = sorted(boundaries)

    return dp[k, n], boundaries


print()
print("=" * 80)
print("CONTIGUOUS TEMPORAL SEGMENTATION")
print("=" * 80)

for k in [2, 3, 4, 5]:

    cost, boundaries = optimal_segmentation(Z, k)

    print()
    print(f"K = {k}")
    print(f"Total within-phase SSE = {cost:.4f}")

    start = 0

    for phase_id, end in enumerate(
        boundaries[1:] + [len(df)],
        start=1
    ):

        seg = Z[start:end]

        center = seg.mean(axis=0)

        variance = np.mean(
            np.sum(
                (seg - center) ** 2,
                axis=1
            )
        )

        print(
            f"  Phase {phase_id}: "
            f"layers {start+1}-{end}, "
            f"length={end-start}, "
            f"within_var={variance:.4f}"
        )

        start = end

# ============================================================
# 6. Save transition data
# ============================================================

out = ROOT / "results/resnet50_temporal_analysis.csv"

df.to_csv(out, index=False)

print()
print("=" * 80)
print("Saved:")
print(out)
print("=" * 80)
