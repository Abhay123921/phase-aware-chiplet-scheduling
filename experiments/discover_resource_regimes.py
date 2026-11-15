import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/workspace/P04")

inp = ROOT / "results/resnet50_resource_signatures.csv"

df = pd.read_csv(inp)

# ============================================================
# Resource signature
# ============================================================

features = [
    "compute_intensity",
    "memory_access_per_op",
    "buffer_bandwidth",
    "dram_bandwidth",
    "dram_energy_fraction",
]

X = df[features].values.astype(float)

# Standardize globally so different units do not dominate.
mean = X.mean(axis=0)
std = X.std(axis=0)

Z = (X - mean) / std


# ============================================================
# Distance matrix
# ============================================================

def pairwise_distance(A):
    diff = A[:, None, :] - A[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


D = pairwise_distance(Z)


# ============================================================
# Simple K-means implementation
# ============================================================

def kmeans(X, k, iterations=100, seed=42):

    rng = np.random.default_rng(seed)

    n = len(X)

    indices = rng.choice(n, size=k, replace=False)

    centers = X[indices].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(iterations):

        distances = np.sqrt(
            ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        )

        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        for c in range(k):

            members = X[labels == c]

            if len(members) > 0:
                centers[c] = members.mean(axis=0)

    return labels, centers


# ============================================================
# Silhouette score
# ============================================================

def silhouette_score(X, labels):

    n = len(X)

    if len(np.unique(labels)) < 2:
        return -1.0

    D = pairwise_distance(X)

    scores = []

    for i in range(n):

        own = labels == labels[i]

        # Ignore self.
        own[i] = False

        if own.sum() == 0:
            a = 0.0
        else:
            a = D[i, own].mean()

        b_values = []

        for c in np.unique(labels):

            if c == labels[i]:
                continue

            members = labels == c

            if members.sum() > 0:
                b_values.append(D[i, members].mean())

        b = min(b_values)

        if max(a, b) == 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)

        scores.append(s)

    return float(np.mean(scores))


# ============================================================
# Discover regimes for K = 2..6
# ============================================================

print("=" * 80)
print("RESOURCE-REGIME DISCOVERY")
print("=" * 80)

print()
print("Layers:", len(df))

print()
print("Features:")
for f in features:
    print("  ", f)


results = []

for k in range(2, 7):

    labels, centers = kmeans(Z, k)

    score = silhouette_score(Z, labels)

    # Within-cluster SSE
    sse = 0.0

    for c in range(k):

        members = Z[labels == c]

        if len(members) > 0:

            center = members.mean(axis=0)

            sse += np.sum(
                (members - center) ** 2
            )

    results.append((k, score, sse, labels, centers))

    print()
    print("-" * 80)
    print(f"K = {k}")
    print(f"Silhouette score = {score:.4f}")
    print(f"Within-cluster SSE = {sse:.4f}")

    for c in range(k):

        members = np.where(labels == c)[0]

        print()
        print(
            f"Regime {c}: "
            f"{len(members)} layers"
        )

        print(
            "  layers:",
            (members + 1).tolist()
        )


# ============================================================
# Choose K using silhouette, but DO NOT treat it as truth.
# ============================================================

best = max(results, key=lambda x: x[1])

best_k = best[0]
best_score = best[1]
best_labels = best[3]
best_centers = best[4]

print()
print("=" * 80)
print("SELECTED RESOURCE-REGIME PARTITION")
print("=" * 80)

print()
print(
    f"Best silhouette K = {best_k}, "
    f"score = {best_score:.4f}"
)

# ============================================================
# Describe each regime in ORIGINAL units
# ============================================================

print()
print("=" * 80)
print("REGIME RESOURCE PROFILES")
print("=" * 80)

for c in range(best_k):

    members = np.where(best_labels == c)[0]

    print()
    print(
        f"Regime {c}: "
        f"{len(members)} layers"
    )

    for f in features:

        values = df.iloc[members][f]

        print(
            f"  {f:25s} "
            f"mean={values.mean():.4f} "
            f"min={values.min():.4f} "
            f"max={values.max():.4f}"
        )


# ============================================================
# Assign regime to every layer
# ============================================================

df["regime"] = best_labels


print()
print("=" * 80)
print("TEMPORAL REGIME SEQUENCE")
print("=" * 80)

for _, row in df.iterrows():

    print(
        f"Layer {int(row.layer_index):2d} "
        f"{row.layer_id:35s} "
        f"-> Regime {int(row.regime)}"
    )


# ============================================================
# Regime transition matrix
# ============================================================

transition = np.zeros((best_k, best_k), dtype=int)

for i in range(len(best_labels) - 1):

    a = best_labels[i]
    b = best_labels[i + 1]

    transition[a, b] += 1


print()
print("=" * 80)
print("REGIME TRANSITION MATRIX")
print("=" * 80)

print()
print("Rows = current regime")
print("Cols = next regime")
print()

print("      " + " ".join(f"{c:>8}" for c in range(best_k)))

for i in range(best_k):

    print(
        f"{i:>4}  "
        + " ".join(
            f"{transition[i,j]:8d}"
            for j in range(best_k)
        )
    )


# ============================================================
# Transition probabilities
# ============================================================

print()
print("=" * 80)
print("REGIME TRANSITION PROBABILITIES")
print("=" * 80)

prob = np.zeros_like(
    transition,
    dtype=float
)

for i in range(best_k):

    total = transition[i].sum()

    if total > 0:
        prob[i] = transition[i] / total

print()

print("      " + " ".join(f"{c:>8}" for c in range(best_k)))

for i in range(best_k):

    print(
        f"{i:>4}  "
        + " ".join(
            f"{prob[i,j]:8.3f}"
            for j in range(best_k)
        )
    )


# ============================================================
# Persistence / run lengths
# ============================================================

print()
print("=" * 80)
print("REGIME PERSISTENCE")
print("=" * 80)

runs = []

start = 0

for i in range(1, len(best_labels) + 1):

    if (
        i == len(best_labels)
        or best_labels[i] != best_labels[start]
    ):

        regime = best_labels[start]

        length = i - start

        runs.append(
            (regime, start + 1, i, length)
        )

        start = i

for regime, begin, end, length in runs:

    print(
        f"Regime {regime}: "
        f"layers {begin}-{end}, "
        f"length={length}"
    )

print()
print("Run-length statistics:")

for c in range(best_k):

    lengths = [
        r[3]
        for r in runs
        if r[0] == c
    ]

    if lengths:

        print(
            f"  Regime {c}: "
            f"count={len(lengths)}, "
            f"mean={np.mean(lengths):.2f}, "
            f"max={max(lengths)}, "
            f"min={min(lengths)}"
        )


# ============================================================
# Save result
# ============================================================

out = ROOT / "results/resnet50_resource_regimes.csv"

df.to_csv(out, index=False)

print()
print("=" * 80)
print("Saved:")
print(out)
print("=" * 80)
