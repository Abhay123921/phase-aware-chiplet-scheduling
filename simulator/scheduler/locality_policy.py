"""Bounded geometric-locality scoring for compact execution groups.

Geometric overlap represents potential reuse, not guaranteed cache hits.
"""

from collections import deque


class LocalityHistory:
    """Track recently dispatched group footprints independently per chiplet."""

    def __init__(self, chiplet_names, window_size=16, decay=0.8):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")

        self.window_size = window_size
        self.decay = decay
        self.history = {
            name: deque(maxlen=window_size)
            for name in chiplet_names
        }

    @staticmethod
    def _jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def score(self, chiplet_name, features):
        """Return recency-weighted input/weight geometric reuse potential."""
        score = 0.0
        entries = self.history[chiplet_name]

        for age, previous in enumerate(reversed(entries)):
            if features.layer_id != previous.layer_id:
                continue

            recency_weight = self.decay ** age

            input_overlap = self._jaccard(
                features.input_coordinates,
                previous.input_coordinates,
            )
            weight_overlap = self._jaccard(
                features.weight_coordinates,
                previous.weight_coordinates,
            )

            score += recency_weight * (
                0.5 * input_overlap + 0.5 * weight_overlap
            )

        return score

    def record(self, chiplet_name, features):
        """Record a dispatched group's geometric footprint."""
        if chiplet_name not in self.history:
            raise ValueError(f"Unknown chiplet: {chiplet_name}")
        self.history[chiplet_name].append(features)
