import unittest
from types import SimpleNamespace

from simulator.scheduler.locality_policy import LocalityHistory


def features(inputs, weights, layer_id="layer_A"):
    return SimpleNamespace(
        layer_id=layer_id,
        input_coordinates=frozenset(inputs),
        weight_coordinates=frozenset(weights),
    )


class LocalityHistoryTests(unittest.TestCase):

    def test_empty_history_has_zero_score(self):
        history = LocalityHistory(["C1"])
        self.assertEqual(history.score("C1", features({1}, {2})), 0.0)

    def test_identical_footprints_have_maximum_score(self):
        history = LocalityHistory(["C1"])
        item = features({1, 2}, {3, 4})
        history.record("C1", item)
        self.assertAlmostEqual(history.score("C1", item), 1.0)

    def test_history_is_chiplet_specific(self):
        history = LocalityHistory(["C1", "C2"])
        item = features({1}, {2})
        history.record("C1", item)
        self.assertEqual(history.score("C2", item), 0.0)

    def test_different_layers_do_not_share_locality(self):
        history = LocalityHistory(["C1"])
        history.record("C1", features({1, 2}, {3, 4}, "layer_A"))
        candidate = features({1, 2}, {3, 4}, "layer_B")
        self.assertEqual(history.score("C1", candidate), 0.0)

    def test_recency_decay(self):
        history = LocalityHistory(["C1"], decay=0.5)
        item = features({1}, {2})
        history.record("C1", item)
        history.record("C1", features({3}, {4}))
        self.assertAlmostEqual(history.score("C1", item), 0.5)

    def test_history_window_is_bounded(self):
        history = LocalityHistory(["C1"], window_size=2)
        item = features({1}, {2})
        history.record("C1", item)
        history.record("C1", features({3}, {4}))
        history.record("C1", features({5}, {6}))
        self.assertEqual(len(history.history["C1"]), 2)
        self.assertEqual(history.score("C1", item), 0.0)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalityHistory(["C1"], window_size=0)

    def test_invalid_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalityHistory(["C1"], decay=0.0)


if __name__ == "__main__":
    unittest.main()
