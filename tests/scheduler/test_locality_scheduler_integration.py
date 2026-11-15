import unittest
from types import SimpleNamespace

from simulator.profiling.compact_execution_groups import (
    CompactExecutionGroup,
    CompactExecutionGroupSet,
)
from simulator.profiling.compact_resource_estimator import (
    CompactGroupResourceProfile,
)
from simulator.scheduler.compact_group_scheduler import simulate_compact_groups
from simulator.hardware.model import Chiplet, HardwareSystem


SCALES = {
    "input_bw": 1.0,
    "weight_bw": 1.0,
    "output_bw": 1.0,
    "dram_input_bw": 1.0,
    "dram_weight_bw": 1.0,
    "dram_output_bw": 1.0,
}


def make_workload():
    group = CompactExecutionGroup(
        group_id="g0",
        output_origin=(0, 0, 0, 0, 0),
        output_shape=(1, 1, 1, 1, 1),
        spatial_factors=(1, 1, 1, 1, 1),
        logical_slots=1,
        valid_contributions=1,
        padding_slots=0,
    )
    group_set = CompactExecutionGroupSet(
        layer_id="layer0",
        output_axes=("N", "M", "Q", "P", "G"),
        dimensions=(1, 1, 1, 1, 1, 1, 1, 1),
        spatial_factors=(1, 1, 1, 1, 1),
        groups=(group,),
        total_output_coordinates=1,
        total_logical_slots=1,
        total_valid_contributions=1,
        total_padding_slots=0,
    )
    profile = CompactGroupResourceProfile(
        group_id="g0",
        layer_id="layer0",
        estimated_compute_work=100.0,
        reference_cycles=100.0,
        estimated_input_accesses=1.0,
        estimated_weight_accesses=1.0,
        estimated_output_accesses=1.0,
        input_bw=0.1,
        weight_bw=0.1,
        output_bw=0.1,
        dram_input_bw=0.1,
        dram_weight_bw=0.1,
        dram_output_bw=0.1,
        estimation_method="synthetic_test",
    )
    workload = {"synthetic": [(group_set, (profile,))]}
    feature = SimpleNamespace(
        layer_id="layer0",
        group_id="g0",
        input_coordinates=frozenset({("I", 0)}),
        weight_coordinates=frozenset({("W", 0)}),
    )
    features = {"synthetic": [(feature,)]}
    return workload, features


class LocalitySchedulerIntegrationTests(unittest.TestCase):
    def test_locality_policy_completes_synthetic_workload(self):
        workload, features = make_workload()
        result = simulate_compact_groups(
            workload_layers=workload,
            policy="locality_aware",
            calibration_scales=SCALES,
            locality_features=features,
        )
        self.assertEqual(result.completed_groups, 1)
        self.assertGreater(result.makespan, 0)
        self.assertEqual(sum(result.chiplet_group_counts.values()), 1)
        self.assertEqual(result.policy, "locality_aware")

    def test_locality_policy_rejects_missing_features(self):
        workload, _ = make_workload()
        with self.assertRaisesRegex(ValueError, "locality_features are required"):
            simulate_compact_groups(
                workload_layers=workload,
                policy="locality_aware",
                calibration_scales=SCALES,
            )

    def test_locality_policy_rejects_mismatched_group_identity(self):
        workload, features = make_workload()
        features["synthetic"] = [(
            SimpleNamespace(
                layer_id="layer0",
                group_id="wrong_group",
                input_coordinates=frozenset({("I", 0)}),
                weight_coordinates=frozenset({("W", 0)}),
            ),
        )]
        with self.assertRaisesRegex(ValueError, "group_id mismatch"):
            simulate_compact_groups(
                workload_layers=workload,
                policy="locality_aware",
                calibration_scales=SCALES,
                locality_features=features,
            )

    def test_locality_policy_rejects_invalid_tolerance(self):
        workload, features = make_workload()
        with self.assertRaisesRegex(ValueError, "locality_tolerance"):
            simulate_compact_groups(
                workload_layers=workload,
                policy="locality_aware",
                calibration_scales=SCALES,
                locality_features=features,
                locality_tolerance=1.5,
            )


    def test_locality_overlap_influences_chiplet_placement(self):
        groups = tuple(
            CompactExecutionGroup(
                group_id=f"g{i}",
                output_origin=(0, 0, 0, i, 0),
                output_shape=(1, 1, 1, 1, 1),
                spatial_factors=(1, 1, 1, 1, 1),
                logical_slots=1,
                valid_contributions=1,
                padding_slots=0,
            )
            for i in range(3)
        )
        group_set = CompactExecutionGroupSet(
            layer_id="layer_locality",
            output_axes=("N", "M", "Q", "P", "G"),
            dimensions=(1, 1, 1, 3, 1, 1, 1, 1),
            spatial_factors=(1, 1, 1, 1, 1),
            groups=groups,
            total_output_coordinates=3,
            total_logical_slots=3,
            total_valid_contributions=3,
            total_padding_slots=0,
        )
        profiles = tuple(
            CompactGroupResourceProfile(
                group_id=f"g{i}",
                layer_id="layer_locality",
                estimated_compute_work=100.0,
                reference_cycles=100.0,
                estimated_input_accesses=1.0,
                estimated_weight_accesses=1.0,
                estimated_output_accesses=1.0,
                input_bw=0.1,
                weight_bw=0.1,
                output_bw=0.1,
                dram_input_bw=0.1,
                dram_weight_bw=0.1,
                dram_output_bw=0.1,
                estimation_method="synthetic_test",
            )
            for i in range(3)
        )
        workload = {"synthetic": [(group_set, profiles)]}

        feature_0 = SimpleNamespace(
            layer_id="layer_locality",
            group_id="g0",
            input_coordinates=frozenset({("I", 0), ("I", 1)}),
            weight_coordinates=frozenset({("W", 0), ("W", 1)}),
        )
        feature_1 = SimpleNamespace(
            layer_id="layer_locality",
            group_id="g1",
            input_coordinates=frozenset({("I", 2), ("I", 3)}),
            weight_coordinates=frozenset({("W", 2), ("W", 3)}),
        )
        feature_2 = SimpleNamespace(
            layer_id="layer_locality",
            group_id="g2",
            input_coordinates=frozenset({("I", 0), ("I", 1)}),
            weight_coordinates=frozenset({("W", 0), ("W", 1)}),
        )
        features = {
            "synthetic": [(feature_0, feature_1, feature_2)]
        }

        chiplets = {
            name: Chiplet(
                name=name,
                kind="test",
                compute_capacity=1.0,
                memory_capacity=1.0,
                memory_bandwidth=1.0,
                sram_capacity_words=4096,
            )
            for name in ("C1", "C2")
        }
        hardware = HardwareSystem(
            name="two_equal_chiplets",
            chiplets=chiplets,
            nop_topology="mesh",
            nop_bandwidth=1.0,
            nop_hop_latency=1.0,
        )

        result = simulate_compact_groups(
            workload_layers=workload,
            policy="locality_aware",
            hardware=hardware,
            calibration_scales=SCALES,
            locality_features=features,
            locality_tolerance=0.05,
        )

        counts = result.workload_chiplet_counts["synthetic"]
        self.assertEqual(result.completed_groups, 3)
        self.assertEqual(counts["C1"], 2)
        self.assertEqual(counts["C2"], 1)


    def test_lazy_provider_extracts_each_pending_group_once(self):
        workload, _ = make_workload()
        calls = []

        def provider(workload_name, layer_index, group_index, group_set, group):
            calls.append((
                workload_name, layer_index, group_index, group.group_id
            ))
            return SimpleNamespace(
                layer_id=group_set.layer_id,
                group_id=group.group_id,
                input_coordinates=frozenset({("I", 0)}),
                weight_coordinates=frozenset({("W", 0)}),
            )

        result = simulate_compact_groups(
            workload_layers=workload,
            policy="locality_aware",
            calibration_scales=SCALES,
            locality_feature_provider=provider,
        )

        self.assertEqual(result.completed_groups, 1)
        self.assertEqual(
            calls,
            [("synthetic", 0, 0, "g0")],
        )


if __name__ == "__main__":
    unittest.main()
