import unittest
from pathlib import Path

from simulator.profiling.mapping_parser import parse_mapping_file
from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    build_operator_dependency_graph,
)
from simulator.profiling.gemm_spatial_execution_groups import (
    build_gemm_spatial_execution_groups,
)
from simulator.profiling.execution_group_profiler import (
    build_execution_group_profiles,
)
from simulator.profiling.execution_group_tensor_access import (
    build_execution_group_tensor_accesses,
)
from simulator.profiling.gemm_compact_resource_estimator import (
    estimate_gemm_execution_group_resources,
)
from simulator.profiling.gemm_scheduler_adapter import (
    adapt_gemm_execution_groups_for_scheduler,
)


class GEMMSchedulerAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload_id = "gemm_12x10x18"
        cls.dimensions = {"M": 12, "N": 10, "K": 18}
        operator_id = "gemm0"

        mapping = parse_mapping_file(
            Path("timeloop/outputs/gemm/gemm_12x10x18")
            / "timeloop-mapper.map.yaml"
        )
        operator_graph = build_operator_dependency_graph(
            workload_id=cls.workload_id,
            operators=(
                OperatorNode(
                    operator_id=operator_id,
                    operator_type="gemm",
                    input_tensors=("Inputs", "Weights"),
                    output_tensors=("C",),
                    attributes={"workload_dimensions": cls.dimensions},
                ),
            ),
            tensors=(
                TensorNode(
                    tensor_id="Inputs", shape=(12, 18), is_external=True
                ),
                TensorNode(
                    tensor_id="Weights", shape=(18, 10), is_external=True
                ),
                TensorNode(
                    tensor_id="C",
                    shape=(12, 10),
                    producer_operator_id=operator_id,
                ),
            ),
        )

        cls.graph, ldgs = build_gemm_spatial_execution_groups(
            mapping=mapping,
            workload_dimensions=cls.dimensions,
            workload_id=cls.workload_id,
            operator_graph=operator_graph,
            operator_id=operator_id,
        )
        group_profiles = build_execution_group_profiles(
            execution_group_graph=cls.graph,
            operator_graph=operator_graph,
            ldg_graphs=ldgs,
        )
        group_accesses = build_execution_group_tensor_accesses(
            execution_group_graph=cls.graph,
            operator_graph=operator_graph,
            ldg_graphs=ldgs,
        )
        layer_profile = {
            "reference_compute_work": 2160,
            "reference_execution_cycles": 144,
            "buffer_input_shared_bw": 4.50,
            "buffer_weight_shared_bw": 6.25,
            "buffer_output_shared_bw": 2.50,
            "dram_input_transfers": 216,
            "dram_weight_transfers": 180,
            "dram_output_transfers": 120,
        }
        cls.profiles = estimate_gemm_execution_group_resources(
            execution_group_graph=cls.graph,
            group_profiles=group_profiles,
            group_accesses=group_accesses,
            layer_profile=layer_profile,
        )

    def test_adapter_preserves_order_and_profile_identity(self):
        group_set, profiles = adapt_gemm_execution_groups_for_scheduler(
            execution_group_graph=self.graph,
            gemm_profiles=self.profiles,
        )
        self.assertEqual(group_set.layer_id, self.workload_id)
        self.assertEqual(len(group_set.groups), 8)
        self.assertEqual(len(profiles), 8)

        expected_ids = tuple(
            group.execution_group_id
            for group in self.graph.groups.values()
        )
        actual_ids = tuple(group.group_id for group in group_set.groups)
        profile_ids = tuple(profile.group_id for profile in profiles)

        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(profile_ids, expected_ids)

    def test_adapter_preserves_original_profile_objects(self):
        group_set, profiles = adapt_gemm_execution_groups_for_scheduler(
            execution_group_graph=self.graph,
            gemm_profiles=self.profiles,
        )
        for original, adapted in zip(self.profiles, profiles):
            self.assertIs(original, adapted)

    def test_adapter_rejects_profile_count_mismatch(self):
        with self.assertRaises(ValueError):
            adapt_gemm_execution_groups_for_scheduler(
                execution_group_graph=self.graph,
                gemm_profiles=self.profiles[:-1],
            )

    def test_adapter_rejects_profile_order_mismatch(self):
        with self.assertRaises(ValueError):
            adapt_gemm_execution_groups_for_scheduler(
                execution_group_graph=self.graph,
                gemm_profiles=tuple(reversed(self.profiles)),
            )


if __name__ == "__main__":
    unittest.main()
