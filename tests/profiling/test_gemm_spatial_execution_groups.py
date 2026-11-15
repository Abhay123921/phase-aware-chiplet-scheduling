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


class GEMMSpatialExecutionGroupTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workload_id = "gemm_12x10x18"
        cls.operator_id = "gemm0"
        cls.dimensions = {"M": 12, "N": 10, "K": 18}

        mapping_path = (
            Path("timeloop/outputs/gemm/gemm_12x10x18")
            / "timeloop-mapper.map.yaml"
        )
        cls.mapping = parse_mapping_file(mapping_path)

        cls.operator_graph = build_operator_dependency_graph(
            workload_id=cls.workload_id,
            operators=(
                OperatorNode(
                    operator_id=cls.operator_id,
                    operator_type="gemm",
                    input_tensors=("Inputs", "Weights"),
                    output_tensors=("C",),
                    attributes={
                        "workload_dimensions": cls.dimensions,
                        "interface_source": "Timeloop GEMM mapping",
                    },
                ),
            ),
            tensors=(
                TensorNode(
                    tensor_id="Inputs",
                    shape=(12, 18),
                    is_external=True,
                ),
                TensorNode(
                    tensor_id="Weights",
                    shape=(18, 10),
                    is_external=True,
                ),
                TensorNode(
                    tensor_id="C",
                    shape=(12, 10),
                    producer_operator_id=cls.operator_id,
                ),
            ),
        )

        cls.execution_graph, cls.ldgs = (
            build_gemm_spatial_execution_groups(
                mapping=cls.mapping,
                workload_dimensions=cls.dimensions,
                workload_id=cls.workload_id,
                operator_graph=cls.operator_graph,
                operator_id=cls.operator_id,
            )
        )

    def test_real_mapping_produces_expected_spatial_groups(self):
        self.assertEqual(len(self.ldgs), 12 * 10)
        self.assertEqual(self.execution_graph.group_count, 8)

        group_sizes = sorted(
            group.metadata["output_element_count"]
            for group in self.execution_graph.groups.values()
        )
        self.assertEqual(group_sizes, [15] * 8)

        self.assertEqual(
            self.execution_graph.summary()["full_coverage_ldg_count"],
            120,
        )

    def test_all_valid_mac_work_is_assigned_exactly_once(self):
        self.assertTrue(
            self.execution_graph.validate(
                operator_graph=self.operator_graph,
                ldg_graphs=self.ldgs,
            )
        )

        assigned = []
        for group in self.execution_graph.groups.values():
            for reference in group.ldg_references:
                assigned.extend(reference.work_item_ids)

        expected_valid_ids = [
            (ldg.workload_id, ldg.operator_id, ldg.group_id, work_item_id)
            for ldg in self.ldgs
            for work_item_id, item in ldg.work_items.items()
            if item.is_valid
        ]

        self.assertEqual(len(expected_valid_ids), 12 * 10 * 18)
        self.assertEqual(len(assigned), 12 * 10 * 18)

        actual_ownership = []
        for group in self.execution_graph.groups.values():
            for reference in group.ldg_references:
                actual_ownership.extend(
                    (
                        reference.workload_id,
                        reference.operator_id,
                        reference.ldg_group_id,
                        work_item_id,
                    )
                    for work_item_id in reference.work_item_ids
                )

        self.assertEqual(len(actual_ownership), len(set(actual_ownership)))
        self.assertEqual(set(actual_ownership), set(expected_valid_ids))

    def test_groups_are_labelled_as_logical_candidates(self):
        for group in self.execution_graph.groups.values():
            self.assertEqual(
                group.metadata["interpretation"],
                "candidate logical scheduling unit",
            )
            self.assertFalse(group.metadata["physical_execution_claim"])
            self.assertEqual(group.input_tensors, ("Inputs", "Weights"))
            self.assertEqual(group.output_tensors, ("C",))

    def test_invalid_spatial_target_is_rejected(self):
        with self.assertRaises(ValueError):
            build_gemm_spatial_execution_groups(
                mapping=self.mapping,
                workload_dimensions=self.dimensions,
                workload_id=self.workload_id,
                operator_graph=self.operator_graph,
                operator_id=self.operator_id,
                spatial_target="nonexistent_target",
            )


if __name__ == "__main__":
    unittest.main()


from simulator.profiling.execution_group_profiler import (
    build_execution_group_profiles,
)
from simulator.profiling.execution_group_tensor_access import (
    build_execution_group_tensor_accesses,
)


class GEMMSpatialProfilerIntegrationTests(GEMMSpatialExecutionGroupTests):

    def test_each_group_has_logical_mac_profile(self):
        profiles = build_execution_group_profiles(
            execution_group_graph=self.execution_graph,
            operator_graph=self.operator_graph,
            ldg_graphs=self.ldgs,
        )

        self.assertEqual(len(profiles), 8)

        for profile in profiles.values():
            self.assertEqual(profile.work_count, 270)
            self.assertEqual(profile.work_unit, "logical_MAC")
            self.assertEqual(
                profile.resource_signature.operation_counts["MAC"],
                270,
            )
            self.assertIsNone(profile.estimated_cycles)
            self.assertIsNone(profile.measured_cycles)

    def test_each_group_has_reconstructed_logical_tensor_accesses(self):
        accesses = build_execution_group_tensor_accesses(
            execution_group_graph=self.execution_graph,
            operator_graph=self.operator_graph,
            ldg_graphs=self.ldgs,
        )

        self.assertEqual(len(accesses), 8)

        for group_result in accesses.values():
            by_tensor = {
                access.tensor_name: access
                for access in group_result["tensor_accesses"]
            }

            self.assertEqual(by_tensor["Inputs"].read_elements, 270)
            self.assertEqual(by_tensor["Inputs"].unique_elements, 54)

            self.assertEqual(by_tensor["Weights"].read_elements, 270)
            self.assertEqual(by_tensor["Weights"].unique_elements, 90)

            self.assertEqual(by_tensor["C"].unique_elements, 15)
            self.assertIsNone(by_tensor["C"].read_elements)
            self.assertIsNone(by_tensor["C"].write_elements)

            self.assertEqual(
                group_result["diagnostics"]["valid_work_items"],
                270,
            )

            self.assertIsNone(by_tensor["Inputs"].read_bytes)
            self.assertIsNone(by_tensor["Weights"].read_bytes)
            self.assertIsNone(by_tensor["Inputs"].memory_level)
            self.assertIsNone(by_tensor["Weights"].memory_level)
