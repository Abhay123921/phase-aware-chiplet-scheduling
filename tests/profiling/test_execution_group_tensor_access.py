import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, "tests/profiling")

from test_real_execution_group_integration import (
    RealExecutionGroupIntegrationTests as Fixtures,
    build_real_execution_group,
)

from simulator.profiling.convolution_adapter import (
    ConvolutionProjectionAdapter,
)
from simulator.profiling.execution_group_graph import (
    LDGWorkReference,
    ExecutionGroup,
    build_execution_group_dependency_graph,
)
from simulator.profiling.execution_group_tensor_access import (
    build_execution_group_tensor_accesses,
)


class ExecutionGroupTensorAccessTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Fixtures.setUpClass()

        problem = yaml.safe_load(
            Path(
                "timeloop/outputs/conv1_reference/"
                "parsed-processed-input.yaml"
            ).read_text()
        )["problem"]

        cls.conv_adapter = ConvolutionProjectionAdapter.from_problem(problem)

    def build_profile(self, name, ldg, operator_graph):
        graph, group, _ = build_real_execution_group(
            ldg=ldg,
            operator_graph=operator_graph,
            grouping_id=f"{name}_tensor_access_test",
        )

        adapters = {}
        if name == "conv":
            adapters[(ldg.workload_id, ldg.operator_id)] = self.conv_adapter

        profile = build_execution_group_tensor_accesses(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=(ldg,),
            convolution_projection_adapters=adapters,
        )[group.execution_group_id]

        return profile

    def access_by_name(self, profile):
        return {
            access.tensor_name: access
            for access in profile["tensor_accesses"]
        }

    def test_convolution_access_counts_and_padding(self):
        profile = self.build_profile(
            "conv", Fixtures.conv_ldg, Fixtures.conv_operator_graph
        )
        accesses = self.access_by_name(profile)

        self.assertEqual(accesses["Inputs"].read_elements, 48)
        self.assertEqual(accesses["Weights"].read_elements, 48)
        self.assertEqual(accesses["Inputs"].unique_elements, 48)
        self.assertEqual(accesses["Weights"].unique_elements, 48)
        self.assertEqual(accesses["Outputs"].unique_elements, 1)

        diagnostics = profile["diagnostics"]
        self.assertEqual(diagnostics["source_ldg_logical_slots"], 147)
        self.assertEqual(diagnostics["source_ldg_invalid_work_items"], 99)
        self.assertEqual(diagnostics["source_ldg_padding_slots"], 99)
        self.assertEqual(diagnostics["valid_work_items"], 48)

    def test_gemm_access_counts_and_no_padding(self):
        profile = self.build_profile(
            "gemm", Fixtures.gemm_ldg, Fixtures.gemm_operator_graph
        )
        accesses = self.access_by_name(profile)

        self.assertEqual(accesses["Inputs"].read_elements, 18)
        self.assertEqual(accesses["Weights"].read_elements, 18)
        self.assertEqual(accesses["Inputs"].unique_elements, 18)
        self.assertEqual(accesses["Weights"].unique_elements, 18)
        self.assertEqual(accesses["C"].unique_elements, 1)

        diagnostics = profile["diagnostics"]
        self.assertEqual(diagnostics["source_ldg_logical_slots"], 18)
        self.assertEqual(diagnostics["source_ldg_invalid_work_items"], 0)
        self.assertEqual(diagnostics["source_ldg_padding_slots"], 0)

    def test_split_convolution_groups_conserve_read_contributions(self):
        ldg = Fixtures.conv_ldg
        operator_graph = Fixtures.conv_operator_graph
        operator = operator_graph.operators[ldg.operator_id]

        valid_ids = sorted(
            work_id
            for work_id, item in ldg.work_items.items()
            if item.is_valid
        )
        midpoint = len(valid_ids) // 2

        groups = []
        for group_id, work_ids in (
            ("conv-left", valid_ids[:midpoint]),
            ("conv-right", valid_ids[midpoint:]),
        ):
            groups.append(
                ExecutionGroup(
                    execution_group_id=group_id,
                    operator_id=ldg.operator_id,
                    ldg_references=(
                        LDGWorkReference(
                            workload_id=ldg.workload_id,
                            operator_id=ldg.operator_id,
                            ldg_group_id=ldg.group_id,
                            work_item_ids=tuple(work_ids),
                        ),
                    ),
                    grouping_level="split-test",
                    input_tensors=tuple(operator.input_tensors),
                    output_tensors=(ldg.output.output_tensor,),
                )
            )

        graph = build_execution_group_dependency_graph(
            workload_id=ldg.workload_id,
            grouping_id="conv_split_tensor_access_test",
            groups=groups,
            full_coverage_ldgs=(
                (ldg.workload_id, ldg.operator_id, ldg.group_id),
            ),
        )

        profiles = build_execution_group_tensor_accesses(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=(ldg,),
            convolution_projection_adapters={
                (ldg.workload_id, ldg.operator_id): self.conv_adapter
            },
        )

        input_total = sum(
            self.access_by_name(profile)["Inputs"].read_elements
            for profile in profiles.values()
        )
        weight_total = sum(
            self.access_by_name(profile)["Weights"].read_elements
            for profile in profiles.values()
        )

        self.assertEqual(input_total, 48)
        self.assertEqual(weight_total, 48)

    def test_unknown_physical_byte_counts_remain_unknown(self):
        profile = self.build_profile(
            "gemm", Fixtures.gemm_ldg, Fixtures.gemm_operator_graph
        )

        for access in profile["tensor_accesses"]:
            self.assertIsNone(access.element_size_bytes)
            self.assertIsNone(access.read_bytes)
            self.assertIsNone(access.write_bytes)
            self.assertIsNone(access.memory_level)

        for footprint in profile["memory_footprints"]:
            self.assertIsNone(footprint.bytes)
            self.assertEqual(footprint.memory_level, "logical_tensor")


if __name__ == "__main__":
    unittest.main()
