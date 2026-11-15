import unittest
from dataclasses import replace
import sys

sys.path.insert(0, "tests/profiling")

from test_real_execution_group_integration import (
    RealExecutionGroupIntegrationTests as Fixtures,
)

from simulator.profiling.execution_group_graph import (
    LDGWorkReference,
    ExecutionGroup,
    build_execution_group_dependency_graph,
)
from simulator.profiling.tlep_schema import ProvenanceKind

from simulator.profiling.execution_group_profiler import (
    build_execution_group_profiles,
)


class ExecutionGroupProfilerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Fixtures.setUpClass()
        cls.fixtures = (
            ("conv", Fixtures.conv_ldg, Fixtures.conv_operator_graph),
            ("gemm", Fixtures.gemm_ldg, Fixtures.gemm_operator_graph),
        )

    def build_graph(self, ldg, operator_graph, groups_data):
        operator = operator_graph.operators[ldg.operator_id]
        groups = []

        for group_id, work_ids in groups_data:
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
                    grouping_level="test",
                    input_tensors=tuple(operator.input_tensors),
                    output_tensors=(ldg.output.output_tensor,),
                )
            )

        graph = build_execution_group_dependency_graph(
            workload_id=ldg.workload_id,
            grouping_id="profiler-tests",
            groups=groups,
            full_coverage_ldgs=(
                (ldg.workload_id, ldg.operator_id, ldg.group_id),
            ),
        )
        return graph

    def test_split_group_work_is_conserved(self):
        for name, ldg, operator_graph in self.fixtures:
            valid_ids = sorted(
                work_id for work_id, item in ldg.work_items.items()
                if item.is_valid
            )
            midpoint = len(valid_ids) // 2

            graph = self.build_graph(
                ldg,
                operator_graph,
                [
                    (f"{name}-a", valid_ids[:midpoint]),
                    (f"{name}-b", valid_ids[midpoint:]),
                ],
            )

            profiles = build_execution_group_profiles(
                execution_group_graph=graph,
                operator_graph=operator_graph,
                ldg_graphs=(ldg,),
            )

            total = sum(profile.work_count for profile in profiles.values())
            expected = sum(
                item.work_count for item in ldg.work_items.values()
                if item.is_valid
            )

            self.assertEqual(total, expected)

    def test_hierarchy_partition_counts_and_provenance(self):
        expected_counts = {
            "conv": {
                "dram": 1.0,
                "buffer": 48.0,
                "inter_pe_temporal": 1.0,
                "spatial": 1.0,
                "reg": 1.0,
            },
            "gemm": {
                "dram": 2.0,
                "buffer": 9.0,
                "inter_pe_temporal": 1.0,
                "spatial": 1.0,
                "reg": 1.0,
            },
        }

        expected_legacy_totals = {
            "conv": 52.0,
            "gemm": 14.0,
        }

        for name, ldg, operator_graph in self.fixtures:
            valid_ids = tuple(
                work_id for work_id, item in ldg.work_items.items()
                if item.is_valid
            )
            graph = self.build_graph(
                ldg,
                operator_graph,
                [(f"{name}-hierarchy", valid_ids)],
            )

            profile = build_execution_group_profiles(
                execution_group_graph=graph,
                operator_graph=operator_graph,
                ldg_graphs=(ldg,),
            )[f"{name}-hierarchy"]

            demands = profile.resource_signature.additional_demands

            self.assertEqual(
                demands["distinct_hierarchy_partitions"],
                expected_legacy_totals[name],
            )

            for level, expected in expected_counts[name].items():
                key = f"hierarchy_partitions.{level}"
                self.assertEqual(demands[key], expected)

                provenance_key = (
                    f"resource_signature.additional_demands.{key}"
                )
                self.assertIn(provenance_key, profile.provenance)
                self.assertEqual(
                    profile.provenance[provenance_key].kind,
                    ProvenanceKind.RECONSTRUCTED,
                )


    def test_missing_work_count_does_not_produce_mac_total(self):
        _, original, operator_graph = self.fixtures[0]

        ldg = replace(original, work_items=dict(original.work_items))
        first_id = next(
            work_id for work_id, item in ldg.work_items.items()
            if item.is_valid
        )
        ldg.work_items[first_id] = replace(
            ldg.work_items[first_id],
            work_count=None,
        )

        valid_ids = tuple(
            work_id for work_id, item in ldg.work_items.items()
            if item.is_valid
        )
        graph = self.build_graph(
            ldg, operator_graph, [("missing-count", valid_ids)]
        )

        profile = build_execution_group_profiles(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=(ldg,),
        )["missing-count"]

        self.assertIsNone(profile.work_count)
        self.assertEqual(profile.resource_signature.operation_counts, {})

    def test_unsupported_adapter_does_not_produce_mac_total(self):
        _, original, operator_graph = self.fixtures[1]

        ldg = replace(original, work_items=dict(original.work_items))
        first_id = next(
            work_id for work_id, item in ldg.work_items.items()
            if item.is_valid
        )
        item = ldg.work_items[first_id]
        metadata = dict(item.metadata)
        metadata["adapter"] = "unsupported_test_adapter"

        ldg.work_items[first_id] = replace(
            item,
            metadata=metadata,
        )

        valid_ids = tuple(
            work_id for work_id, item in ldg.work_items.items()
            if item.is_valid
        )
        graph = self.build_graph(
            ldg, operator_graph, [("unsupported-adapter", valid_ids)]
        )

        profile = build_execution_group_profiles(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=(ldg,),
        )["unsupported-adapter"]

        self.assertIsNone(profile.work_count)
        self.assertEqual(profile.resource_signature.operation_counts, {})


if __name__ == "__main__":
    unittest.main()
