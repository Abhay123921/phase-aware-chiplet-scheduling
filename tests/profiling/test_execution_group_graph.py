"""Tests for execution-group dependency graph invariants."""

import json
import unittest

from simulator.profiling.logical_dependency_graph import (
    LogicalWorkItem,
    build_logical_dependency_graph,
)
from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    build_operator_dependency_graph,
)
from simulator.profiling.execution_group_graph import (
    LDGWorkReference,
    ExecutionGroup,
    ExecutionGroupDependency,
    build_execution_group_dependency_graph,
)


class ExecutionGroupGraphTests(unittest.TestCase):

    def make_operator_graph(self):
        operators = [
            OperatorNode(
                "conv", "Conv2D",
                ("image", "conv_weights"),
                ("features",),
            ),
            OperatorNode(
                "gemm", "GEMM",
                ("features", "gemm_weights"),
                ("logits",),
            ),
        ]

        tensors = [
            TensorNode("image", is_external=True),
            TensorNode("conv_weights", is_external=True),
            TensorNode("features", producer_operator_id="conv"),
            TensorNode("gemm_weights", is_external=True),
            TensorNode("logits", producer_operator_id="gemm"),
        ]

        return build_operator_dependency_graph(
            workload_id="conv-gemm",
            operators=operators,
            tensors=tensors,
        )

    def make_ldg(self, operator_id, group_id, output_tensor):
        items = [
            LogicalWorkItem(
                work_item_id=f"{group_id}:work:0",
                hierarchy_indices={"dram": 0, "buffer": 0},
                reduction_coordinates={"K": 0},
                is_valid=True,
                work_count=1,
            ),
            LogicalWorkItem(
                work_item_id=f"{group_id}:work:1",
                hierarchy_indices={"dram": 0, "buffer": 1},
                reduction_coordinates={"K": 1},
                is_valid=False,
                invalid_reason="padding",
                work_count=0,
            ),
        ]

        return build_logical_dependency_graph(
            workload_id="conv-gemm",
            operator_id=operator_id,
            group_id=group_id,
            output_tensor=output_tensor,
            output_coordinates={"M": 0},
            reduction_axes=("K",),
            reduction_operator="sum",
            hierarchy_levels=("dram", "buffer"),
            work_items=items,
        )

    def make_fixture(self):
        operator_graph = self.make_operator_graph()

        conv_ldg = self.make_ldg("conv", "conv-output-0", "features")
        gemm_ldg = self.make_ldg("gemm", "gemm-output-0", "logits")

        conv_group = ExecutionGroup(
            execution_group_id="conv-eg-0",
            operator_id="conv",
            ldg_references=(
                LDGWorkReference(
                    "conv-gemm", "conv", "conv-output-0",
                    ("conv-output-0:work:0",),
                ),
            ),
            grouping_level="logical-output",
            input_tensors=("image", "conv_weights"),
            output_tensors=("features",),
        )

        gemm_group = ExecutionGroup(
            execution_group_id="gemm-eg-0",
            operator_id="gemm",
            ldg_references=(
                LDGWorkReference(
                    "conv-gemm", "gemm", "gemm-output-0",
                    ("gemm-output-0:work:0",),
                ),
            ),
            grouping_level="logical-output",
            input_tensors=("features", "gemm_weights"),
            output_tensors=("logits",),
        )

        dependency = ExecutionGroupDependency(
            source_group_id="conv-eg-0",
            target_group_id="gemm-eg-0",
            tensor_id="features",
        )

        graph = build_execution_group_dependency_graph(
            workload_id="conv-gemm",
            grouping_id="output-granularity",
            groups=(conv_group, gemm_group),
            dependencies=(dependency,),
        )

        return graph, operator_graph, (conv_ldg, gemm_ldg)

    def test_valid_cross_operator_execution_group_graph(self):
        graph, operator_graph, ldgs = self.make_fixture()

        self.assertTrue(
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)
        )
        self.assertEqual(graph.group_count, 2)
        self.assertEqual(graph.dependency_count, 1)
        self.assertEqual(
            graph.topological_order(),
            ("conv-eg-0", "gemm-eg-0"),
        )

    def test_summary_and_json_serialization(self):
        graph, operator_graph, ldgs = self.make_fixture()
        graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

        summary = graph.summary()
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["dependency_count"], 1)

        decoded = json.loads(graph.to_json())
        self.assertEqual(decoded["grouping_id"], "output-granularity")
        self.assertEqual(len(decoded["groups"]), 2)
        self.assertEqual(len(decoded["dependencies"]), 1)

    def test_rejects_padding_work_item(self):
        graph, operator_graph, ldgs = self.make_fixture()
        conv_group = graph.groups["conv-eg-0"]

        bad_reference = LDGWorkReference(
            "conv-gemm", "conv", "conv-output-0",
            ("conv-output-0:work:1",),
        )

        graph.groups["conv-eg-0"] = ExecutionGroup(
            execution_group_id=conv_group.execution_group_id,
            operator_id=conv_group.operator_id,
            ldg_references=(bad_reference,),
            grouping_level=conv_group.grouping_level,
            input_tensors=conv_group.input_tensors,
            output_tensors=conv_group.output_tensors,
        )

        with self.assertRaisesRegex(ValueError, "padding"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_duplicate_work_item_ownership(self):
        graph, operator_graph, ldgs = self.make_fixture()
        original = graph.groups["conv-eg-0"]

        duplicate_group = ExecutionGroup(
            execution_group_id="conv-eg-1",
            operator_id="conv",
            ldg_references=original.ldg_references,
            grouping_level="logical-output",
            input_tensors=original.input_tensors,
            output_tensors=original.output_tensors,
        )
        graph.groups[duplicate_group.execution_group_id] = duplicate_group

        with self.assertRaisesRegex(ValueError, "multiple groups"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_unknown_ldg_work_item(self):
        graph, operator_graph, ldgs = self.make_fixture()
        original = graph.groups["conv-eg-0"]

        bad_reference = LDGWorkReference(
            "conv-gemm", "conv", "conv-output-0", ("missing-work-item",)
        )
        graph.groups["conv-eg-0"] = ExecutionGroup(
            execution_group_id=original.execution_group_id,
            operator_id=original.operator_id,
            ldg_references=(bad_reference,),
            grouping_level=original.grouping_level,
            input_tensors=original.input_tensors,
            output_tensors=original.output_tensors,
        )

        with self.assertRaisesRegex(ValueError, "Unknown LDG work item"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_unknown_ldg_reference(self):
        graph, operator_graph, ldgs = self.make_fixture()
        original = graph.groups["conv-eg-0"]

        bad_reference = LDGWorkReference(
            "conv-gemm", "conv", "missing-ldg", ("work:0",)
        )
        graph.groups["conv-eg-0"] = ExecutionGroup(
            execution_group_id=original.execution_group_id,
            operator_id=original.operator_id,
            ldg_references=(bad_reference,),
            grouping_level=original.grouping_level,
            input_tensors=original.input_tensors,
            output_tensors=original.output_tensors,
        )

        with self.assertRaisesRegex(ValueError, "Unknown LDG reference"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_dependency_without_operator_tensor_edge(self):
        graph, operator_graph, ldgs = self.make_fixture()

        graph.dependencies = (
            ExecutionGroupDependency(
                "conv-eg-0", "gemm-eg-0", "image"
            ),
        )

        with self.assertRaisesRegex(ValueError, "source-group output"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_group_tensor_not_declared_by_operator(self):
        graph, operator_graph, ldgs = self.make_fixture()
        original = graph.groups["conv-eg-0"]

        graph.groups["conv-eg-0"] = ExecutionGroup(
            execution_group_id=original.execution_group_id,
            operator_id=original.operator_id,
            ldg_references=original.ldg_references,
            grouping_level=original.grouping_level,
            input_tensors=("not-an-operator-input",),
            output_tensors=original.output_tensors,
        )

        with self.assertRaisesRegex(ValueError, "input not declared"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=ldgs)

    def test_rejects_intra_operator_dependency_in_v1(self):
        graph, operator_graph, ldgs = self.make_fixture()
        original = graph.groups["conv-eg-0"]

        second_ldg = self.make_ldg("conv", "conv-output-1", "features")

        second_group = ExecutionGroup(
            execution_group_id="conv-eg-1",
            operator_id="conv",
            ldg_references=(
                LDGWorkReference(
                    "conv-gemm", "conv", "conv-output-1",
                    ("conv-output-1:work:0",),
                ),
            ),
            grouping_level="logical-output",
            input_tensors=original.input_tensors,
            output_tensors=original.output_tensors,
        )
        graph.groups[second_group.execution_group_id] = second_group
        graph.dependencies = (
            ExecutionGroupDependency(
                "conv-eg-0", "conv-eg-1", "features"
            ),
        )

        with self.assertRaisesRegex(ValueError, "Intra-operator"):
            graph.validate(
                operator_graph=operator_graph,
                ldg_graphs=ldgs + (second_ldg,),
            )

    def test_builder_rejects_cycle(self):
        group_a = ExecutionGroup(
            "a", "op", (
                LDGWorkReference("w", "op", "g1", ("x",)),
            ), "test",
            output_tensors=("tensor",),
        )
        group_b = ExecutionGroup(
            "b", "op", (
                LDGWorkReference("w", "op", "g2", ("y",)),
            ), "test",
            output_tensors=("tensor",),
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            build_execution_group_dependency_graph(
                workload_id="w",
                grouping_id="cycle-test",
                groups=(group_a, group_b),
                dependencies=(
                    ExecutionGroupDependency("a", "b", "tensor"),
                    ExecutionGroupDependency("b", "a", "tensor"),
                ),
            )

    def test_rejects_duplicate_group_ids(self):
        graph, _, _ = self.make_fixture()
        group = graph.groups["conv-eg-0"]

        with self.assertRaisesRegex(ValueError, "Duplicate execution-group ID"):
            build_execution_group_dependency_graph(
                workload_id="conv-gemm",
                grouping_id="duplicate-test",
                groups=(group, group),
            )


    def make_coverage_fixture(self, include_second_group=True, declare_coverage=True):
        operator_graph = self.make_operator_graph()

        items = [
            LogicalWorkItem(
                work_item_id="coverage:work:0",
                hierarchy_indices={"dram": 0, "buffer": 0},
                reduction_coordinates={"K": 0},
                is_valid=True,
                work_count=1,
            ),
            LogicalWorkItem(
                work_item_id="coverage:work:1",
                hierarchy_indices={"dram": 0, "buffer": 1},
                reduction_coordinates={"K": 1},
                is_valid=True,
                work_count=1,
            ),
            LogicalWorkItem(
                work_item_id="coverage:padding:2",
                hierarchy_indices={"dram": 0, "buffer": 2},
                reduction_coordinates={"K": 2},
                is_valid=False,
                invalid_reason="padding",
                work_count=0,
            ),
        ]

        ldg = build_logical_dependency_graph(
            workload_id="conv-gemm",
            operator_id="conv",
            group_id="coverage-output",
            output_tensor="features",
            output_coordinates={"M": 0},
            reduction_axes=("K",),
            reduction_operator="sum",
            hierarchy_levels=("dram", "buffer"),
            work_items=items,
        )

        groups = [
            ExecutionGroup(
                execution_group_id="coverage-eg-0",
                operator_id="conv",
                ldg_references=(
                    LDGWorkReference(
                        "conv-gemm", "conv", "coverage-output",
                        ("coverage:work:0",),
                    ),
                ),
                grouping_level="reduction-partition",
                input_tensors=("image", "conv_weights"),
                output_tensors=("features",),
            )
        ]

        if include_second_group:
            groups.append(
                ExecutionGroup(
                    execution_group_id="coverage-eg-1",
                    operator_id="conv",
                    ldg_references=(
                        LDGWorkReference(
                            "conv-gemm", "conv", "coverage-output",
                            ("coverage:work:1",),
                        ),
                    ),
                    grouping_level="reduction-partition",
                    input_tensors=("image", "conv_weights"),
                    output_tensors=("features",),
                )
            )

        kwargs = {}
        if declare_coverage:
            kwargs["full_coverage_ldgs"] = (
                ("conv-gemm", "conv", "coverage-output"),
            )

        graph = build_execution_group_dependency_graph(
            workload_id="conv-gemm",
            grouping_id="coverage-test",
            groups=groups,
            **kwargs,
        )
        return graph, operator_graph, ldg

    def test_accepts_complete_multi_group_coverage_and_serializes_contract(self):
        graph, operator_graph, ldg = self.make_coverage_fixture()

        self.assertTrue(
            graph.validate(operator_graph=operator_graph, ldg_graphs=(ldg,))
        )
        self.assertEqual(graph.group_count, 2)
        self.assertEqual(graph.summary()["full_coverage_ldg_count"], 1)

        decoded = json.loads(graph.to_json())
        self.assertEqual(
            decoded["full_coverage_ldgs"],
            [["conv-gemm", "conv", "coverage-output"]],
        )

    def test_rejects_missing_valid_work_item_under_full_coverage(self):
        graph, operator_graph, ldg = self.make_coverage_fixture(
            include_second_group=False
        )

        with self.assertRaisesRegex(ValueError, "Full coverage failed.*1 valid work items missing"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=(ldg,))

    def test_rejects_unknown_full_coverage_ldg_identity(self):
        graph, operator_graph, ldg = self.make_coverage_fixture()
        graph.full_coverage_ldgs = (
            ("conv-gemm", "conv", "unknown-output"),
        )

        with self.assertRaisesRegex(ValueError, "Unknown full-coverage LDG identity"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=(ldg,))


    def test_allows_partial_grouping_without_full_coverage_declaration(self):
        graph, operator_graph, ldg = self.make_coverage_fixture(
            include_second_group=False,
            declare_coverage=False,
        )
        self.assertTrue(
            graph.validate(operator_graph=operator_graph, ldg_graphs=(ldg,))
        )
        self.assertEqual(graph.summary()["full_coverage_ldg_count"], 0)

    def test_duplicate_ownership_still_rejected_when_full_coverage_declared(self):
        graph, operator_graph, ldg = self.make_coverage_fixture()
        original = graph.groups["coverage-eg-0"]

        duplicate = ExecutionGroup(
            execution_group_id="coverage-eg-duplicate",
            operator_id=original.operator_id,
            ldg_references=original.ldg_references,
            grouping_level=original.grouping_level,
            input_tensors=original.input_tensors,
            output_tensors=original.output_tensors,
        )
        graph.groups[duplicate.execution_group_id] = duplicate

        with self.assertRaisesRegex(ValueError, "multiple groups"):
            graph.validate(operator_graph=operator_graph, ldg_graphs=(ldg,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
