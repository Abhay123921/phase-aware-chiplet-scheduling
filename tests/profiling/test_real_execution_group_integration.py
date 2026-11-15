"""Integration tests connecting real Timeloop LDGs to candidate execution groups.

These tests validate logical references and provenance. They do not establish
physical execution, tile scheduling, concurrency, or execution timing.
"""

import unittest
from pathlib import Path

import yaml

from simulator.profiling.mapping_parser import (
    load_mapping_and_workload,
    parse_mapping_file,
)
from simulator.profiling.cross_level_reduction import (
    build_padding_aware_reduction_group,
)
from simulator.profiling.convolution_ldg_adapter import build_convolution_ldg
from simulator.profiling.gemm_ldg_adapter import build_gemm_ldg_from_mapping
from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    build_operator_dependency_graph,
)
from simulator.profiling.execution_group_graph import (
    LDGWorkReference,
    ExecutionGroup,
    build_execution_group_dependency_graph,
)
from simulator.profiling.execution_group_reduction import (
    ReductionParticipant,
    ExecutionGroupReductionRequirement,
    ReductionPartialResult,
    ReductionReadinessTracker,
)


ROOT = Path("timeloop/outputs")
CONV_DIR = ROOT / "conv1_reference"
GEMM_DIR = ROOT / "gemm" / "gemm_12x10x18"


def build_single_operator_fixture(
    *,
    workload_id,
    operator_id,
    operator_type,
    input_shapes,
    output_tensor,
    output_shape,
    workload_dimensions,
    timeloop_output_data_space,
    adapter_output_tensor,
):
    """Construct a workload-specific operator graph from Timeloop metadata."""

    input_names = tuple(input_shapes)

    operators = [
        OperatorNode(
            operator_id=operator_id,
            operator_type=operator_type,
            input_tensors=input_names,
            output_tensors=(output_tensor,),
            attributes={
                "workload_dimensions": dict(workload_dimensions),
                "timeloop_output_data_space": timeloop_output_data_space,
                "adapter_output_tensor": adapter_output_tensor,
                "interface_source": "Timeloop parsed-processed-input.yaml",
            },
        )
    ]

    tensors = [
        TensorNode(
            tensor_id=name,
            shape=tuple(shape),
            is_external=True,
            metadata={"timeloop_data_space": name},
        )
        for name, shape in input_shapes.items()
    ]

    tensors.append(
        TensorNode(
            tensor_id=output_tensor,
            shape=tuple(output_shape),
            producer_operator_id=operator_id,
            metadata={
                "timeloop_data_space": timeloop_output_data_space,
                "adapter_output_tensor": adapter_output_tensor,
                "name_correspondence": (
                    "adapter output name corresponds to Timeloop output "
                    "data-space name"
                ),
            },
        )
    )

    return build_operator_dependency_graph(
        workload_id=workload_id,
        operators=operators,
        tensors=tensors,
    )


def build_real_execution_group(*, ldg, operator_graph, grouping_id):
    """Reference every valid LDG item in one candidate logical group."""

    valid_ids = tuple(
        work_item_id
        for work_item_id, work_item in ldg.work_items.items()
        if work_item.is_valid
    )

    group = ExecutionGroup(
        execution_group_id=f"{ldg.operator_id}:{ldg.group_id}:candidate",
        operator_id=ldg.operator_id,
        ldg_references=(
            LDGWorkReference(
                workload_id=ldg.workload_id,
                operator_id=ldg.operator_id,
                ldg_group_id=ldg.group_id,
                work_item_ids=valid_ids,
            ),
        ),
        grouping_level="one-logical-output-reduction-group",
        input_tensors=tuple(
            operator_graph.operators[ldg.operator_id].input_tensors
        ),
        output_tensors=(ldg.output.output_tensor,),
        metadata={
            "interpretation": "candidate logical scheduling unit",
            "physical_execution_claim": False,
        },
    )

    graph = build_execution_group_dependency_graph(
        workload_id=ldg.workload_id,
        grouping_id=grouping_id,
        groups=(group,),
    )

    return graph, group, valid_ids


class RealExecutionGroupIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Real convolution mapping and problem definition.
        conv_mapping = parse_mapping_file(
            CONV_DIR / "timeloop-mapper.map.yaml"
        )
        conv_problem = yaml.safe_load(
            (CONV_DIR / "parsed-processed-input.yaml").read_text()
        )["problem"]

        conv_instance = conv_problem["instance"]

        conv_reduction_group = build_padding_aware_reduction_group(
            mapping=conv_mapping,
            problem=conv_problem,
            workload_id="resnet50",
            operator_id="conv1_reference",
            output_coordinates={
                "N": 0,
                "M": 0,
                "Q": 0,
                "P": 0,
                "G": 0,
            },
        )
        cls.conv_ldg = build_convolution_ldg(conv_reduction_group)

        cls.conv_operator_graph = build_single_operator_fixture(
            workload_id="resnet50",
            operator_id="conv1_reference",
            operator_type="convolution",
            input_shapes={
                "Inputs": (
                    conv_instance["N"],
                    conv_instance["C"],
                    conv_instance["H"],
                    conv_instance["W"],
                ),
                "Weights": (
                    conv_instance["C"],
                    conv_instance["M"],
                    conv_instance["R"],
                    conv_instance["S"],
                    conv_instance["G"],
                ),
            },
            output_tensor="Outputs",
            output_shape=(
                conv_instance["N"],
                conv_instance["M"],
                conv_instance["Q"],
                conv_instance["P"],
                conv_instance["G"],
            ),
            workload_dimensions=conv_instance,
            timeloop_output_data_space="Outputs",
            adapter_output_tensor="Outputs",
        )

        # Real GEMM mapping and problem definition.
        gemm_mapping, gemm_dimensions = load_mapping_and_workload(
            GEMM_DIR / "timeloop-mapper.map.yaml",
            GEMM_DIR / "parsed-processed-input.yaml",
        )
        cls.gemm_dimensions = gemm_dimensions

        cls.gemm_ldg = build_gemm_ldg_from_mapping(
            mapping=gemm_mapping,
            workload_dimensions=gemm_dimensions,
            workload_id="gemm_12x10x18",
            operator_id="gemm0",
            group_id="C_0_0",
            output_coordinates={"M": 0, "N": 0},
        )

        cls.gemm_operator_graph = build_single_operator_fixture(
            workload_id="gemm_12x10x18",
            operator_id="gemm0",
            operator_type="gemm",
            input_shapes={
                "Inputs": (
                    gemm_dimensions["M"],
                    gemm_dimensions["K"],
                ),
                "Weights": (
                    gemm_dimensions["K"],
                    gemm_dimensions["N"],
                ),
            },
            output_tensor="C",
            output_shape=(
                gemm_dimensions["M"],
                gemm_dimensions["N"],
            ),
            workload_dimensions=gemm_dimensions,
            timeloop_output_data_space="Outputs",
            adapter_output_tensor="C",
        )

    def test_real_convolution_ldg_maps_to_candidate_execution_group(self):
        ldg = self.conv_ldg
        operator_graph = self.conv_operator_graph

        self.assertTrue(ldg.validate())
        self.assertTrue(operator_graph.validate())

        graph, group, valid_ids = build_real_execution_group(
            ldg=ldg,
            operator_graph=operator_graph,
            grouping_id="conv1-reference-logical-output",
        )

        self.assertTrue(
            graph.validate(
                operator_graph=operator_graph,
                ldg_graphs=(ldg,),
            )
        )

        self.assertEqual(graph.workload_id, "resnet50")
        self.assertEqual(ldg.operator_id, "conv1_reference")
        self.assertEqual(ldg.output.output_tensor, "Outputs")

        self.assertEqual(ldg.logical_slot_count, 147)
        self.assertEqual(ldg.valid_work_item_count, 48)
        self.assertEqual(ldg.invalid_work_item_count, 99)

        self.assertEqual(len(valid_ids), 48)
        self.assertEqual(len(group.ldg_references), 1)
        self.assertEqual(
            len(group.ldg_references[0].work_item_ids),
            48,
        )
        self.assertEqual(group.output_tensors, ("Outputs",))
        self.assertEqual(
            group.input_tensors,
            ("Inputs", "Weights"),
        )

        referenced_ids = set(group.ldg_references[0].work_item_ids)
        invalid_ids = {
            work_item_id
            for work_item_id, item in ldg.work_items.items()
            if not item.is_valid
        }
        self.assertTrue(referenced_ids.isdisjoint(invalid_ids))

    def test_real_gemm_ldg_maps_to_candidate_execution_group(self):
        ldg = self.gemm_ldg
        operator_graph = self.gemm_operator_graph

        self.assertTrue(ldg.validate())
        self.assertTrue(operator_graph.validate())

        graph, group, valid_ids = build_real_execution_group(
            ldg=ldg,
            operator_graph=operator_graph,
            grouping_id="gemm-12x10x18-logical-output",
        )

        self.assertTrue(
            graph.validate(
                operator_graph=operator_graph,
                ldg_graphs=(ldg,),
            )
        )

        self.assertEqual(
            self.gemm_dimensions,
            {"M": 12, "N": 10, "K": 18},
        )
        self.assertEqual(graph.workload_id, "gemm_12x10x18")
        self.assertEqual(ldg.operator_id, "gemm0")
        self.assertEqual(ldg.group_id, "C_0_0")

        self.assertEqual(ldg.logical_slot_count, 18)
        self.assertEqual(ldg.valid_work_item_count, 18)
        self.assertEqual(ldg.invalid_work_item_count, 0)

        self.assertEqual(len(valid_ids), 18)
        self.assertEqual(
            len(group.ldg_references[0].work_item_ids),
            18,
        )
        self.assertEqual(group.output_tensors, ("C",))
        self.assertEqual(
            group.input_tensors,
            ("Inputs", "Weights"),
        )

        output_metadata = operator_graph.tensors["C"].metadata
        self.assertEqual(
            output_metadata["timeloop_data_space"],
            "Outputs",
        )
        self.assertEqual(
            output_metadata["adapter_output_tensor"],
            "C",
        )

    def _validate_two_way_reduction_partition(
        self, *, ldg, operator_graph, grouping_id, partition_predicate,
        expected_sizes,
    ):
        valid_items = {
            work_item_id: item
            for work_item_id, item in ldg.work_items.items()
            if item.is_valid
        }

        partitions = (
            {
                work_item_id
                for work_item_id, item in valid_items.items()
                if partition_predicate(item)
            },
            {
                work_item_id
                for work_item_id, item in valid_items.items()
                if not partition_predicate(item)
            },
        )

        self.assertEqual(
            tuple(len(partition) for partition in partitions),
            expected_sizes,
        )
        self.assertTrue(partitions[0].isdisjoint(partitions[1]))
        self.assertEqual(set.union(*partitions), set(valid_items))

        operator = operator_graph.operators[ldg.operator_id]
        groups = []
        for index, work_ids in enumerate(partitions):
            groups.append(
                ExecutionGroup(
                    execution_group_id=f"{grouping_id}:eg-{index}",
                    operator_id=ldg.operator_id,
                    ldg_references=(
                        LDGWorkReference(
                            workload_id=ldg.workload_id,
                            operator_id=ldg.operator_id,
                            ldg_group_id=ldg.group_id,
                            work_item_ids=tuple(sorted(work_ids)),
                        ),
                    ),
                    grouping_level="reduction-coordinate-partition",
                    input_tensors=tuple(operator.input_tensors),
                    output_tensors=(ldg.output.output_tensor,),
                    metadata={
                        "partition_index": index,
                        "physical_execution_claim": False,
                    },
                )
            )

        identity = (ldg.workload_id, ldg.operator_id, ldg.group_id)
        graph = build_execution_group_dependency_graph(
            workload_id=ldg.workload_id,
            grouping_id=grouping_id,
            groups=groups,
            full_coverage_ldgs=(identity,),
        )

        self.assertTrue(
            graph.validate(
                operator_graph=operator_graph,
                ldg_graphs=(ldg,),
            )
        )

        assigned = [
            work_id
            for group in graph.groups.values()
            for reference in group.ldg_references
            for work_id in reference.work_item_ids
        ]
        self.assertEqual(len(assigned), len(valid_items))
        self.assertEqual(set(assigned), set(valid_items))

        invalid_ids = {
            work_id
            for work_id, item in ldg.work_items.items()
            if not item.is_valid
        }
        self.assertTrue(set(assigned).isdisjoint(invalid_ids))
        return graph

    def test_real_convolution_two_group_reduction_partition_has_full_coverage(self):
        graph = self._validate_two_way_reduction_partition(
            ldg=self.conv_ldg,
            operator_graph=self.conv_operator_graph,
            grouping_id="conv1-reference-channel-partition",
            partition_predicate=lambda item: (
                item.reduction_coordinates["C"] == 0
            ),
            expected_sizes=(16, 32),
        )
        self.assertEqual(graph.group_count, 2)
        self.assertEqual(graph.summary()["full_coverage_ldg_count"], 1)

    def test_real_gemm_two_group_reduction_partition_has_full_coverage(self):
        graph = self._validate_two_way_reduction_partition(
            ldg=self.gemm_ldg,
            operator_graph=self.gemm_operator_graph,
            grouping_id="gemm-12x10x18-k-partition",
            partition_predicate=lambda item: (
                item.reduction_coordinates["K"] < 9
            ),
            expected_sizes=(9, 9),
        )
        self.assertEqual(graph.group_count, 2)
        self.assertEqual(graph.summary()["full_coverage_ldg_count"], 1)

    def _assert_reduction_contract(self, *, ldg, operator_graph,
                                   grouping_id, partition_predicate,
                                   expected_sizes, requirement_id):
        graph = self._validate_two_way_reduction_partition(
            ldg=ldg,
            operator_graph=operator_graph,
            grouping_id=grouping_id,
            partition_predicate=partition_predicate,
            expected_sizes=expected_sizes,
        )

        participants = tuple(
            ReductionParticipant(
                execution_group_id=group.execution_group_id,
                work_item_ids=group.ldg_references[0].work_item_ids,
            )
            for group in sorted(
                graph.groups.values(),
                key=lambda item: item.execution_group_id,
            )
        )

        requirement = ExecutionGroupReductionRequirement(
            requirement_id=requirement_id,
            workload_id=ldg.workload_id,
            operator_id=ldg.operator_id,
            ldg_group_id=ldg.group_id,
            output_tensor=ldg.output.output_tensor,
            output_coordinates=dict(ldg.output.output_coordinates),
            reduction_axes=tuple(ldg.output.reduction_axes),
            reduction_operator=ldg.output.reduction_operator,
            expected_contributors=ldg.output.expected_contributors,
            participants=participants,
        )

        self.assertTrue(requirement.validate_against(graph, (ldg,)))
        self.assertEqual(graph.dependency_count, 0)

        tracker = ReductionReadinessTracker(requirement)
        self.assertFalse(tracker.is_ready)
        self.assertEqual(tracker.completed_group_count, 0)
        self.assertEqual(tracker.expected_group_count, 2)
        self.assertEqual(len(tracker.missing_group_ids), 2)

        first, second = participants

        first_result = ReductionPartialResult(
            requirement_id=requirement_id,
            execution_group_id=first.execution_group_id,
            work_item_ids=first.work_item_ids,
        )
        self.assertFalse(tracker.submit(first_result))
        self.assertEqual(tracker.completed_group_count, 1)
        self.assertFalse(tracker.is_ready)

        # An incomplete result must be rejected without changing tracker state.
        incomplete_result = ReductionPartialResult(
            requirement_id=requirement_id,
            execution_group_id=second.execution_group_id,
            work_item_ids=second.work_item_ids[:-1],
        )
        with self.assertRaises(ValueError):
            tracker.submit(incomplete_result)

        self.assertEqual(tracker.completed_group_count, 1)
        self.assertFalse(tracker.is_ready)

        second_result = ReductionPartialResult(
            requirement_id=requirement_id,
            execution_group_id=second.execution_group_id,
            work_item_ids=second.work_item_ids,
        )
        self.assertTrue(tracker.submit(second_result))
        self.assertTrue(tracker.is_ready)
        self.assertEqual(tracker.completed_group_count, 2)
        self.assertEqual(tracker.missing_group_ids, ())

        expected_ids = {
            work_id
            for work_id, item in ldg.work_items.items()
            if item.is_valid
        }
        self.assertEqual(tracker.completed_work_item_ids, expected_ids)

        # A participant cannot submit its completion more than once.
        with self.assertRaises(ValueError):
            tracker.submit(second_result)

        self.assertEqual(tracker.completed_group_count, 2)
        self.assertTrue(tracker.is_ready)

    def test_real_convolution_reduction_contract_and_readiness(self):
        self._assert_reduction_contract(
            ldg=self.conv_ldg,
            operator_graph=self.conv_operator_graph,
            grouping_id="conv1-reference-reduction-contract",
            partition_predicate=lambda item: (
                item.reduction_coordinates["C"] == 0
            ),
            expected_sizes=(16, 32),
            requirement_id="conv1-output-reduction",
        )

    def test_real_gemm_reduction_contract_and_readiness(self):
        self._assert_reduction_contract(
            ldg=self.gemm_ldg,
            operator_graph=self.gemm_operator_graph,
            grouping_id="gemm-reduction-contract",
            partition_predicate=lambda item: (
                item.reduction_coordinates["K"] < 9
            ),
            expected_sizes=(9, 9),
            requirement_id="gemm-output-reduction",
        )

    def _make_reduction_requirement(self, ldg, graph, requirement_id):
        participants = tuple(
            ReductionParticipant(
                execution_group_id=group.execution_group_id,
                work_item_ids=group.ldg_references[0].work_item_ids,
            )
            for group in sorted(
                graph.groups.values(),
                key=lambda item: item.execution_group_id,
            )
        )
        return ExecutionGroupReductionRequirement(
            requirement_id=requirement_id,
            workload_id=ldg.workload_id,
            operator_id=ldg.operator_id,
            ldg_group_id=ldg.group_id,
            output_tensor=ldg.output.output_tensor,
            output_coordinates=dict(ldg.output.output_coordinates),
            reduction_axes=tuple(ldg.output.reduction_axes),
            reduction_operator=ldg.output.reduction_operator,
            expected_contributors=ldg.output.expected_contributors,
            participants=participants,
        )

    def test_requirement_coordinates_are_immutable(self):
        graph = self._validate_two_way_reduction_partition(
            ldg=self.gemm_ldg,
            operator_graph=self.gemm_operator_graph,
            grouping_id="gemm-immutable-contract",
            partition_predicate=lambda item: (
                item.reduction_coordinates["K"] < 9
            ),
            expected_sizes=(9, 9),
        )
        requirement = self._make_reduction_requirement(
            self.gemm_ldg, graph, "gemm-immutable-output"
        )
        original = dict(requirement.output_coordinates)

        with self.assertRaises(TypeError):
            requirement.output_coordinates["M"] = 999

        self.assertEqual(dict(requirement.output_coordinates), original)

    def test_requirement_rejects_malformed_participant_type(self):
        with self.assertRaises(TypeError):
            ExecutionGroupReductionRequirement(
                requirement_id="malformed",
                workload_id="workload",
                operator_id="operator",
                ldg_group_id="ldg",
                output_tensor="output",
                output_coordinates={"M": 0},
                reduction_axes=("K",),
                reduction_operator="sum",
                expected_contributors=1,
                participants=(object(),),
            )

    def test_requirement_rejects_unlisted_group_referencing_source_ldg(self):
        graph = self._validate_two_way_reduction_partition(
            ldg=self.gemm_ldg,
            operator_graph=self.gemm_operator_graph,
            grouping_id="gemm-extra-reference",
            partition_predicate=lambda item: (
                item.reduction_coordinates["K"] < 9
            ),
            expected_sizes=(9, 9),
        )
        requirement = self._make_reduction_requirement(
            self.gemm_ldg, graph, "gemm-extra-reference-output"
        )

        original_group = next(iter(graph.groups.values()))
        extra_group = ExecutionGroup(
            execution_group_id="unlisted-extra-group",
            operator_id=original_group.operator_id,
            ldg_references=original_group.ldg_references,
            grouping_level="extra-reference-test",
            input_tensors=original_group.input_tensors,
            output_tensors=original_group.output_tensors,
        )
        graph.groups[extra_group.execution_group_id] = extra_group

        with self.assertRaisesRegex(ValueError, "exactly match"):
            requirement.validate_against(graph, (self.gemm_ldg,))

    def test_real_workloads_remain_independent(self):
        self.assertNotEqual(
            self.conv_ldg.workload_id,
            self.gemm_ldg.workload_id,
        )
        self.assertNotEqual(
            self.conv_ldg.operator_id,
            self.gemm_ldg.operator_id,
        )

        self.assertEqual(
            self.conv_operator_graph.workload_id,
            self.conv_ldg.workload_id,
        )
        self.assertEqual(
            self.gemm_operator_graph.workload_id,
            self.gemm_ldg.workload_id,
        )


if __name__ == "__main__":
    unittest.main()
