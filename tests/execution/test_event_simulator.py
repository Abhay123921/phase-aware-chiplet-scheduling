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
from simulator.profiling.execution_group_reduction import (
    ReductionParticipant,
    ExecutionGroupReductionRequirement,
)
from simulator.execution.event_simulator import (
    DependencyAwareEventSimulator,
    WorkloadExecutionSpec,
)


def build_workload(workload_id, arrival_time=0.0):
    """Build a valid two-participant reduction followed by a consumer."""

    producer_ldg = build_logical_dependency_graph(
        workload_id=workload_id,
        operator_id="producer",
        group_id="producer-output",
        output_tensor="mid",
        output_coordinates={"M": 0},
        reduction_axes=("K",),
        reduction_operator="sum",
        hierarchy_levels=("dram",),
        work_items=(
            LogicalWorkItem(
                work_item_id="k0",
                hierarchy_indices={"dram": 0},
                reduction_coordinates={"K": 0},
                is_valid=True,
            ),
            LogicalWorkItem(
                work_item_id="k1",
                hierarchy_indices={"dram": 1},
                reduction_coordinates={"K": 1},
                is_valid=True,
            ),
        ),
    )

    consumer_ldg = build_logical_dependency_graph(
        workload_id=workload_id,
        operator_id="consumer",
        group_id="consumer-output",
        output_tensor="final",
        output_coordinates={"M": 0},
        reduction_axes=("K",),
        reduction_operator="sum",
        hierarchy_levels=("dram",),
        work_items=(
            LogicalWorkItem(
                work_item_id="consumer-work",
                hierarchy_indices={"dram": 0},
                reduction_coordinates={"K": 0},
                is_valid=True,
            ),
        ),
    )

    operator_graph = build_operator_dependency_graph(
        workload_id=workload_id,
        operators=(
            OperatorNode(
                operator_id="producer",
                operator_type="compute",
                input_tensors=("input",),
                output_tensors=("mid",),
            ),
            OperatorNode(
                operator_id="consumer",
                operator_type="compute",
                input_tensors=("mid",),
                output_tensors=("final",),
            ),
        ),
        tensors=(
            TensorNode(tensor_id="input", is_external=True),
            TensorNode(
                tensor_id="mid",
                producer_operator_id="producer",
            ),
            TensorNode(
                tensor_id="final",
                producer_operator_id="consumer",
            ),
        ),
    )

    groups = (
        ExecutionGroup(
            execution_group_id="p0",
            operator_id="producer",
            ldg_references=(
                LDGWorkReference(
                    workload_id, "producer", "producer-output", ("k0",)
                ),
            ),
            grouping_level="reduction-partition",
            input_tensors=("input",),
            output_tensors=("mid",),
        ),
        ExecutionGroup(
            execution_group_id="p1",
            operator_id="producer",
            ldg_references=(
                LDGWorkReference(
                    workload_id, "producer", "producer-output", ("k1",)
                ),
            ),
            grouping_level="reduction-partition",
            input_tensors=("input",),
            output_tensors=("mid",),
        ),
        ExecutionGroup(
            execution_group_id="c0",
            operator_id="consumer",
            ldg_references=(
                LDGWorkReference(
                    workload_id, "consumer", "consumer-output",
                    ("consumer-work",),
                ),
            ),
            grouping_level="operator-output",
            input_tensors=("mid",),
            output_tensors=("final",),
        ),
    )

    graph = build_execution_group_dependency_graph(
        workload_id=workload_id,
        grouping_id="synthetic-two-way-reduction",
        groups=groups,
        # The reduction contract, not ordinary group-to-group edges,
        # gates the consumer until both partial results are complete.
        dependencies=(),
        full_coverage_ldgs=(
            (workload_id, "producer", "producer-output"),
            (workload_id, "consumer", "consumer-output"),
        ),
    )

    requirement = ExecutionGroupReductionRequirement(
        requirement_id="mid-reduction",
        workload_id=workload_id,
        operator_id="producer",
        ldg_group_id="producer-output",
        output_tensor="mid",
        output_coordinates={"M": 0},
        reduction_axes=("K",),
        reduction_operator="sum",
        expected_contributors=2,
        participants=(
            ReductionParticipant("p0", ("k0",)),
            ReductionParticipant("p1", ("k1",)),
        ),
    )

    return WorkloadExecutionSpec(
        workload_id=workload_id,
        operator_graph=operator_graph,
        execution_group_graph=graph,
        ldg_graphs=(producer_ldg, consumer_ldg),
        reduction_requirements=(requirement,),
        group_durations={"p0": 2.0, "p1": 3.0, "c0": 1.0},
        arrival_time=arrival_time,
        reduction_consumers={"mid-reduction": ("c0",)},
    )


class EventSimulatorIntegrationTests(unittest.TestCase):

    def test_reduction_barrier_releases_consumer_only_when_ready(self):
        workload = build_workload("A")

        result = DependencyAwareEventSimulator(
            (workload,), execution_slots=2
        ).run()

        self.assertEqual(result.group_start_times[("A", "p0")], 0.0)
        self.assertEqual(result.group_start_times[("A", "p1")], 0.0)

        self.assertEqual(result.group_completion_times[("A", "p0")], 2.0)
        self.assertEqual(result.group_completion_times[("A", "p1")], 3.0)

        self.assertEqual(
            result.reduction_ready_times[("A", "mid-reduction")], 3.0
        )
        self.assertEqual(result.group_start_times[("A", "c0")], 3.0)
        self.assertEqual(result.group_completion_times[("A", "c0")], 4.0)
        self.assertEqual(result.makespan, 4.0)

    def test_two_workloads_have_independent_group_identities(self):
        workloads = (
            build_workload("A"),
            build_workload("B"),
        )

        result = DependencyAwareEventSimulator(
            workloads, execution_slots=4
        ).run()

        for workload_id in ("A", "B"):
            self.assertEqual(
                result.group_start_times[(workload_id, "p0")], 0.0
            )
            self.assertEqual(
                result.group_start_times[(workload_id, "p1")], 0.0
            )
            self.assertEqual(
                result.reduction_ready_times[
                    (workload_id, "mid-reduction")
                ],
                3.0,
            )
            self.assertEqual(
                result.group_start_times[(workload_id, "c0")], 3.0
            )
            self.assertEqual(
                result.completion_times[workload_id], 4.0
            )

        self.assertEqual(result.makespan, 4.0)
        self.assertEqual(len(result.group_completion_times), 6)

    def test_later_arrival_is_respected(self):
        workloads = (
            build_workload("A", arrival_time=0.0),
            build_workload("B", arrival_time=1.0),
        )

        result = DependencyAwareEventSimulator(
            workloads, execution_slots=4
        ).run()

        self.assertEqual(result.group_start_times[("A", "p0")], 0.0)
        self.assertEqual(result.group_start_times[("B", "p0")], 1.0)
        self.assertEqual(result.group_start_times[("B", "p1")], 1.0)

        self.assertEqual(result.completion_times["A"], 4.0)
        self.assertEqual(result.completion_times["B"], 5.0)
        self.assertEqual(result.makespan, 5.0)

    def test_invalid_execution_slot_count_is_rejected(self):
        with self.assertRaises(ValueError):
            DependencyAwareEventSimulator(
                (build_workload("A"),), execution_slots=0
            )

    def test_incomplete_duration_map_is_rejected(self):
        workload = build_workload("A")
        invalid = WorkloadExecutionSpec(
            workload_id=workload.workload_id,
            operator_graph=workload.operator_graph,
            execution_group_graph=workload.execution_group_graph,
            ldg_graphs=workload.ldg_graphs,
            reduction_requirements=workload.reduction_requirements,
            group_durations={"p0": 2.0},
            arrival_time=workload.arrival_time,
            reduction_consumers=workload.reduction_consumers,
        )

        with self.assertRaises(ValueError):
            DependencyAwareEventSimulator((invalid,), execution_slots=2)


if __name__ == "__main__":
    unittest.main()
