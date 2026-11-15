"""Unit tests for generic Logical Dependency Graph invariants."""

import unittest
from dataclasses import replace

from simulator.profiling.logical_dependency_graph import (
    LDGEdge,
    LogicalWorkItem,
    PartitionNode,
    build_logical_dependency_graph,
)


class LogicalDependencyGraphTests(unittest.TestCase):

    def make_graph(self):
        items = [
            LogicalWorkItem(
                work_item_id="work:0",
                hierarchy_indices={"dram": 0, "buffer": 0},
                reduction_coordinates={"C": 0},
                is_valid=True,
                work_count=1,
            ),
            LogicalWorkItem(
                work_item_id="work:1",
                hierarchy_indices={"dram": 0, "buffer": 0},
                reduction_coordinates={"C": 1},
                is_valid=True,
                work_count=1,
            ),
            LogicalWorkItem(
                work_item_id="work:2",
                hierarchy_indices={"dram": 0, "buffer": 1},
                reduction_coordinates={"C": 2},
                is_valid=False,
                invalid_reason="padding",
            ),
        ]

        return build_logical_dependency_graph(
            workload_id="unit-test",
            operator_id="toy-op",
            group_id="toy-group",
            output_tensor="Y",
            output_coordinates={"M": 0},
            reduction_axes=("C",),
            reduction_operator="sum",
            hierarchy_levels=("dram", "buffer"),
            work_items=items,
        )

    def assert_invalid(self, graph):
        with self.assertRaises(ValueError):
            graph.validate()

    def test_valid_graph_with_padding_and_shared_prefix(self):
        graph = self.make_graph()

        self.assertTrue(graph.validate())
        self.assertEqual(graph.logical_slot_count, 3)
        self.assertEqual(graph.valid_work_item_count, 2)
        self.assertEqual(graph.invalid_work_item_count, 1)

        contains = [e for e in graph.edges if e.relation == "contains"]
        contributes = [e for e in graph.edges if e.relation == "contributes_to"]
        refinements = [e for e in graph.edges if e.relation == "refines"]

        self.assertEqual(len(contains), 3)
        self.assertEqual(len(contributes), 2)
        self.assertEqual(len(refinements), 2)
        self.assertFalse(any(e.source_id == "work:2" for e in contributes))

    def test_rejects_unsupported_relation(self):
        graph = self.make_graph()
        graph.edges += (LDGEdge("work:0", graph.output.output_id, "depends_on"),)
        self.assert_invalid(graph)

    def test_rejects_extra_contains_edge(self):
        graph = self.make_graph()
        graph.edges += (LDGEdge("unknown-partition", "work:0", "contains"),)
        self.assert_invalid(graph)

    def test_rejects_missing_ownership_edge(self):
        graph = self.make_graph()
        graph.edges = tuple(
            e for e in graph.edges
            if not (e.relation == "contains" and e.target_id == "work:0")
        )
        self.assert_invalid(graph)

    def test_rejects_extra_refinement_edge(self):
        graph = self.make_graph()
        graph.edges += (LDGEdge("unknown-parent", "unknown-child", "refines"),)
        self.assert_invalid(graph)

    def test_rejects_missing_refinement_edge(self):
        graph = self.make_graph()
        graph.edges = tuple(e for e in graph.edges if e.relation != "refines")
        self.assert_invalid(graph)

    def test_rejects_invalid_edge_endpoint(self):
        graph = self.make_graph()
        graph.edges += (
            LDGEdge("missing-work-item", graph.output.output_id, "contributes_to"),
        )
        self.assert_invalid(graph)

    def test_rejects_padding_work_item_contribution(self):
        graph = self.make_graph()
        graph.edges += (
            LDGEdge("work:2", graph.output.output_id, "contributes_to"),
        )
        self.assert_invalid(graph)

    def test_rejects_extra_partition(self):
        graph = self.make_graph()
        extra = PartitionNode(
            partition_id="partition:extra",
            level="buffer",
            prefix=(99, 99),
            parent_id=None,
            logical_slots=0,
            valid_slots=0,
            invalid_slots=0,
        )
        graph.partitions[extra.partition_id] = extra
        self.assert_invalid(graph)

    def test_rejects_partition_work_item_id_collision(self):
        graph = self.make_graph()
        partition_id = next(iter(graph.partitions))
        item = graph.work_items.pop("work:0")
        graph.work_items[partition_id] = replace(
            item, work_item_id=partition_id
        )
        self.assert_invalid(graph)

    def test_rejects_output_partition_id_collision(self):
        graph = self.make_graph()
        graph.output = replace(
            graph.output,
            output_id=next(iter(graph.partitions)),
        )
        self.assert_invalid(graph)

    def test_rejects_incorrect_partition_occupancy(self):
        graph = self.make_graph()
        partition_id = next(iter(graph.partitions))
        node = graph.partitions[partition_id]
        graph.partitions[partition_id] = replace(
            node,
            logical_slots=node.logical_slots + 1,
        )
        self.assert_invalid(graph)


if __name__ == "__main__":
    unittest.main(verbosity=2)
