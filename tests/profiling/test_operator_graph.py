"""Unit tests for the generic operator dependency graph."""

import json
import unittest
from dataclasses import replace

from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    TensorDependency,
    OperatorDependencyGraph,
    build_operator_dependency_graph,
)


class OperatorGraphTests(unittest.TestCase):

    def make_conv_gemm_graph(self):
        operators = [
            OperatorNode(
                "conv1", "Conv2D",
                ("image", "conv_weights"),
                ("features",),
            ),
            OperatorNode(
                "gemm1", "GEMM",
                ("features", "gemm_weights"),
                ("logits",),
            ),
        ]

        tensors = [
            TensorNode("image", (1, 3, 224, 224), "float32", is_external=True),
            TensorNode("conv_weights", (64, 3, 7, 7), "float32", is_external=True),
            TensorNode("features", (1, 64, 112, 112), "float32", "conv1"),
            TensorNode("gemm_weights", (1000, 64), "float32", is_external=True),
            TensorNode("logits", (1, 1000), "float32", "gemm1"),
        ]

        return build_operator_dependency_graph(
            workload_id="conv-gemm",
            operators=operators,
            tensors=tensors,
        )

    def assert_invalid(self, graph):
        with self.assertRaises(ValueError):
            graph.validate()

    def test_valid_conv_to_gemm_graph(self):
        graph = self.make_conv_gemm_graph()

        self.assertTrue(graph.validate())
        self.assertEqual(graph.operator_count, 2)
        self.assertEqual(graph.tensor_count, 5)
        self.assertEqual(graph.external_tensor_count, 3)
        self.assertEqual(graph.topological_order(), ("conv1", "gemm1"))
        self.assertEqual(len(graph.dependencies), 1)
        self.assertEqual(graph.dependencies[0].tensor_id, "features")

    def test_external_inputs_do_not_create_internal_edges(self):
        graph = self.make_conv_gemm_graph()

        self.assertEqual(
            {edge.tensor_id for edge in graph.dependencies},
            {"features"},
        )

    def test_branching_one_producer_multiple_consumers(self):
        operators = [
            OperatorNode("producer", "Conv2D", ("x",), ("shared",)),
            OperatorNode("consumer_a", "GEMM", ("shared",), ("a",)),
            OperatorNode("consumer_b", "GEMM", ("shared",), ("b",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("shared", producer_operator_id="producer"),
            TensorNode("a", producer_operator_id="consumer_a"),
            TensorNode("b", producer_operator_id="consumer_b"),
        ]

        graph = build_operator_dependency_graph(
            workload_id="branching",
            operators=operators,
            tensors=tensors,
        )

        self.assertTrue(graph.validate())
        self.assertEqual(len(graph.dependencies), 2)
        self.assertEqual(
            {edge.consumer_operator_id for edge in graph.dependencies},
            {"consumer_a", "consumer_b"},
        )

    def test_merging_multiple_producers_into_one_consumer(self):
        operators = [
            OperatorNode("producer_a", "Conv2D", ("x",), ("a",)),
            OperatorNode("producer_b", "Conv2D", ("y",), ("b",)),
            OperatorNode("merge", "GEMM", ("a", "b"), ("out",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("y", is_external=True),
            TensorNode("a", producer_operator_id="producer_a"),
            TensorNode("b", producer_operator_id="producer_b"),
            TensorNode("out", producer_operator_id="merge"),
        ]

        graph = build_operator_dependency_graph(
            workload_id="merging",
            operators=operators,
            tensors=tensors,
        )

        self.assertTrue(graph.validate())
        self.assertEqual(len(graph.dependencies), 2)
        self.assertEqual(
            {edge.producer_operator_id for edge in graph.dependencies},
            {"producer_a", "producer_b"},
        )

    def test_multiple_tensors_between_same_operator_pair(self):
        operators = [
            OperatorNode("producer", "OpA", ("x",), ("a", "b")),
            OperatorNode("consumer", "OpB", ("a", "b"), ("out",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("a", producer_operator_id="producer"),
            TensorNode("b", producer_operator_id="producer"),
            TensorNode("out", producer_operator_id="consumer"),
        ]

        graph = build_operator_dependency_graph(
            workload_id="two-tensors",
            operators=operators,
            tensors=tensors,
        )

        self.assertEqual(len(graph.dependencies), 2)
        self.assertEqual(
            {edge.tensor_id for edge in graph.dependencies},
            {"a", "b"},
        )

    def test_independent_operators_are_supported(self):
        operators = [
            OperatorNode("left", "Conv2D", ("x",), ("a",)),
            OperatorNode("right", "GEMM", ("y",), ("b",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("y", is_external=True),
            TensorNode("a", producer_operator_id="left"),
            TensorNode("b", producer_operator_id="right"),
        ]

        graph = build_operator_dependency_graph(
            workload_id="independent",
            operators=operators,
            tensors=tensors,
        )

        self.assertTrue(graph.validate())
        self.assertEqual(len(graph.dependencies), 0)
        self.assertEqual(set(graph.topological_order()), {"left", "right"})

    def test_rejects_cycle(self):
        operators = [
            OperatorNode("a", "OpA", ("y",), ("x",)),
            OperatorNode("b", "OpB", ("x",), ("y",)),
        ]
        tensors = [
            TensorNode("x", producer_operator_id="a"),
            TensorNode("y", producer_operator_id="b"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="cycle",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_unknown_input_tensor(self):
        operators = [
            OperatorNode("a", "OpA", ("missing",), ("out",)),
        ]
        tensors = [
            TensorNode("out", producer_operator_id="a"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="unknown-input",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_unknown_producer(self):
        operators = [
            OperatorNode("consumer", "OpB", ("x",), ("out",)),
        ]
        tensors = [
            TensorNode("x", producer_operator_id="missing"),
            TensorNode("out", producer_operator_id="consumer"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="unknown-producer",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_producer_declaration_mismatch(self):
        operators = [
            OperatorNode("a", "OpA", ("x",), ("out",)),
            OperatorNode("b", "OpB", ("y",), ("other",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("y", is_external=True),
            TensorNode("out", producer_operator_id="b"),
            TensorNode("other", producer_operator_id="b"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="producer-mismatch",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_external_tensor_as_operator_output(self):
        operators = [
            OperatorNode("a", "OpA", ("x",), ("out",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("out", is_external=True),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="external-output",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_duplicate_operator_ids(self):
        operators = [
            OperatorNode("a", "OpA", ("x",), ("out1",)),
            OperatorNode("a", "OpB", ("x",), ("out2",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("out1", producer_operator_id="a"),
            TensorNode("out2", producer_operator_id="a"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="duplicate-operator",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_duplicate_tensor_ids(self):
        operators = [
            OperatorNode("a", "OpA", ("x",), ("out",)),
        ]
        tensors = [
            TensorNode("x", is_external=True),
            TensorNode("x", is_external=True),
            TensorNode("out", producer_operator_id="a"),
        ]

        with self.assertRaises(ValueError):
            build_operator_dependency_graph(
                workload_id="duplicate-tensor",
                operators=operators,
                tensors=tensors,
            )

    def test_rejects_missing_dependency(self):
        graph = self.make_conv_gemm_graph()
        graph.dependencies = ()

        self.assert_invalid(graph)

    def test_rejects_extra_dependency(self):
        graph = self.make_conv_gemm_graph()
        graph.dependencies += (
            TensorDependency(
                tensor_id="image",
                producer_operator_id="conv1",
                consumer_operator_id="gemm1",
            ),
        )

        self.assert_invalid(graph)

    def test_rejects_duplicate_dependency(self):
        graph = self.make_conv_gemm_graph()
        graph.dependencies += (graph.dependencies[0],)

        self.assert_invalid(graph)

    def test_json_serialization(self):
        graph = self.make_conv_gemm_graph()
        payload = json.loads(graph.to_json())

        self.assertEqual(payload["workload_id"], "conv-gemm")
        self.assertEqual(len(payload["operators"]), 2)
        self.assertEqual(len(payload["tensors"]), 5)
        self.assertEqual(len(payload["dependencies"]), 1)
        self.assertEqual(
            payload["dependencies"][0]["tensor_id"],
            "features",
        )

    def test_rejects_invalid_tensor_region(self):
        with self.assertRaises(ValueError):
            TensorDependency(
                tensor_id="x",
                producer_operator_id="a",
                consumer_operator_id="b",
                tensor_region={"N": (5, 2)},
            )

    def test_rejects_self_dependency(self):
        with self.assertRaises(ValueError):
            TensorDependency(
                tensor_id="x",
                producer_operator_id="a",
                consumer_operator_id="a",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
