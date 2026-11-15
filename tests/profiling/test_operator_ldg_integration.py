"""Integration tests connecting operator graphs with real operator LDGs.

The operator-level tensor interface is synthetic. Convolution and GEMM
LDGs are independently reconstructed from real Timeloop mapping artifacts.
This test does not claim a physically executable Conv-to-GEMM pipeline.
"""

import unittest
from pathlib import Path

import yaml

from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    build_operator_dependency_graph,
)
from simulator.profiling.mapping_parser import (
    load_mapping_and_workload,
    parse_mapping_file,
)
from simulator.profiling.cross_level_reduction import (
    build_padding_aware_reduction_group,
)
from simulator.profiling.convolution_ldg_adapter import (
    build_convolution_ldg,
)
from simulator.profiling.gemm_ldg_adapter import (
    build_gemm_ldg_from_mapping,
)


ROOT = Path(__file__).resolve().parents[2]
CONV_DIR = ROOT / "timeloop/outputs/conv1_reference"
GEMM_DIR = ROOT / "timeloop/outputs/gemm/gemm_12x10x18"

REQUIRED_ARTIFACTS = (
    CONV_DIR / "timeloop-mapper.map.yaml",
    CONV_DIR / "parsed-processed-input.yaml",
    GEMM_DIR / "timeloop-mapper.map.yaml",
    GEMM_DIR / "parsed-processed-input.yaml",
)


class OperatorLDGIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Required Timeloop artifacts are unavailable: "
                + ", ".join(missing)
            )

        cls.operator_graph = build_operator_dependency_graph(
            workload_id="synthetic_conv_to_gemm_integration",
            operators=[
                OperatorNode(
                    operator_id="conv1",
                    operator_type="convolution",
                    input_tensors=("image", "conv_weights"),
                    output_tensors=("conv_output_fixture",),
                ),
                OperatorNode(
                    operator_id="gemm0",
                    operator_type="gemm",
                    input_tensors=("conv_output_fixture", "gemm_weights"),
                    output_tensors=("gemm_output",),
                ),
            ],
            tensors=[
                TensorNode(
                    tensor_id="image",
                    is_external=True,
                    shape=(1, 3, 224, 224),
                ),
                TensorNode(
                    tensor_id="conv_weights",
                    is_external=True,
                    shape=(64, 3, 7, 7),
                ),
                TensorNode(
                    tensor_id="conv_output_fixture",
                    producer_operator_id="conv1",
                    metadata={
                        "synthetic_interface": True,
                        "note": "Not a shape-validated real pipeline edge",
                    },
                ),
                TensorNode(
                    tensor_id="gemm_weights",
                    is_external=True,
                ),
                TensorNode(
                    tensor_id="gemm_output",
                    producer_operator_id="gemm0",
                ),
            ],
        )

        conv_mapping = parse_mapping_file(
            CONV_DIR / "timeloop-mapper.map.yaml"
        )
        conv_problem = yaml.safe_load(
            (CONV_DIR / "parsed-processed-input.yaml").read_text()
        )["problem"]

        conv_group = build_padding_aware_reduction_group(
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
        cls.conv_ldg = build_convolution_ldg(conv_group)

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

    def test_operator_graph_has_exact_tensor_dependency(self):
        graph = self.operator_graph

        self.assertTrue(graph.validate())
        self.assertEqual(graph.operator_count, 2)
        self.assertEqual(graph.tensor_count, 5)
        self.assertEqual(graph.external_tensor_count, 3)
        self.assertEqual(graph.topological_order(), ("conv1", "gemm0"))

        self.assertEqual(len(graph.dependencies), 1)
        dependency = graph.dependencies[0]

        self.assertEqual(dependency.tensor_id, "conv_output_fixture")
        self.assertEqual(dependency.producer_operator_id, "conv1")
        self.assertEqual(dependency.consumer_operator_id, "gemm0")

        self.assertTrue(
            graph.tensors["conv_output_fixture"].metadata[
                "synthetic_interface"
            ]
        )

    def test_real_convolution_ldg_preserves_padding(self):
        graph = self.conv_ldg

        self.assertTrue(graph.validate())
        self.assertEqual(graph.workload_id, "resnet50")
        self.assertEqual(graph.operator_id, "conv1_reference")
        self.assertEqual(graph.logical_slot_count, 147)
        self.assertEqual(graph.valid_work_item_count, 48)
        self.assertEqual(graph.invalid_work_item_count, 99)

    def test_real_gemm_ldg_reconstructs_mapped_reduction(self):
        graph = self.gemm_ldg

        self.assertEqual(
            self.gemm_dimensions,
            {"M": 12, "N": 10, "K": 18},
        )
        self.assertTrue(graph.validate())
        self.assertEqual(graph.workload_id, "gemm_12x10x18")
        self.assertEqual(graph.operator_id, "gemm0")
        self.assertEqual(graph.logical_slot_count, 18)
        self.assertEqual(graph.valid_work_item_count, 18)
        self.assertEqual(graph.invalid_work_item_count, 0)

    def test_operator_and_intra_operator_graphs_remain_distinct(self):
        operator_graph = self.operator_graph
        conv_ldg = self.conv_ldg
        gemm_ldg = self.gemm_ldg

        self.assertNotEqual(
            operator_graph.workload_id,
            conv_ldg.workload_id,
        )
        self.assertNotEqual(
            conv_ldg.workload_id,
            gemm_ldg.workload_id,
        )

        self.assertEqual(conv_ldg.operator_id, "conv1_reference")
        self.assertEqual(gemm_ldg.operator_id, "gemm0")

        # The operator dependency is tensor-level. It is not an LDG edge
        # and does not establish physical execution or synchronization.
        self.assertEqual(len(operator_graph.dependencies), 1)
        self.assertTrue(conv_ldg.validate())
        self.assertTrue(gemm_ldg.validate())


if __name__ == "__main__":
    unittest.main()
