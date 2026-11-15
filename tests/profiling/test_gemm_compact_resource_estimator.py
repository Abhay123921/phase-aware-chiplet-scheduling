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


class GEMMCompactResourceEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload_id = "gemm_12x10x18"
        cls.operator_id = "gemm0"
        cls.dimensions = {"M": 12, "N": 10, "K": 18}

        mapping_path = (
            Path("timeloop/outputs/gemm/gemm_12x10x18")
            / "timeloop-mapper.map.yaml"
        )
        mapping = parse_mapping_file(mapping_path)

        operator_graph = build_operator_dependency_graph(
            workload_id=cls.workload_id,
            operators=(
                OperatorNode(
                    operator_id=cls.operator_id,
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
                    producer_operator_id=cls.operator_id,
                ),
            ),
        )

        cls.execution_graph, cls.ldgs = build_gemm_spatial_execution_groups(
            mapping=mapping,
            workload_dimensions=cls.dimensions,
            workload_id=cls.workload_id,
            operator_graph=operator_graph,
            operator_id=cls.operator_id,
        )
        cls.group_profiles = build_execution_group_profiles(
            execution_group_graph=cls.execution_graph,
            operator_graph=operator_graph,
            ldg_graphs=cls.ldgs,
        )
        cls.group_accesses = build_execution_group_tensor_accesses(
            execution_group_graph=cls.execution_graph,
            operator_graph=operator_graph,
            ldg_graphs=cls.ldgs,
        )

        cls.layer_profile = {
            "reference_compute_work": 2160,
            "reference_execution_cycles": 144,
            "buffer_input_shared_bw": 4.50,
            "buffer_weight_shared_bw": 6.25,
            "buffer_output_shared_bw": 2.50,
            "dram_input_transfers": 216,
            "dram_weight_transfers": 180,
            "dram_output_transfers": 120,
        }

    def test_group_profiles_and_resource_conservation(self):
        results = estimate_gemm_execution_group_resources(
            execution_group_graph=self.execution_graph,
            group_profiles=self.group_profiles,
            group_accesses=self.group_accesses,
            layer_profile=self.layer_profile,
        )

        self.assertEqual(len(results), 8)
        self.assertAlmostEqual(
            sum(p.estimated_compute_work for p in results), 2160
        )
        self.assertAlmostEqual(
            sum(p.reference_cycles for p in results), 144
        )

        for profile in results:
            self.assertAlmostEqual(profile.estimated_compute_work, 270)
            self.assertAlmostEqual(profile.reference_cycles, 18)
            self.assertEqual(profile.estimated_input_accesses, 270)
            self.assertEqual(profile.estimated_weight_accesses, 270)
            self.assertEqual(profile.estimated_output_accesses, 15)

        for field, expected in (
            ("input_bw", 4.50),
            ("weight_bw", 6.25),
            ("output_bw", 2.50),
        ):
            weighted = sum(
                getattr(p, field) * p.reference_cycles / 144
                for p in results
            )
            self.assertAlmostEqual(weighted, expected)

        for field, expected in (
            ("dram_input_bw", 216),
            ("dram_weight_bw", 180),
            ("dram_output_bw", 120),
        ):
            transfers = sum(
                getattr(p, field) * p.reference_cycles
                for p in results
            )
            self.assertAlmostEqual(transfers, expected)

    def test_group_bandwidths_are_estimates(self):
        results = estimate_gemm_execution_group_resources(
            execution_group_graph=self.execution_graph,
            group_profiles=self.group_profiles,
            group_accesses=self.group_accesses,
            layer_profile=self.layer_profile,
        )
        for profile in results:
            self.assertIn("estimated", profile.estimation_method)


if __name__ == "__main__":
    unittest.main()


class GEMMMultiMappingEstimatorTests(unittest.TestCase):
    CASES = (
        {
            "name": "gemm_16x16x16",
            "dimensions": {"M": 16, "N": 16, "K": 16},
            "macs": 4096,
            "cycles": 256,
            "groups": 32,
            "buffer": (5.0, 2.0, 15.0),
            "dram": (256, 256, 256),
        },
        {
            "name": "gemm_16x8x32",
            "dimensions": {"M": 16, "N": 8, "K": 32},
            "macs": 4096,
            "cycles": 256,
            "groups": 64,
            "buffer": (10.0, 2.0, 3.5),
            "dram": (512, 256, 128),
        },
    )

    def test_all_remaining_mappings_conserve_resources(self):
        for case in self.CASES:
            with self.subTest(mapping=case["name"]):
                name = case["name"]
                dims = case["dimensions"]
                operator_id = "gemm0"

                mapping = parse_mapping_file(
                    Path("timeloop/outputs/gemm") / name
                    / "timeloop-mapper.map.yaml"
                )

                operator_graph = build_operator_dependency_graph(
                    workload_id=name,
                    operators=(
                        OperatorNode(
                            operator_id=operator_id,
                            operator_type="gemm",
                            input_tensors=("Inputs", "Weights"),
                            output_tensors=("C",),
                            attributes={"workload_dimensions": dims},
                        ),
                    ),
                    tensors=(
                        TensorNode(
                            tensor_id="Inputs",
                            shape=(dims["M"], dims["K"]),
                            is_external=True,
                        ),
                        TensorNode(
                            tensor_id="Weights",
                            shape=(dims["K"], dims["N"]),
                            is_external=True,
                        ),
                        TensorNode(
                            tensor_id="C",
                            shape=(dims["M"], dims["N"]),
                            producer_operator_id=operator_id,
                        ),
                    ),
                )

                graph, ldgs = build_gemm_spatial_execution_groups(
                    mapping=mapping,
                    workload_dimensions=dims,
                    workload_id=name,
                    operator_graph=operator_graph,
                    operator_id=operator_id,
                )

                profiles = build_execution_group_profiles(
                    execution_group_graph=graph,
                    operator_graph=operator_graph,
                    ldg_graphs=ldgs,
                )
                accesses = build_execution_group_tensor_accesses(
                    execution_group_graph=graph,
                    operator_graph=operator_graph,
                    ldg_graphs=ldgs,
                )

                input_bw, weight_bw, output_bw = case["buffer"]
                dram_input, dram_weight, dram_output = case["dram"]

                layer_profile = {
                    "reference_compute_work": case["macs"],
                    "reference_execution_cycles": case["cycles"],
                    "buffer_input_shared_bw": input_bw,
                    "buffer_weight_shared_bw": weight_bw,
                    "buffer_output_shared_bw": output_bw,
                    "dram_input_transfers": dram_input,
                    "dram_weight_transfers": dram_weight,
                    "dram_output_transfers": dram_output,
                }

                results = estimate_gemm_execution_group_resources(
                    execution_group_graph=graph,
                    group_profiles=profiles,
                    group_accesses=accesses,
                    layer_profile=layer_profile,
                )

                self.assertEqual(len(results), case["groups"])
                self.assertAlmostEqual(
                    sum(p.estimated_compute_work for p in results),
                    case["macs"],
                )
                self.assertAlmostEqual(
                    sum(p.reference_cycles for p in results),
                    case["cycles"],
                )

                for field, expected in (
                    ("input_bw", input_bw),
                    ("weight_bw", weight_bw),
                    ("output_bw", output_bw),
                ):
                    actual = sum(
                        getattr(p, field) * p.reference_cycles
                        / case["cycles"]
                        for p in results
                    )
                    self.assertAlmostEqual(actual, expected)

                for field, expected in (
                    ("dram_input_bw", dram_input),
                    ("dram_weight_bw", dram_weight),
                    ("dram_output_bw", dram_output),
                ):
                    actual = sum(
                        getattr(p, field) * p.reference_cycles
                        for p in results
                    )
                    self.assertAlmostEqual(actual, expected)

                self.assertTrue(
                    all(p.reference_cycles > 0 for p in results)
                )


if __name__ == "__main__":
    unittest.main()
