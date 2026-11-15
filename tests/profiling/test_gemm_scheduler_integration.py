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
from simulator.profiling.gemm_scheduler_adapter import (
    adapt_gemm_execution_groups_for_scheduler,
)
from simulator.scheduler.compact_group_scheduler import simulate_compact_groups


class GEMMReactiveSchedulerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload_id = "gemm_12x10x18"
        dimensions = {"M": 12, "N": 10, "K": 18}
        operator_id = "gemm0"
        mapping = parse_mapping_file(
            Path("timeloop/outputs/gemm/gemm_12x10x18")
            / "timeloop-mapper.map.yaml"
        )
        operator_graph = build_operator_dependency_graph(
            workload_id=cls.workload_id,
            operators=(
                OperatorNode(
                    operator_id=operator_id,
                    operator_type="gemm",
                    input_tensors=("Inputs", "Weights"),
                    output_tensors=("C",),
                    attributes={"workload_dimensions": dimensions},
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
                    producer_operator_id=operator_id,
                ),
            ),
        )
        graph, ldgs = build_gemm_spatial_execution_groups(
            mapping=mapping,
            workload_dimensions=dimensions,
            workload_id=cls.workload_id,
            operator_graph=operator_graph,
            operator_id=operator_id,
        )
        tile_profiles = build_execution_group_profiles(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=ldgs,
        )
        accesses = build_execution_group_tensor_accesses(
            execution_group_graph=graph,
            operator_graph=operator_graph,
            ldg_graphs=ldgs,
        )
        layer_profile = {
            "reference_compute_work": 2160,
            "reference_execution_cycles": 144,
            "buffer_input_shared_bw": 4.50,
            "buffer_weight_shared_bw": 6.25,
            "buffer_output_shared_bw": 2.50,
            "dram_input_transfers": 216,
            "dram_weight_transfers": 180,
            "dram_output_transfers": 120,
        }
        gemm_profiles = estimate_gemm_execution_group_resources(
            execution_group_graph=graph,
            group_profiles=tile_profiles,
            group_accesses=accesses,
            layer_profile=layer_profile,
        )
        cls.group_set, cls.profiles = adapt_gemm_execution_groups_for_scheduler(
            execution_group_graph=graph,
            gemm_profiles=gemm_profiles,
        )

    def test_adapted_gemm_runs_in_existing_reactive_scheduler(self):
        scales = {
            "input_bw": 1.0,
            "weight_bw": 1.0,
            "output_bw": 1.0,
            "dram_input_bw": 1.0,
            "dram_weight_bw": 1.0,
            "dram_output_bw": 1.0,
        }
        result = simulate_compact_groups(
            workload_layers={
                self.workload_id: [(self.group_set, self.profiles)]
            },
            policy="reactive",
            calibration_scales=scales,
        )
        self.assertEqual(result.completed_groups, 8)
        self.assertGreater(result.makespan, 0)
        self.assertEqual(sum(result.chiplet_group_counts.values()), 8)
        self.assertEqual(
            sum(result.workload_chiplet_counts[self.workload_id].values()),
            8,
        )
        self.assertEqual(result.policy, "reactive")


if __name__ == "__main__":
    unittest.main()


class MixedCNNGEMMConcurrencyTests(GEMMReactiveSchedulerIntegrationTests):
    def test_cnn_and_gemm_run_together_in_reactive_scheduler(self):
        from types import SimpleNamespace

        cnn_group = SimpleNamespace(group_id='cnn_synthetic_group_0')
        cnn_group_set = SimpleNamespace(
            layer_id='cnn_synthetic_layer',
            groups=(cnn_group,),
        )
        cnn_profile = SimpleNamespace(
            group_id='cnn_synthetic_group_0',
            reference_cycles=80.0,
            input_bw=0.1,
            weight_bw=0.1,
            output_bw=0.1,
            dram_input_bw=0.1,
            dram_weight_bw=0.1,
            dram_output_bw=0.1,
        )

        workload = {
            'cnn_synthetic': [(cnn_group_set, (cnn_profile,))],
            self.workload_id: [(self.group_set, self.profiles)],
        }
        scales = {
            'input_bw': 1.0,
            'weight_bw': 1.0,
            'output_bw': 1.0,
            'dram_input_bw': 1.0,
            'dram_weight_bw': 1.0,
            'dram_output_bw': 1.0,
        }

        result = simulate_compact_groups(
            workload_layers=workload,
            policy='reactive',
            calibration_scales=scales,
        )

        self.assertEqual(result.completed_groups, 9)
        self.assertGreater(result.makespan, 0)
        self.assertEqual(
            sum(result.workload_chiplet_counts['cnn_synthetic'].values()),
            1,
        )
        self.assertEqual(
            sum(result.workload_chiplet_counts[self.workload_id].values()),
            8,
        )
        self.assertEqual(sum(result.chiplet_group_counts.values()), 9)
