import unittest
from types import SimpleNamespace

from simulator.scheduler.compact_group_scheduler import simulate_compact_groups
from simulator.profiling.compact_execution_groups import (
    CompactExecutionGroup,
    CompactExecutionGroupSet,
)
from simulator.profiling.compact_resource_estimator import (
    CompactGroupResourceProfile,
)

from simulator.scheduler.compact_group_scheduler import (
    _classify_resource_phase,
)


SCALES = {
    "input_bw": 1.0,
    "weight_bw": 1.0,
    "output_bw": 1.0,
    "dram_input_bw": 1.0,
    "dram_weight_bw": 1.0,
    "dram_output_bw": 1.0,
}


def make_profile(local=0.1, dram=0.1):
    return SimpleNamespace(
        input_bw=local,
        weight_bw=local,
        output_bw=local,
        dram_input_bw=dram,
        dram_weight_bw=dram,
        dram_output_bw=dram,
    )


class ResourcePhaseClassificationTests(unittest.TestCase):

    def test_compute_dominant_profile(self):
        phase = _classify_resource_phase(
            make_profile(local=0.1, dram=0.1),
            SCALES,
            active_dram_pressure=0.0,
        )
        self.assertEqual(phase, "compute")

    def test_local_memory_dominant_profile(self):
        phase = _classify_resource_phase(
            make_profile(local=2.0, dram=0.1),
            SCALES,
            active_dram_pressure=0.0,
        )
        self.assertEqual(phase, "local_memory")

    def test_dram_dominant_profile(self):
        phase = _classify_resource_phase(
            make_profile(local=0.1, dram=2.0),
            SCALES,
            active_dram_pressure=0.0,
        )
        self.assertEqual(phase, "dram")

    def test_near_tied_profile_is_mixed(self):
        phase = _classify_resource_phase(
            make_profile(local=1.05, dram=0.1),
            SCALES,
            active_dram_pressure=0.0,
        )
        self.assertEqual(phase, "mixed")




def make_scheduling_profile(local=0.1, dram=0.1):
    return CompactGroupResourceProfile(
        group_id="synthetic_group_0",
        layer_id="synthetic_layer",
        estimated_compute_work=100.0,
        reference_cycles=100.0,
        estimated_input_accesses=1.0,
        estimated_weight_accesses=1.0,
        estimated_output_accesses=1.0,
        input_bw=local,
        weight_bw=local,
        output_bw=local,
        dram_input_bw=dram,
        dram_weight_bw=dram,
        dram_output_bw=dram,
    )


def make_synthetic_workload(profile):
    group = CompactExecutionGroup(
        group_id="synthetic_group_0",
        output_origin=(0, 0, 0, 0, 0),
        output_shape=(1, 1, 1, 1, 1),
        spatial_factors=(1, 1, 1, 1, 1),
        logical_slots=1,
        valid_contributions=1,
        padding_slots=0,
    )
    group_set = CompactExecutionGroupSet(
        layer_id="synthetic_layer",
        output_axes=("N", "M", "Q", "P", "G"),
        dimensions=(1, 1, 1, 1, 1, 1, 1, 1),
        spatial_factors=(1, 1, 1, 1, 1),
        groups=(group,),
        total_output_coordinates=1,
        total_logical_slots=1,
        total_valid_contributions=1,
        total_padding_slots=0,
    )
    return {"synthetic": [(group_set, (profile,))]}


class CausalPhaseAwareSchedulingTests(unittest.TestCase):

    def run_policy(self, local, dram):
        profile = make_scheduling_profile(local, dram)
        result = simulate_compact_groups(
            workload_layers=make_synthetic_workload(profile),
            policy="causal_phase_aware",
            calibration_scales=SCALES,
        )
        self.assertEqual(result.completed_groups, 1)
        self.assertEqual(sum(result.chiplet_group_counts.values()), 1)
        return result

    def test_compute_phase_prefers_compute_rich_chiplet(self):
        result = self.run_policy(local=0.1, dram=0.1)
        counts = result.workload_chiplet_counts["synthetic"]
        self.assertEqual(counts["C1"] + counts["C2"], 1)

    def test_local_memory_phase_prefers_memory_rich_chiplet(self):
        result = self.run_policy(local=2.0, dram=0.1)
        self.assertEqual(
            result.workload_chiplet_counts["synthetic"]["C3"], 1
        )

    def test_dram_phase_does_not_force_memory_chiplet(self):
        result = self.run_policy(local=0.1, dram=2.0)
        self.assertNotEqual(
            result.workload_chiplet_counts["synthetic"]["C3"], 1
        )

    def test_mixed_phase_completes_without_affinity_exclusion(self):
        result = self.run_policy(local=1.05, dram=0.1)
        self.assertGreater(result.makespan, 0)


if __name__ == "__main__":
    unittest.main()
