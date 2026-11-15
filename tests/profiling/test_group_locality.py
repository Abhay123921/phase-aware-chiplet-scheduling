import unittest

from simulator.profiling.projection import DataSpaceProjection
from simulator.profiling.convolution_adapter import (
    ConvolutionProjectionAdapter,
)
from simulator.profiling.compact_execution_groups import (
    CompactExecutionGroup,
    CompactExecutionGroupSet,
)
from simulator.profiling.group_locality import (
    analyze_group_locality,
    compare_group_locality,
)


def make_adapter(width=2, wpad=0):
    projections = {
        "Inputs": DataSpaceProjection(
            name="Inputs",
            axes=(
                (("N",),),
                (("C",),),
                (("R",), ("P",)),
                (("S",), ("Q",)),
                (("G",),),
            ),
        ),
        "Weights": DataSpaceProjection(
            name="Weights",
            axes=(
                (("C",),),
                (("M",),),
                (("R",),),
                (("S",),),
                (("G",),),
            ),
        ),
        "Outputs": DataSpaceProjection(
            name="Outputs",
            axes=(
                (("N",),),
                (("M",),),
                (("Q",),),
                (("P",),),
                (("G",),),
            ),
        ),
    }

    instance = {
        "N": 1,
        "C": 2,
        "M": 2,
        "R": 1,
        "S": 1,
        "P": 2,
        "Q": 1,
        "G": 1,
        "W": width,
        "H": 1,
        "Wpad": wpad,
        "Hpad": 0,
    }
    return ConvolutionProjectionAdapter(instance, projections)


def make_group_set():
    axes = ("N", "M", "Q", "P", "G")
    dimensions = (
        ("N", 1), ("M", 2), ("Q", 1), ("P", 2), ("G", 1),
        ("C", 2), ("R", 1), ("S", 1),
    )

    groups = []
    for index, (m, p) in enumerate(((0, 0), (0, 1), (1, 0))):
        groups.append(
            CompactExecutionGroup(
                group_id=f"synthetic:tile:{index}",
                output_origin=(0, m, 0, p, 0),
                output_shape=(1, 1, 1, 1, 1),
                spatial_factors=(1, 1, 1, 1, 1),
                logical_slots=2,
                valid_contributions=2,
                padding_slots=0,
            )
        )

    return CompactExecutionGroupSet(
        layer_id="synthetic",
        output_axes=axes,
        dimensions=dimensions,
        spatial_factors=tuple((axis, 1) for axis in axes),
        groups=tuple(groups),
        total_output_coordinates=3,
        total_logical_slots=6,
        total_valid_contributions=6,
        total_padding_slots=0,
    )


class GroupLocalityTests(unittest.TestCase):

    def setUp(self):
        self.adapter = make_adapter()
        self.group_set = make_group_set()

    def test_input_and_weight_reuse(self):
        features = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[0]
        )

        self.assertEqual(features.input_access_attempts, 2)
        self.assertEqual(features.unique_input_elements, 2)
        self.assertEqual(features.input_reuse_factor, 1.0)

        self.assertEqual(features.weight_access_attempts, 2)
        self.assertEqual(features.unique_weight_elements, 2)
        self.assertEqual(features.weight_reuse_factor, 1.0)

    def test_16_bit_footprint_calculation(self):
        features = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[0],
            element_bytes=2,
        )

        self.assertEqual(features.input_footprint_bytes, 4)
        self.assertEqual(features.weight_footprint_bytes, 4)
        self.assertEqual(features.output_footprint_bytes, 2)
        self.assertEqual(features.total_footprint_bytes, 10)

    def test_same_spatial_position_shares_inputs(self):
        left = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[0]
        )
        right = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[2]
        )

        overlap = compare_group_locality(left, right)

        self.assertEqual(overlap.shared_input_elements, 2)
        self.assertEqual(overlap.input_jaccard, 1.0)
        self.assertEqual(overlap.shared_weight_elements, 0)
        self.assertEqual(overlap.weight_jaccard, 0.0)

    def test_adjacent_positions_share_weights(self):
        left = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[0]
        )
        right = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[1]
        )

        overlap = compare_group_locality(left, right)

        self.assertEqual(overlap.shared_input_elements, 0)
        self.assertEqual(overlap.shared_weight_elements, 2)
        self.assertEqual(overlap.weight_jaccard, 1.0)

    def test_overlap_is_symmetric(self):
        left = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[0]
        )
        right = analyze_group_locality(
            self.adapter, self.group_set, self.group_set.groups[2]
        )

        forward = compare_group_locality(left, right)
        reverse = compare_group_locality(right, left)

        self.assertEqual(
            forward.shared_input_elements,
            reverse.shared_input_elements,
        )
        self.assertAlmostEqual(
            forward.left_input_overlap_fraction,
            reverse.right_input_overlap_fraction,
        )
        self.assertAlmostEqual(
            forward.right_input_overlap_fraction,
            reverse.left_input_overlap_fraction,
        )

    def test_padding_is_excluded_from_input_footprint(self):
        padded_adapter = make_adapter(width=1, wpad=1)

        features = analyze_group_locality(
            padded_adapter, self.group_set, self.group_set.groups[0]
        )

        self.assertEqual(features.input_access_attempts, 0)
        self.assertEqual(features.unique_input_elements, 0)
        self.assertEqual(features.input_footprint_bytes, 0)
        self.assertEqual(features.padding_fraction, 0.0)

    def test_invalid_element_size_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_group_locality(
                self.adapter,
                self.group_set,
                self.group_set.groups[0],
                element_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
