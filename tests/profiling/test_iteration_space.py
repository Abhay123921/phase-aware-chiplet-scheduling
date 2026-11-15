import itertools
import unittest

from simulator.profiling.iteration_space import (
    IterationLevel,
    SymbolicIterationSpace,
    encode_level,
    encode_workload_coordinates,
)


def make_space():
    dimensions = {"M": 2, "N": 3, "K": 4}

    levels = (
        IterationLevel(
            "dram", "DRAM", "temporal",
            {"M": 1, "N": 1, "K": 1}, (), 1,
        ),
        IterationLevel(
            "buffer", "buffer", "temporal",
            {"M": 1, "N": 1, "K": 2}, (("K", 2),), 2,
        ),
        IterationLevel(
            "inter_pe_temporal", "inter_PE_spatial", "temporal",
            {"M": 1, "N": 1, "K": 1}, (), 1,
        ),
        IterationLevel(
            "spatial", "inter_PE_spatial", "spatial",
            {"M": 2, "N": 1, "K": 1}, (("M", 2),), 2,
        ),
        IterationLevel(
            "reg", "reg", "temporal",
            {"M": 1, "N": 3, "K": 2},
            (("N", 3), ("K", 2)), 6,
        ),
    )

    return SymbolicIterationSpace(dimensions, levels)


class IterationSpaceEncoderTests(unittest.TestCase):

    def test_exhaustive_coordinate_round_trip(self):
        space = make_space()

        for m, n, k in itertools.product(range(2), range(3), range(4)):
            coordinates = {"M": m, "N": n, "K": k}
            indices = encode_workload_coordinates(space, coordinates)

            decoded = space.decode(
                dram_index=indices["dram"],
                buffer_index=indices["buffer"],
                inter_pe_temporal_index=indices["inter_pe_temporal"],
                spatial_index=indices["spatial"],
                reg_index=indices["reg"],
            )

            self.assertEqual(decoded.workload_coordinates, coordinates)

    def test_encoder_returns_all_five_hierarchy_levels(self):
        indices = encode_workload_coordinates(
            make_space(), {"M": 1, "N": 2, "K": 3}
        )
        self.assertEqual(
            set(indices),
            {"dram", "buffer", "inter_pe_temporal", "spatial", "reg"},
        )

    def test_rejects_missing_coordinate(self):
        with self.assertRaises(ValueError):
            encode_workload_coordinates(make_space(), {"M": 0, "N": 0})

    def test_rejects_extra_coordinate(self):
        with self.assertRaises(ValueError):
            encode_workload_coordinates(
                make_space(), {"M": 0, "N": 0, "K": 0, "X": 0}
            )

    def test_rejects_out_of_bounds_coordinate(self):
        with self.assertRaises(ValueError):
            encode_workload_coordinates(
                make_space(), {"M": 2, "N": 0, "K": 0}
            )

    def test_rejects_boolean_coordinate(self):
        with self.assertRaises(ValueError):
            encode_workload_coordinates(
                make_space(), {"M": True, "N": 0, "K": 0}
            )

    def test_rejects_non_integer_coordinate(self):
        with self.assertRaises(ValueError):
            encode_workload_coordinates(
                make_space(), {"M": 0, "N": 1.5, "K": 0}
            )

    def test_encode_level_rejects_missing_digit(self):
        level = make_space().get_level("reg")
        with self.assertRaises(ValueError):
            encode_level(level, {"N": 1})

    def test_encode_level_rejects_out_of_range_digit(self):
        level = make_space().get_level("reg")
        with self.assertRaises(ValueError):
            encode_level(level, {"M": 0, "N": 3, "K": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
