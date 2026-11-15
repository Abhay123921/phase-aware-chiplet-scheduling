import unittest

from simulator.profiling.gemm_ldg_adapter import (
    HIERARCHY_LEVELS,
    build_gemm_ldg,
    build_gemm_ldg_from_mapping,
)
from simulator.profiling.mapping_parser import (
    MappingDescription,
    parse_nest,
)
from simulator.profiling.iteration_space import SymbolicIterationSpace


def make_hierarchy(K=6):
    return {
        k: {
            "dram": 0,
            "buffer": 0,
            "inter_pe_temporal": k // 3,
            "spatial": k % 3,
            "reg": 0,
        }
        for k in range(K)
    }


def build_graph(m=0, n=0, M=4, N=5, K=6, hierarchy=None, padded_K=None):
    hierarchy_extent = K if padded_K is None else padded_K
    return build_gemm_ldg(
        workload_id="gemm_test",
        operator_id="gemm_0",
        group_id=f"C_{m}_{n}",
        M=M,
        N=N,
        K=K,
        padded_K=padded_K,
        output_coordinates={"M": m, "N": n},
        hierarchy_indices_by_k=(
            make_hierarchy(hierarchy_extent) if hierarchy is None else hierarchy
        ),
    )




def make_test_mapping():
    entries = [
        {"target": "DRAM", "type": "temporal",
         "factors": "M2 N1 K1", "permutation": "KMN"},
        {"target": "buffer", "type": "temporal",
         "factors": "M1 N2 K2", "permutation": "KNM"},
        {"target": "inter_PE_spatial", "type": "temporal",
         "factors": "M1 N1 K2", "permutation": "MKN"},
        {"target": "inter_PE_spatial", "type": "spatial",
         "factors": "M2 N1 K2", "permutation": "KNM"},
        {"target": "reg", "type": "temporal",
         "factors": "M1 N3 K1", "permutation": "NKM"},
    ]
    return MappingDescription(
        nests=[parse_nest(entry) for entry in entries]
    )


class GEMMLDGAdapterTests(unittest.TestCase):

    def test_basic_gemm_graph(self):
        graph = build_graph()
        self.assertTrue(graph.validate())
        self.assertEqual(graph.logical_slot_count, 6)
        self.assertEqual(graph.valid_work_item_count, 6)
        self.assertEqual(graph.invalid_work_item_count, 0)

    def test_exact_contribution_edges(self):
        graph = build_graph()
        edges = [e for e in graph.edges if e.relation == "contributes_to"]
        self.assertEqual(len(edges), 6)
        self.assertEqual(
            {e.source_id for e in edges},
            set(graph.work_items),
        )
        self.assertEqual(
            {e.target_id for e in edges},
            {graph.output.output_id},
        )

    def test_input_coordinates_are_correct(self):
        graph = build_graph(m=2, n=3)

        for k, item in enumerate(graph.work_items.values()):
            self.assertEqual(
                item.metadata["input_a_coordinates"],
                {"M": 2, "K": k},
            )
            self.assertEqual(
                item.metadata["input_b_coordinates"],
                {"K": k, "N": 3},
            )

    def test_multiple_output_coordinates(self):
        first = build_graph(m=0, n=0)
        second = build_graph(m=2, n=3)

        self.assertEqual(first.output.output_coordinates, {"M": 0, "N": 0})
        self.assertEqual(second.output.output_coordinates, {"M": 2, "N": 3})
        self.assertNotEqual(
            set(first.work_items),
            set(second.work_items),
        )

    def test_hierarchy_levels_are_preserved(self):
        graph = build_graph()
        self.assertEqual(
            graph.hierarchy_levels,
            HIERARCHY_LEVELS,
        )
        self.assertEqual(
            graph.summary()["partition_counts"],
            {
                "dram": 1,
                "buffer": 1,
                "inter_pe_temporal": 2,
                "spatial": 6,
                "reg": 6,
            },
        )

    def test_rejects_zero_or_negative_dimensions(self):
        for name in ("M", "N", "K"):
            for value in (0, -1):
                dimensions = {"M": 4, "N": 5, "K": 6}
                dimensions[name] = value
                with self.subTest(dimension=name, value=value):
                    with self.assertRaises(ValueError):
                        build_graph(**dimensions)

    def test_rejects_non_integer_dimension(self):
        valid_hierarchy = make_hierarchy(6)

        for value in (6.5, True, "6"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_gemm_ldg(
                        workload_id="gemm_test",
                        operator_id="gemm_0",
                        group_id="invalid_dimension",
                        M=4,
                        N=5,
                        K=value,
                        output_coordinates={"M": 0, "N": 0},
                        hierarchy_indices_by_k=valid_hierarchy,
                    )

    def test_rejects_out_of_bounds_output_coordinates(self):
        for coordinates in (
            {"M": 4, "N": 0},
            {"M": 0, "N": 5},
            {"M": -1, "N": 0},
            {"M": 0, "N": -1},
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    build_gemm_ldg(
                        workload_id="gemm_test",
                        operator_id="gemm_0",
                        group_id="bad_output",
                        M=4,
                        N=5,
                        K=6,
                        output_coordinates=coordinates,
                        hierarchy_indices_by_k=make_hierarchy(),
                    )

    def test_rejects_missing_or_extra_output_coordinates(self):
        for coordinates in (
            {"M": 0},
            {"N": 0},
            {"M": 0, "N": 0, "K": 0},
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    build_gemm_ldg(
                        workload_id="gemm_test",
                        operator_id="gemm_0",
                        group_id="bad_output",
                        M=4,
                        N=5,
                        K=6,
                        output_coordinates=coordinates,
                        hierarchy_indices_by_k=make_hierarchy(),
                    )

    def test_rejects_missing_reduction_index(self):
        hierarchy = make_hierarchy()
        del hierarchy[5]

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)

    def test_rejects_extra_reduction_index(self):
        hierarchy = make_hierarchy()
        hierarchy[6] = {"dram": 0, "buffer": 2, "reg": 0}

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)

    def test_rejects_missing_hierarchy_level(self):
        hierarchy = make_hierarchy()
        del hierarchy[2]["reg"]

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)

    def test_rejects_extra_hierarchy_level(self):
        hierarchy = make_hierarchy()
        hierarchy[2]["unexpected"] = 0

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)

    def test_rejects_negative_hierarchy_index(self):
        hierarchy = make_hierarchy()
        hierarchy[2]["buffer"] = -1

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)

    def test_rejects_non_integer_hierarchy_index(self):
        hierarchy = make_hierarchy()
        hierarchy[2]["reg"] = 1.5

        with self.assertRaises(ValueError):
            build_graph(hierarchy=hierarchy)



    def test_padded_reduction_retains_invalid_slots(self):
        graph = build_graph(K=6, padded_K=8)
        self.assertTrue(graph.validate())
        self.assertEqual(graph.logical_slot_count, 8)
        self.assertEqual(graph.valid_work_item_count, 6)
        self.assertEqual(graph.invalid_work_item_count, 2)
        self.assertEqual(graph.output.expected_contributors, 6)

        invalid_items = [
            item for item in graph.work_items.values() if not item.is_valid
        ]
        self.assertEqual(len(invalid_items), 2)
        self.assertEqual(
            {item.reduction_coordinates["K"] for item in invalid_items},
            {6, 7},
        )
        for item in invalid_items:
            self.assertEqual(item.invalid_reason, "reduction_padding")
            self.assertTrue(item.metadata["is_padding"])
            self.assertNotIn("input_a_coordinates", item.metadata)
            self.assertNotIn("input_b_coordinates", item.metadata)

    def test_padding_has_no_contribution_edges(self):
        graph = build_graph(K=6, padded_K=8)
        contributions = [
            e for e in graph.edges if e.relation == "contributes_to"
        ]
        invalid_ids = {
            item.work_item_id
            for item in graph.work_items.values()
            if not item.is_valid
        }
        self.assertEqual(len(contributions), 6)
        self.assertTrue(
            all(e.source_id not in invalid_ids for e in contributions)
        )

    def test_omitted_padded_extent_preserves_unpadded_behavior(self):
        graph = build_graph(K=6)
        self.assertEqual(graph.logical_slot_count, 6)
        self.assertEqual(graph.valid_work_item_count, 6)
        self.assertEqual(graph.invalid_work_item_count, 0)

    def test_rejects_padded_extent_smaller_than_reduction_extent(self):
        with self.assertRaises(ValueError):
            build_graph(K=6, padded_K=5, hierarchy=make_hierarchy(6))

    def test_rejects_non_integer_or_boolean_padded_extent(self):
        for value in (6.5, True, "8"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_graph(
                        K=6, padded_K=value, hierarchy=make_hierarchy(6)
                    )

    def test_rejects_missing_hierarchy_entry_for_padded_position(self):
        hierarchy = make_hierarchy(8)
        del hierarchy[7]
        with self.assertRaises(ValueError):
            build_graph(K=6, padded_K=8, hierarchy=hierarchy)

    def test_rejects_extra_hierarchy_entry_beyond_padded_extent(self):
        hierarchy = make_hierarchy(9)
        with self.assertRaises(ValueError):
            build_graph(K=6, padded_K=8, hierarchy=hierarchy)

    def test_validates_hierarchy_indices_in_padded_positions(self):
        hierarchy = make_hierarchy(8)
        hierarchy[7]["reg"] = -1
        with self.assertRaises(ValueError):
            build_graph(K=6, padded_K=8, hierarchy=hierarchy)



class GEMMMappingDerivedTests(unittest.TestCase):

    def build_mapping_graph(self, **overrides):
        arguments = {
            "mapping": make_test_mapping(),
            "workload_dimensions": {"M": 4, "N": 6, "K": 8},
            "workload_id": "gemm_mapping_test",
            "operator_id": "gemm_0",
            "group_id": "C_2_4",
            "output_coordinates": {"M": 2, "N": 4},
            "logical_K": 6,
        }
        arguments.update(overrides)
        return build_gemm_ldg_from_mapping(**arguments)

    def test_mapping_derived_graph_and_padding(self):
        graph = self.build_mapping_graph()

        self.assertTrue(graph.validate())
        self.assertEqual(graph.hierarchy_levels, HIERARCHY_LEVELS)
        self.assertEqual(graph.logical_slot_count, 8)
        self.assertEqual(graph.valid_work_item_count, 6)
        self.assertEqual(graph.invalid_work_item_count, 2)
        self.assertEqual(graph.output.expected_contributors, 6)

    def test_mapping_derived_partition_counts(self):
        graph = self.build_mapping_graph()

        self.assertEqual(
            graph.summary()["partition_counts"],
            {
                "dram": 1,
                "buffer": 2,
                "inter_pe_temporal": 4,
                "spatial": 8,
                "reg": 8,
            },
        )

    def test_mapping_derived_coordinate_round_trips(self):
        mapping = make_test_mapping()
        dimensions = {"M": 4, "N": 6, "K": 8}
        graph = self.build_mapping_graph(logical_K=8)
        space = SymbolicIterationSpace.from_mapping(mapping, dimensions)

        for k in range(8):
            item = graph.work_items[f"gemm_0:m2:n4:k{k}"]
            decoded = space.decode(
                dram_index=item.hierarchy_indices["dram"],
                buffer_index=item.hierarchy_indices["buffer"],
                inter_pe_temporal_index=item.hierarchy_indices[
                    "inter_pe_temporal"
                ],
                spatial_index=item.hierarchy_indices["spatial"],
                reg_index=item.hierarchy_indices["reg"],
            )
            self.assertEqual(
                decoded.workload_coordinates,
                {"M": 2, "N": 4, "K": k},
            )

    def test_mapping_padding_has_no_contribution_edges(self):
        graph = self.build_mapping_graph()
        contributions = [
            edge for edge in graph.edges
            if edge.relation == "contributes_to"
        ]
        invalid_ids = {
            item.work_item_id
            for item in graph.work_items.values()
            if not item.is_valid
        }

        self.assertEqual(len(contributions), 6)
        self.assertTrue(
            all(edge.source_id not in invalid_ids for edge in contributions)
        )

    def test_mapping_constructor_rejects_invalid_logical_extent(self):
        for logical_k in (0, -1, 9, 6.5, True):
            with self.subTest(logical_K=logical_k):
                with self.assertRaises(ValueError):
                    self.build_mapping_graph(logical_K=logical_k)

    def test_mapping_constructor_rejects_invalid_dimensions(self):
        for dimensions in (
            {"M": 4, "N": 6},
            {"M": 4, "N": 6, "K": 8, "G": 1},
            {"M": 4, "N": 6, "K": 0},
            {"M": 4, "N": 6, "K": True},
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    self.build_mapping_graph(
                        workload_dimensions=dimensions
                    )

    def test_mapping_constructor_rejects_invalid_output_coordinates(self):
        for coordinates in (
            {"M": 2},
            {"N": 4},
            {"M": 2, "N": 4, "K": 0},
            {"M": 4, "N": 4},
            {"M": 2, "N": -1},
            {"M": True, "N": 4},
            {"M": 2.5, "N": 4},
            {"M": 2, "N": 4.0},
            None,
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    self.build_mapping_graph(
                        output_coordinates=coordinates
                    )



if __name__ == "__main__":
    unittest.main(verbosity=2)
