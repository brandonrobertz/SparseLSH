import unittest

from scipy.sparse import csr_matrix

from sparselsh import LSH
from tests.base_test import LSHTestBase


class NoExtraDataTestCase(unittest.TestCase, LSHTestBase):
    """
    Some basic tests without using extra data
    """

    def test_can_index_matrix_input(self):
        X = csr_matrix([
            [3, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0,  1],
            [1, 1, 1, 1, 1, 1,  1]
        ])

        # I've changed the last 1 to a 0
        X_sim = csr_matrix([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0]
        ])

        num_hashtables = 2
        lsh_args = (4, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=num_hashtables,
            storage_config={"dict": None}
        )
        lsh = LSH(
            *lsh_args, **lsh_kwargs
        )
        self.load_planes(lsh)
        lsh.index(X)

        # returns list of results per input point
        num_results = 2
        query_results = lsh.query(
            X_sim, num_results=num_results
        )
        print("query_results", query_results)
        self.assertEqual(
            len(query_results), X_sim.shape[0],
            "Query results len should equal number of query points"
        )
        first_point_result = query_results[0]
        print("first_point_result", len(first_point_result),
              first_point_result)

        self.assertTrue(
            len(first_point_result) <= num_results,
            "Query results should have correct number of results"
        )

        print("first_point_result[0]", first_point_result[0])
        self.assertEqual(
            len(first_point_result[0]), 2,
            "Result should have two items: result data, dist"
        )

        s_point, similarity = first_point_result[0]
        self.assertEqual(
            len(s_point), 1,
            "Result tuple length should only have one item"
        )
        point = s_point[0]

        self.assertSparseEqual(point, csr_matrix(X.getrow(2)))
        self.assertTrue(similarity is not None)

    def test_with_distance_threshold(self):
        X = csr_matrix([
            [3, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0,  1],
            [1, 1, 1, 1, 1, 1,  1]
        ])

        # I've changed the last 1 to a 0
        X_sim = csr_matrix([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0]]
        )

        lsh_args = (4, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=2,
            storage_config={"dict": None}
        )
        lsh = LSH(
            *lsh_args, **lsh_kwargs
        )
        self.load_planes(lsh)
        lsh.index(X)

        results_all = lsh.query(X_sim, distance_func="cosine",
                                dist_threshold=1.0)
        print("results_all", results_all)
        self.assertEqual(
            len(results_all[0]), X.shape[0],
            "Incorrect number of results for dist_threshold < 1.0"
        )

        results_some = lsh.query(X_sim, distance_func="cosine",
                                dist_threshold=0.5)
        print("results_some", results_some)
        self.assertEqual(
            len(results_some[0]), 2,
            "Incorrect number of results for dist_threshold < 0.5"
        )

        results_none = lsh.query(X_sim, distance_func="cosine",
                                dist_threshold=0.00001)
        print("results_none", results_none)
        self.assertEqual(
            len(results_none[0]), 0,
            "Incorrect number of results for dist_threshold near zero"
        )


if __name__ == '__main__':
    unittest.main()
