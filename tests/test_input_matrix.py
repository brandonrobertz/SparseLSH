import unittest

from scipy.sparse import csr_matrix

from sparselsh import LSH
from tests.base_test import LSHTestBase


class MatrixIndexTestCase(unittest.TestCase, LSHTestBase):
    """
    Ensure we can pass rows and matrix to the index function
    and have it work the same. This tests both row format
    and matrix format, ensures the projected hash tables are
    the same and also the query results are the same.
    """

    def test_can_index_matrix_input(self):
        X = csr_matrix([
            [3, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0,  1],
            [1, 1, 1, 1, 1, 1,  1]
        ])

        # One class number for each input point
        y = ["one", "two", "three"]

        # two comparisons
        X_sim = csr_matrix([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ])

        # use same LSH args for both
        num_hashtables = 2
        lsh_args = (4, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=num_hashtables,
            storage_config={"dict": None}
        )

        # create LSH instances
        lsh_row = LSH(
            *lsh_args, **lsh_kwargs
        )
        lsh_matrix = LSH(
            *lsh_args, **lsh_kwargs
        )
        # sync uniform planes so hashes are the same
        uniform_planes = lsh_row._generate_uniform_planes()
        lsh_row.uniform_planes = uniform_planes
        lsh_matrix.uniform_planes = uniform_planes

        # add each item row-wise (old version)
        for ix in range(X.shape[0]):
            x = X.getrow(ix)
            c = y[ix]
            lsh_row.index(x, extra_data=c)

        # then index again as a matrix
        lsh_matrix.index(X, extra_data=y)

        # make sure the hash tables are the same
        for table_index, hash_table1 in enumerate(lsh_row.hash_tables):
            hash_table2 = lsh_matrix.hash_tables[table_index]
            keys1 = hash_table1.keys()
            keys2 = hash_table2.keys()
            self.assertEqual(keys1, keys2)
            for key in keys1:
                values1 = hash_table1.get_val(key)
                values2 = hash_table2.get_val(key)
                self.assertEqual(
                    len(values1), len(values2),
                    f"Table keys '{key}' have different numbers of values"
                )
                for result_ix, (val1, extra1) in enumerate(values1):
                    val2, extra2 = values2[result_ix]
                    eq = val1.todense() == val2.todense()
                    self.assertTrue(eq.all, "Resulting points mismatch!")
                    self.assertEqual(extra1, extra2, "Extra data mismatch!")

        # returns list of results per input point
        query_results = lsh_row.query(
            X_sim, num_results=2
        )
        print("* query_results", query_results)
        self.assertLessEqual(
            X_sim.shape[0], len(query_results),
            "Query length should match input points"
        )
        first_query_result = query_results[0]
        print("first_query_result", first_query_result)
        self.assertEqual(
            len(first_query_result), 2,
            "Results data should have a max of two results"
        )
        # first_query_result is tuple of N results, N extra data and, if
        # there are multiple results, a lists of distances for each.
        print("first_query_result[0]", first_query_result[0])
        (point1, extra1), dist1 = first_query_result[0]

        query_results = lsh_matrix.query(
            X_sim, num_results=2
        )
        first_query_result = query_results[0]
        (point2, extra2), dist2 = first_query_result[0]
        print("point2", point2, "extra2", extra2)

        print("points1 & 2", point1, point2)
        self.assertSparseEqual(point1, point2)
        self.assertEqual(extra1, extra2)
        self.assertEqual(dist1, dist2)

    def test_with_inp_matrix_with_dist_threshold(self):
        X = csr_matrix([
            [3, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0,  1],
            [1, 1, 1, 1, 1, 1,  1]
        ])

        # One class number for each input point
        y = ["one", "two", "three"]

        # I've changed the last 1 to a 0
        X_sim = csr_matrix([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0]
        ])

        lsh_args = (4, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=2,
            storage_config={"dict": None}
        )
        lsh = LSH(
            *lsh_args, **lsh_kwargs
        )
        self.load_planes(lsh)
        lsh.index(X, extra_data=y)

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
