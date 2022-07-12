# thanks to S. Bhargav for submitting this test code
import unittest

from scipy.sparse import csr_matrix

from sparselsh import LSH


class IndexThenQueryTestCase(unittest.TestCase):
    """ Holding place for general regression tests for thing we
    find when indexing and then querying items.
    """
    def test_can_query_using_existing_row(self):
        X = csr_matrix([
            [3, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0,  1],
            [1, 1, 1, 1, 1, 1,  1]
        ])

        # One class number for each input point
        y = ["0", "blah", "kill"]
        # I've changed the last 0 to a 1
        X_sim = csr_matrix([
            [1, 1, 1, 1, 1, 1, 1]
        ])

        lsh = LSH(4,
                  X.shape[1],
                  num_hashtables=1,
                  storage_config={"dict":None})

        for ix in range(X.shape[0]):
            x = X.getrow(ix)
            c = y[ix]
            lsh.index(x, extra_data=c)

        # find the point in X nearest to X_sim
        distance_functions = (
            None,
            "hamming", "euclidean", "true_euclidean",
            "cosine", "l1norm"
        )
        for dist in distance_functions:
            points = lsh.query(X_sim, distance_func=dist, num_results=1)

        self.assertEqual(type(points[0][0][0]), csr_matrix)
        truth = points[0][0][0].todense() == X_sim.todense()
        self.assertTrue(truth.all())


if __name__ == '__main__':
    unittest.main()
