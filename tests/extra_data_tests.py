import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sparselsh import LSH


class ExtraDataTestCase(unittest.TestCase):
    """
    Regressions related to extra data on LSH indices.
    """
    def test_accepts_same_dimensional_extra_data(self):
        lsh = LSH(hash_size=256, input_dim=2)

        data = csr_matrix([[0, 0], [100, 100], [200, 200]])
        extra_data = [0.1, 0.2, 0.3]
        lsh.index(data, extra_data=extra_data)
        results = lsh.query(
            csr_matrix([0, 0]), num_results=1)
        (row, extra), dist = results[0]
        self.assertEqual(extra, 0.1)

    def test_rejects_bad_extra_data_dimensions(self):
        lsh = LSH(hash_size=8, input_dim=2)
        data = csr_matrix([[0, 0], [100, 100], [200, 200]])
        extra_data = [0.1, 0.2, 0.3]
        with self.assertRaises(AssertionError):
            lsh.index(data, extra_data=extra_data[:-1])

    def test_int_extra_data_regression(self):
        """
        This tests the case where extra_data is a single
        object, this caused an exception in versions <2.1.1
        """
        lsh = LSH(hash_size=8, input_dim=2)
        data = [[0, 0], [100, 100], [200, 200]]
        for ix, point in enumerate(data):
            x = csr_matrix(point)
            lsh.index(x, extra_data=ix)

    def test_string_extra_data_regression(self):
        lsh = LSH(hash_size=8, input_dim=2)
        data = [[0, 0], [100, 100], [200, 200]]
        for ix, point in enumerate(data):
            x = csr_matrix(point)
            lsh.index(x, extra_data=str(ix))

    def test_numpy_extra_data(self):
        lsh = LSH(hash_size=8, input_dim=2)
        data = [[0, 0], [100, 100], [200, 200]]
        extra_datas = []
        for ix, point in enumerate(data):
            x = csr_matrix(point)
            extra_datas.append(np.array([ix]))
            lsh.index(x, extra_data=extra_datas[-1])
        results = lsh.query(
            csr_matrix(data[0]), num_results=1)
        (row, extra), dist = results[0]
        self.assertEqual(extra, extra_datas[0])


if __name__ == '__main__':
    unittest.main()
