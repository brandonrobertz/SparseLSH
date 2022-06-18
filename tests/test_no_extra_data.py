import unittest
import json

import numpy as np
from sparselsh import LSH
from scipy import sparse
from scipy.sparse import csr_matrix, issparse


class NoExtraDataTestCase(unittest.TestCase):
    """
    Some basic tests without using extra data
    """
    def assertSparseEqual(self, val1, val2, *args):
        # assert issparse(val1) and issparse(val2), \
        #         f"Inputs aren't sparse: val1: {val1} val2: {val2}"
        return self.assertArrayEqual(
            val1.todense(), val2.todense(), *args
        )

    def assertArrayEqual(self, val1, val2, *args):
        eq = val1 == val2
        # import IPython; IPython.embed(); import time; time.sleep(2)
        msg = None
        if len(args):
            msg = args[0]
        self.assertTrue(eq.all, msg)

    def test_can_index_matrix_input(self):
        X = csr_matrix( [
            [ 3, 0, 0, 0, 0, 0, -1],
            [ 0, 1, 0, 0, 0, 0,  1],
            [ 1, 1, 1, 1, 1, 1,  1] ])

        # I've changed the last 1 to a 0
        X_sim = csr_matrix([[1, 1, 1, 1, 1, 1, 0],[1, 1, 1, 1, 1, 1, 0]])

        # use same LSH args for both
        num_hashtables = 2
        lsh_args = (4, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=num_hashtables,
            storage_config={"dict":None}
        )

        lsh = LSH(
            *lsh_args, **lsh_kwargs
        )

        saved_planes = [csr_matrix([[-0.2586132576069767, -1.1467848186203173, 0.2697298030427595, -0.377458679001725, 1.3052402867277504, 0.31483703091996196, 1.513445416298458], [-0.2895660057548345, -0.7986952895045719, 0.7614818981744725, -2.1358449200954333, 0.5090224747396664, -0.06953046543085778, 0.7493919893431094], [0.6861926063623005, 0.3571181383681801, 1.3019747947515221, 1.3894976084286044, 0.02607465489125483, 1.2071457981209999, -0.7283376015801192], [0.1481051577292634, -0.6785562744163371, 0.6941996168786435, -0.9687625845202252, -0.4545466392322145, 1.4472261695278974, 0.6012986927974218]]), csr_matrix([[0.6220576243208975, -0.7547027904168463, -1.4306193528454358, -0.16551886914402317, -0.8677382090635611, 0.6528559042628781, 0.1759168995252135], [-2.8141565064625746, 0.32452454919067236, -0.10345405564210859, 1.0689118637878108, -0.8746891151827245, -0.2372559865548032, -1.1587310202819618], [0.41084261137751393, 0.8600037293781597, 0.6127642916348477, 0.5863335379149291, 2.0896838102256963, -0.8381558581655791, 0.6259764043035636], [1.0313369826195353, 1.1342721063034615, 0.08686365780293809, 0.5946237152553618, 1.0040265143547742, 0.018188123787919046, 0.32914737484925594]])]
        lsh.uniform_planes = np.array(saved_planes)
        # print("JSON Plane", [ plane.todense().tolist() for plane in lsh.uniform_planes ])
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
        print("first_point_result", len(first_point_result), first_point_result)

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


if __name__ == '__main__':
    unittest.main()
