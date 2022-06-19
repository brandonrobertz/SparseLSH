import numpy as np
from scipy.sparse import csr_matrix, issparse


class LSHTestBase(object):
    PLANES = [
        csr_matrix([
            [-0.2586132576069767, -1.1467848186203173, 0.2697298030427595,
             -0.377458679001725, 1.3052402867277504, 0.31483703091996196,
             1.513445416298458],
            [-0.2895660057548345, -0.7986952895045719, 0.7614818981744725,
             -2.1358449200954333, 0.5090224747396664, -0.06953046543085778,
             0.7493919893431094],
            [0.6861926063623005, 0.3571181383681801, 1.3019747947515221,
             1.3894976084286044, 0.02607465489125483, 1.2071457981209999,
             -0.7283376015801192],
            [0.1481051577292634, -0.6785562744163371, 0.6941996168786435,
             -0.9687625845202252, -0.4545466392322145, 1.4472261695278974,
             0.6012986927974218]
        ]),
        csr_matrix([
            [0.6220576243208975, -0.7547027904168463, -1.4306193528454358,
             -0.16551886914402317, -0.8677382090635611, 0.6528559042628781,
             0.1759168995252135],
            [-2.8141565064625746, 0.32452454919067236, -0.10345405564210859,
             1.0689118637878108, -0.8746891151827245, -0.2372559865548032,
             -1.1587310202819618],
            [0.41084261137751393, 0.8600037293781597, 0.6127642916348477,
             0.5863335379149291, 2.0896838102256963, -0.8381558581655791,
             0.6259764043035636],
            [1.0313369826195353, 1.1342721063034615, 0.08686365780293809,
             0.5946237152553618, 1.0040265143547742, 0.018188123787919046,
             0.32914737484925594]
        ])
    ]

    def assertSparseEqual(self, val1, val2, *args):
        assert issparse(val1) and issparse(val2), \
                f"Inputs aren't sparse: val1: {val1} val2: {val2}"
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

    def load_planes(self, lsh):
        saved_planes = self.PLANES
        lsh.uniform_planes = np.array(saved_planes)

