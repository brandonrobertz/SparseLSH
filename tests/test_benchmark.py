import os
import timeit
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sparselsh import LSH
from tests.base_test import LSHTestBase


class BenchmarkingTestCase(unittest.TestCase, LSHTestBase):
    """
    Do some simple benchmarking

    NOTE: This test only runs if the SPARSELSH_BENCH environment
    variable is set. You can run it like so:

        SPARSELSH_BENCH=1 pytest tests/test_benchmark.py
    """
    def rand_sparse(self, m, n, sparsity=0.98):
        """
        Build a deterministc random sparse matrix. It's important
        that this is deterministic for reproducibility.
        """
        print("Building sparse random matrix...")
        _s = timeit.default_timer()
        np.random.seed(7)
        X_rnd = np.random.rand(m, n)
        np.putmask(X_rnd, X_rnd < sparsity, 0)
        X_sparse = csr_matrix(X_rnd)
        print(f"Done in {timeit.default_timer()-_s}s")
        return X_sparse

    @unittest.skipIf(not os.environ.get("SPARSELSH_BENCH"),
                     "Skipping benchmark")
    def test_query_performance(self):
        # small hashsize to create large buckets
        hashsize = 2
        # rows
        m = hashsize * 100_000
        # features
        n = 10
        # random sparse matrix
        X = self.rand_sparse(m, n)
        print("X:", X.__repr__())
        # make one huge sparse query matrix with lots of rows
        X_query = self.rand_sparse(10, n)
        print("X_query:", X_query.__repr__())

        num_hashtables = 1
        lsh_args = (hashsize, X.shape[1])
        lsh_kwargs = dict(
            num_hashtables=num_hashtables,
            storage_config={"dict": None}
        )
        lsh = LSH(
            *lsh_args, **lsh_kwargs
        )

        print("Starting index benchmark...")
        timing = timeit.timeit(lambda: lsh.index(X), number=1)
        print("Index time:", timing)

        print("Starting query benchmark...")
        timings = []
        for i in range(5):
            t = timeit.timeit(lambda: lsh.query(X_query), number=1)
            print(f"{i}: Query timing: {t}")
            timings.append(t)
        print("Query avg time:", np.mean(timings))


if __name__ == '__main__':
    unittest.main()
