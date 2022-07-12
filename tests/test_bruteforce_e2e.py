""" This file implements are very crude, brute-force test function
that makes sure that all storage backends work (don't crash) with
all distance functions. This will write to disk, etc. Run from the
SparseLSH root directory.
"""
import sys
import unittest

from scipy.sparse import csr_matrix

from sparselsh import LSH


try:
    import redis
except ImportError:
    redis = None

try:
    import bsddb
except ImportError:
    bsddb = None

try:
    import leveldb
except ImportError:
    leveldb = None


X = csr_matrix([
    [0, 0, 0, 0, 0, 0,  1],
    [3, 0, 0, 0, 0, 0, -1],
    [0, 1, 0, 0, 0, 0,  1],
    [0, 0, 0, 9, 0, 0,  1],
    [0, 0, 0, 0, 0, 0,  0],
    [4, 0, 0, 0, 0, 0,  0],
    [0, 0, 0, 0, 0, 0,  1],
    [0, 0, 2, 0, 0, 0,  1],
    [0, 0, 0, 0, 0, 0,  0],
    [1, 1, 1, 1, 1, 1,  1]
])

X_sim = csr_matrix([
    [1, 1, 1, 1, 1, 1, 0]
])
Y_sim = [
    10
]

# Simulate extra-data / classes
y = [
    'one',
    'two',
    'three',
    'four',
    'zero',
    'zero',
    'seven',
    'eight',
    'nine',
    'last',
]


def lsh_instance(bits, dimensions, hashtables, storage_type):
    """ Build an lsh instance using specified settings & storage method.

    Params:
        bits: n bitlength of hash key
        dimentions: n features in each input point
        hashtables: number of parallel hashtables to construct
        storage_type: type of storage backend. one of:
            'berkeleydb', 'leveldb', 'redis', 'dict'
    """
    if storage_type == 'berkeleydb':
        if sys.version_info[0] < 3:
            conf = {'berkeleydb':{'filename': './db'}}
        else:
            raise NotImplementedError('BerkeleyDB not supported in Python3')
    elif storage_type == 'leveldb':
        conf = {'leveldb':{'db': 'ldb'}}
    elif storage_type == 'redis':
        conf = {'redis': {'host': '127.0.0.1', 'port': 6379}}
    else:
        conf = {'dict': None}

    lsh = LSH(bits,
              dimensions,
              num_hashtables=hashtables,
              storage_config=conf)
    return lsh


def build_index(lsh, X, y):
    for ix in range(X.shape[0]):
        x = X.getrow(ix)
        c = None
        if y:
            c = y[ix]
        lsh.index(x, extra_data=c)


def query_index(lsh, x):
    similar = lsh.query(x, num_results=1)
    #print "SIMILAR", similar
    return similar


def run_lsh(bits, dimensions, hashtables, storage_type, distance_fn):
    lsh = lsh_instance(bits, dimensions, hashtables, storage_type)
    build_index(lsh, X, y)
    # [(<1x7 sparse matrix of type '<class 'numpy.int64'>'>, 'last', array([1.]))]
    similar_points = query_index(lsh, X_sim.getrow(0))
    print("similar_points", similar_points)
    # for s_point in similar_points:
    #     if not s_point:
    #         continue

    #     # first result from first plane
    #     print("s_point", s_point)
    #     point = s_point[0]
    #     print("point", point)
    #     extra_data = s_point[1]
    #     similarity = None
    #     if len(s_point) == 3:
    #         similarity = s_point[2]
    #     print("extra_data", extra_data)
    #     print("similarity", similarity)

    #     # print "Found similar point", point.todense()
    #     # if extra_data:
    #     #     print "with data", extra_data
    #     # print "similarity", similarity
    #     errmsg = "Similar point returned is invalid (not a sparse csr mx)"
    #     assert(type(point) == csr_matrix), errmsg

    for result_ix, s_point in enumerate(similar_points):
        print("result_ix:", result_ix, "s_point:", s_point)
        point = s_point[0][0]
        print("point:", point.todense())
        extra_data = s_point[0][1] if len(s_point[0]) == 2 else None
        print("extra_data:", extra_data)
        similarity = s_point[1]
        # print "Found similar point", point.todense()
        # if extra_data:
        #     print "with data", extra_data
        # print "similarity", similarity
        print("similarity:", similarity)
        errmsg = "Similar point returned is invalid (not a sparse csr mx)"
        assert(type(point) == csr_matrix), errmsg




class TestCase(unittest.TestCase):
    def test_e2e_run_lsh(self):
        # Run this whole thing this many times to tease our
        # any errors caused by plane randomization
        n_runs = 5
        # Settings
        bits = 4
        dimensions = X.shape[1]
        hashtables = [1, 2, 3]
        storage_types = ["dict"]
        # if redis is not None:
        #     storage_types.append("redis")
        if bsddb is not None:
            storage_types.append("berkeleydb")
        if leveldb is not None:
            storage_types.append("leveldb")

        if sys.version_info[0] == 2:
            storage_types.append("berkeleydb")
        distance_functions = (
            "hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"
        )
        for _ in range(n_runs):
            for h in hashtables:
                for s in storage_types:
                    print("Testing with backend", s)
                    for d in distance_functions:
                        print("Testing distance function", d)
                        run_lsh(bits, dimensions, h, s, d)


if __name__ == "__main__":
    unittest.main()
