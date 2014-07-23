""" This file implements are very crude, brute-force test function
that makes sure that all storage backends work (don't crash) with
all distance functions. This will write to disk, etc. Run from the
SparseLSH root directory.
"""
from scipy.sparse import csr_matrix
from sparselsh import LSH

X = csr_matrix( [
    [ 0, 0, 0, 0, 0, 0, 1],
    [ 3, 0, 0, 0, 0, 0, -1],
    [ 0, 1, 0, 0, 0, 0, 1],
    [ 0, 0, 0, 9, 0, 0, 1],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 4, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 1],
    [ 0, 0, 2, 0, 0, 0, 1],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 1, 1, 1]
])

X_sim = csr_matrix( [
    [ 1, 1, 1, 1, 1, 1, 0]
])
Y_sim = [
    10
]


# Simulate extra-data / classes
y = [
    0,
    3,
    0,
    0,
    0,
    4,
    0,
    0,
    0,
    10,
]

def lsh_instance( bits, dimensions, hashtables, storage_type):
    """ Build an lsh instance using specified settings & storage method.

    Params:
        bits: n bitlength of hash key
        dimentions: n features in each input point
        hashtables: number of parallel hashtables to construct
        storage_type: type of storage backend. one of:
            'berkeleydb', 'leveldb', 'redis', 'dict'
    """
    if storage_type == 'berkeleydb':
        conf = {'berkeleydb':{'filename': './db'}}
    elif storage_type == 'leveldb':
        conf = {'leveldb':{'db': 'ldb'}}
    elif storage_type == 'redis':
        conf = {'redis': {'host': '127.0.0.1', 'port': 6379}}
    else:
        conf = {'dict': None}

    lsh = LSH( bits,
               dimensions,
               num_hashtables=hashtables,
               storage_config=conf)
    return lsh

def build_index( lsh, X, y):
    for ix in xrange(X.shape[0]):
        x = X.getrow(ix)
        c = None
        if y:
            c = y[ix]
        lsh.index( x, extra_data=c)

def query_index( lsh, x):
    similar = lsh.query(x, num_results=1)
    #print "SIMILAR", similar
    return similar

def run_lsh( bits, dimensions, hashtables, storage_type, distance_fn):
    lsh = lsh_instance( bits, dimensions, hashtables, storage_type)
    build_index( lsh, X, y)
    similar_points = query_index( lsh, X_sim.getrow(0))
    for s_point in similar_points:
        point = s_point[0][0]
        extra_data = s_point[0][1] if s_point[0] == 2 else None
        similarity = s_point[1]
        # print "Found similar point", point.todense()
        # if extra_data:
        #     print "with data", extra_data
        # print "similarity", similarity
        errmsg = "Similar point returned is invalid (not a sparse csr mx)"
        assert( type(point) == csr_matrix), errmsg

if __name__ == "__main__":
    # Settings
    bits = 4
    dimensions = X.shape[1]
    hashtables = 1
    storage_types = ("dict", "redis", "leveldb", "berkeleydb")
    distance_functions = ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm")
    for s in storage_types:
        for d in distance_functions:
            run_lsh( bits, dimensions, hashtables, s, d)
