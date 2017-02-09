# SparseLSH

A locality sensitive hashing library with an emphasis on large, highly-dimensional datasets.

## Features

- Fast and memory-efficient calculations using sparse matrices.
- Built-in support for key-value storage backends: pure-python, Redis (memory-bound), LevelDB, BerkeleyDB
- Multiple hash indexes support (based on Kay Zhu's lshash)
- Built-in support for common distance/objective functions for ranking outputs.

## Details

SparseLSH is based on a fork of Kay Zhu's lshash, and is suited for datasets that won't
fit into main memory or are highly dimensional. Using sparse matrices
allows for speedups of easily over an order of magnitude compared to using dense, list-based
or numpy array-based vector math. Sparse matrices also makes it possible to deal with
these datasets purely in memory using python dicts or through Redis. When this isn't
appropriate, you can use one of the disk-based key-value stores, LevelDB and BerkeleyDB.
Serialization is done using cPickle (for raw C speedups), falling back to python
pickle if it's not available.

BTC Donations: `1NejrUgQDm34CFyMHuaff9PNsd8zhd7SgR`

## Installation

`SparseLSH` depends on the following libraries:
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)

Optionally (for in-memory and disk-based persistence):
- [redis](https://pypi.python.org/pypi/redis/)
- [leveldb](https://code.google.com/p/py-leveldb/)
- [bsddb](https://pypi.python.org/pypi/bsddb3/6.0.1) (built-in on Python 2.7.x)

To install (minimal install):

    python setup.py install

If you would like to use the LevelDB or Redis
storage backends, you can install the dependencies
from the `optional-requirements.txt`:

    pip install -r optional-requirements.txt

## Quickstart

To create 4-bit hashes for input data of 7 dimensions:

    from sparselsh import LSH
    from scipy.sparse import csr_matrix

    X = csr_matrix( [
        [ 3, 0, 0, 0, 0, 0, -1],
        [ 0, 1, 0, 0, 0, 0,  1],
        [ 1, 1, 1, 1, 1, 1,  1] ])

    # One class number for each input point
    y = [ 0, 3, 10]

    X_sim = csr_matrix( [ [ 1, 1, 1, 1, 1, 1, 0]])

    lsh = LSH( 4,
               X.shape[1],
               num_hashtables=1,
               storage_config={"dict":None})

    for ix in xrange(X.shape[0]):
        x = X.getrow(ix)
        c = y[ix]
        lsh.index( x, extra_data=c)

    # find the point in X nearest to X_sim
    points = lsh.query(X_sim, num_results=1)

The query will result in a list of matrix-class tuple & similarity
score tuples. A lower score is better in this case:

    [((<1x7 sparse matrix of type '<type 'numpy.int64'>'
        with 7 stored elements in Compressed Sparse Row format>, 10), 1)]

We can look at the most similar matched item by accessing the sparse array
and invoking it's `todense` function:

    In [11]: print points[0][0][0].todense()
    [[1 1 1 1 1 1 1]]

## Main Interface

Most of the parameters are supplied at class init time:

    LSH( hash_size,
         input_dim,
         num_of_hashtables=1,
         storage_config=None,
         matrices_filename=None,
         overwrite=False)

Parameters:

    hash_size:
        The length of the resulting binary hash. This controls how many "buckets"
        there will be for items to be sorted into.

    input_dim:
        The dimension of the input vector. This needs to be the same as the input
        points.

    num_hashtables = 1:
        (optional) The number of hash tables used. More hashtables increases the
        probability of hash-collisions and the more similar items are likely
        to be found for a query item.

    storage = None:
        (optional) A dict representing the storage backend and configuration
        options. The following storage backends are supported with the following
        configurations:
            In-Memory Python Dictionary:
                {"dict": None} # Takes no options
            Redis:
                {"redis": {"host": "127.0.0.1", "port": 6379, "db": 0}
            LevelDB:
                {"leveldb":{"db": "ldb"}}
                Where "ldb" specifies the directory to store the LevelDB database.
                (In this case it will be `./ldb/`)
            Berkeley DB:
                {"berkeleydb":{"filename": "./db"}}
                Where "filename" is the location of the database file.

    matrices_filename = None:
        (optional) Specify the path to the .npz file random matrices are stored
        or to be stored if the file does not exist yet. If you change the input
        dimensions or the number of hashtables, you'll need to set the following
        option, overwrite, to True, or delete this file.

    overwrite = False:
        (optional) Whether to overwrite the matrices file if it already exists.

### Index (Add points to hash table):

- To index a data point of a given `LSH` instance:

    lsh.index(input_point, extra_data=None)

Parameters:

    input_point:
        The input data point is an array or tuple of numbers of input_dim.

    extra_data = None:
        (optional) Extra data to be added along with the input_point.
        This can be used to hold values like class labels, URIs, titles, etc.

This function returns nothing.

### Query (Search for similar points)

To query a data point against a given `LSH` instance:

    lsh.query(query_point, num_results=None, distance_func="euclidean")

Parameters:

    query_point:
        The query data point is a sparse CSR matrix.

    num_results = None:
        (optional) Integer, specifies the max amount of results to be
        returned. If not specified all candidates will be returned as a
        list in ranked order.
        NOTE: You do not save processing by limiting the results. Currently,
        a similarity ranking and sort is done on all items in the hashtable
        regardless if this parameter.

    distance_func = "euclidean":
        (optional) Distance function to use to rank the candidates. By default
        euclidean distance function will be used.

Returns a list of tuples, each of which has the original input point (which
will be a tuple of csr-matrix, extra_data or just the csr-matrix if no extra
data was supplied) and a similarity score.

