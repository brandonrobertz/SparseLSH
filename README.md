======
LSHash
======

A locality sensitive hashing library with an emphasis on large, highly-dimensional datasets.

Features
==========

- Fast and memory-efficient calculations using sparse matrices.
- Built-in support for key-value storage backends: pure-python, Redis (memory-bound), LevelDB, BerkeleyDB
- Multiple hash indexes support (based on Kay Zhu's lshash)
- Built-in support for common distance/objective functions for ranking outputs.

Details
=======

SparseLSH is based on a fork of Kay Zhu's lshash, and is suited for datasets that won't
fit into main memory or have extremely high amounts of features. Using sparse matrices
allows for speedups of easily over an order of magnitude compared to using dense, list-based
vector math. Sparse matrices also makes it possible to deal with these datasets purely in
memory using python dicts or through Redis. When this isn't appropriate, you can use one
of the disk-based key-value stores, LevelDB and BerkeleyDB. Serialization is done using
cPickle (for raw C speedups), falling back to python pickle if it's not available.

Installation
============
`SparseHash` depends on the following libraries:

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)

Optionally (for in-memory and disk-based persistence):

- [redis](https://pypi.python.org/pypi/redis/)
- [leveldb](https://code.google.com/p/py-leveldb/)
- [bsddb](https://pypi.python.org/pypi/bsddb3/6.0.1] (built-in on Python 2.7.x)

- bitarray (if hamming distance is used as distance function)

To install:

.. code-block:: bash

    $ pip install lshash

Quickstart
==========
To create 6-bit hashes for input data of 8 dimensions:

.. code-block:: python

    >>> from lshash import LSHash

    >>> lsh = LSHash(6, 8)
    >>> lsh.index([1,2,3,4,5,6,7,8])
    >>> lsh.index([2,3,4,5,6,7,8,9])
    >>> lsh.index([10,12,99,1,5,31,2,3])
    >>> lsh.query([1,2,3,4,5,6,7,7])
    [((1, 2, 3, 4, 5, 6, 7, 8), 1.0),
     ((2, 3, 4, 5, 6, 7, 8, 9), 11)]


Main Interface
==============

- To initialize a ``LSHash`` instance:

.. code-block:: python

    LSHash(hash_size, input_dim, num_of_hashtables=1, storage_config=None, matrices_filename=None, overwrite=False)

parameters:

``hash_size``:
    The length of the resulting binary hash.
``input_dim``:
    The dimension of the input vector.
``num_hashtables = 1``:
    (optional) The number of hash tables used for multiple lookups.
``storage = None``:
    (optional) Specify the name of the storage to be used for the index
    storage. Options include "redis".
``matrices_filename = None``:
    (optional) Specify the path to the .npz file random matrices are stored
    or to be stored if the file does not exist yet
``overwrite = False``:
    (optional) Whether to overwrite the matrices file if it already exist

- To index a data point of a given ``LSHash`` instance, e.g., ``lsh``:

.. code-block:: python

    lsh.index(input_point, extra_data=None):

parameters:

``input_point``:
    The input data point is an array or tuple of numbers of input_dim.
``extra_data = None``:
    (optional) Extra data to be added along with the input_point.

- To query a data point against a given ``LSHash`` instance, e.g., ``lsh``:

.. code-block:: python

    lsh.query(query_point, num_results=None, distance_func="euclidean"):

parameters:

``query_point``:
    The query data point is an array or tuple of numbers of input_dim.
``num_results = None``:
    (optional) The number of query results to return in ranked order. By
    default all results will be returned.
``distance_func = "euclidean"``:
    (optional) Distance function to use to rank the candidates. By default
    euclidean distance function will be used.
