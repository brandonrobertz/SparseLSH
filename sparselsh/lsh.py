from __future__ import print_function

import os
import json
import numpy as np
from scipy import sparse

from .storage import storage, serialize, deserialize

class LSH(object):
    """ LSH implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. This can be found in your sparse
        matrix by checking the .shape attribute of your matrix. I.E.,
            `csr_dataset.shape[1]`
    :param num_hashtables:
        (optional) The number of hash tables used for multiple look-ups.
        Increasing the number of hashtables increases the probability of
        a hash collision of similar documents, but it also increases the
        amount of work needed to add points.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict`, `berkeleydb`, `leveldb` or
        `redis`. `config` is the configuration used by the backend.
        Example configs for each type are as follows:
        `In-Memory Python Dictionary`:
            {"dict": None} # Takes no options
        `Redis`:
            `{"redis": {"host": hostname, "port": port_num}}`
            Where `hostname` is normally `localhost` and `port` is normally 6379.
        `LevelDB`:
            {'leveldb':{'db': 'ldb'}}
            Where 'db' specifies the directory to store the LevelDB database.
        `Berkeley DB`:
            {'berkeleydb':{'filename': './db'}}
            Where 'filename' is the location of the database file.
        NOTE: Both Redis and Dict are in-memory. Keep this in mind when
        selecting a storage backend.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist.
        This needs to be True if the input dimensions or number of hashtables
        change.
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    # TODO: load sparse file
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(list(npzfiles.items()), key=lambda x: x[0])
                    # TODO: to sparse
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """
        dense_planes = np.random.randn(self.hash_size, self.input_dim)
        return sparse.csr_matrix(dense_planes)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A scipy sparse matrix that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """
        try:
            input_point = input_point.transpose()
            projections = planes.dot(input_point)

        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print(("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSH instance""", e))
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, serial_or_sparse):
        """ Takes either a serialized data structure, a sparse matrix, or tuple
        that has the original input points stored, and returns the original
        input point (a 1 x N sparse matrix).
        """
        # if we get a plain sparse matrix, return it (it's the point itself)
        if sparse.issparse(serial_or_sparse):
            return serial_or_sparse

        # here we have a serialized pickle object
        if isinstance(serial_or_sparse, str):
            try:
                deserial = deserialize(serial_or_sparse)
            except TypeError:
                print("The value stored is not deserializable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:sparse, extra_daa). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            deserial = serial_or_sparse

        # if we deserialized it, we might have the sparse now
        if sparse.issparse(deserial):
            return deserial

        if isinstance(deserial[0], tuple):
            # extra data was supplied, return point
            return tuples[0]

        elif isinstance(deserial, (tuple, list)):
            try:
                return deserial[0]
            except ValueError as e:
                print(("The input needs to be an array-like object", e))
                raise
        else:
            raise TypeError("the input data is not supported")

    def index(self, input_point, extra_data=None):
        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table.

        :param input_point:
            A sparse CSR matrix. The dimension needs to be 1 * `input_dim`.
        :param extra_data:
            (optional) A value to associate with the point. Commonly this is
            a target/class-value of some type.
        """

        assert sparse.issparse(input_point), "input_point needs to be sparse"

        # NOTE: there was a bug with 0-equal extra_data
        # we need to allow blank extra_data if it's provided
        if not isinstance(extra_data, type(None)):
            # NOTE: needs to be tuple so it's set-hashable
            value = (input_point, extra_data)
        else:
            value = input_point

        for i, table in enumerate(self.hash_tables):
            table.append_val(
                self._hash(self.uniform_planes[i], input_point),
                value)

    def _string_bits_to_array( self, hash_key):
        """ Take our hash keys (strings of 0 and 1) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array( [ float(i) for i in hash_key])

    def query(self, query_point, num_results=None, distance_func=None):
        """ Takes `query_point` which is a sparse CSR matrix of 1 x `input_dim`,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A sparse CSR matrix. The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
            NOTE: You do not save processing by limiting the results. Currently,
            a similarity ranking and sort is done on all items in the hashtable.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """
        assert sparse.issparse(query_point), "query_point needs to be sparse"

        candidates = []
        if not distance_func:
            distance_func = "euclidean"

            for i, table in enumerate(self.hash_tables):
                # get hash of query point
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                for key in list(table.keys()):
                    # calculate distance from query point hash to all hashes
                    distance = LSH.hamming_dist(
                        self._string_bits_to_array(key),
                        self._string_bits_to_array(binary_hash))
                    # NOTE: we could make this threshold user defined
                    if distance < 2:
                        members = table.get_list(key)
                        candidates.extend(members)

            d_func = LSH.euclidean_dist_square

        else:

            if distance_func == "euclidean":
                d_func = LSH.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LSH.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LSH.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LSH.cosine_dist
            elif distance_func == "l1norm":
                d_func = LSH.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")

            # TODO: pull out into fn w/ optional threshold arg
            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                candidates.extend(table.get_list(binary_hash)[0])

        # # rank candidates by distance function
        ranked_candidates = []
        for ix in candidates:
            point = self._as_np_array(ix)
            dist = d_func(query_point, point)
            ranked_candidates.append( (ix,dist))

        # TODO: stop sorting when we have top num_results, instead of truncating
        # TODO: (do this by replacing set with ordered set)
        # after we've done the entire list
        ranked_candidates.sort(key=lambda x: x[1])

        return ranked_candidates[:num_results] if num_results else ranked_candidates

    ### distance functions

    @staticmethod
    def hamming_dist(sparse1, sparse2):
        return (sparse1 != sparse2).sum()

    @staticmethod
    def euclidean_dist(x, y):
        diff = x - y
        return sparse.csr_matrix.sqrt( diff.dot(diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = x - y
        if diff.nnz == 0:
            return 0.0
        result = diff.dot(diff.transpose())
        return result.data[0]

    @staticmethod
    def euclidean_dist_centred(x, y):
        diff = x.mean() - y.mean()
        return diff.dot( diff)

    @staticmethod
    def l1norm_dist(x, y):
        return abs(x - y).sum()

    @staticmethod
    def cosine_dist(x, y):
        return 1 - x.dot(y) / ((x.dot(x) * y.dot(y)) ** 0.5)
