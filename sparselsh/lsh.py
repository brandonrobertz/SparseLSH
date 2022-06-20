from __future__ import print_function

from operator import itemgetter
from scipy.sparse import csr_matrix, issparse, vstack
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_distances
import hashlib
import numpy as np
import os

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

        assert isinstance(hash_size, int) and hash_size > 0, "hash_size must be a positive integer"
        assert isinstance(input_dim, int) and input_dim > 0, "input_dim must be a positive integer"
        assert isinstance(num_hashtables, int) and num_hashtables > 0, "num_hashtables must be a positive integer"
        
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
        return csr_matrix(dense_planes)

    def _hash(self, input_points, plane_id):
        """ Generates the binary hashes for `input_points` and returns them.

        :param input_points:
            A scipy sparse matrix that contains some points (numbers).
            The dimension needs to be N x `input_dim`, N>0.
        :param plane_id:
            The plane_id is the ID of one of the random uniform planes with
            a dimension of `hash_size` x `input_dim` that is going to be used
            for obtaining hash keys for the input points.
        """
        try:
            planes = self.uniform_planes[plane_id].T
            projections = input_points.dot(planes)
            signs = (projections > 0)

        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print(("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSH instance""", e))
            raise
        else:
            return np.packbits(signs.toarray(), axis=-1)

    def _as_np_array(self, serial_or_sparse):
        """ Takes either a serialized data structure, a sparse matrix, or tuple
        that has the original input points stored, and returns the original
        input point (a 1 x N sparse matrix).
        """
        # if we get a plain sparse matrix, return it (it's the point itself)
        if issparse(serial_or_sparse):
            return serial_or_sparse

        # here we have a serialized pickle object
        if isinstance(serial_or_sparse, str):
            try:
                deserial = deserialize(serial_or_sparse)
            except TypeError:
                print("The value stored is not deserializable")
                raise
        else:
            # If extra_data exists, `tuple` is the entire
            # (point:sparse, extra_daa). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            deserial = serial_or_sparse

        # if we deserialized it, we might have the sparse now
        if issparse(deserial):
            return deserial

        if isinstance(deserial[0], tuple):
            # extra data was supplied, return point
            return tuple[0]

        elif isinstance(deserial, (tuple, list)):
            try:
                return deserial[0]
            except ValueError as e:
                print(("The input needs to be an array-like object", e))
                raise
        else:
            raise TypeError("the input data is not supported")

    def _bytes_string_to_array(self, hash_key):
        """ Takes a hash key (bytes string) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array(list(hash_key))

    def _get_points_digests(self, points, func='sha1'):
        """ Creates digests / checksums of the input points
        using the provided provided hash algorithm.
        """
        if func == 'md5':
            digests = tuple([hashlib.md5(points[i].toarray()).digest() for i in range(points.shape[0])])
        elif func == 'sha1':
            digests = tuple([hashlib.sha1(points[i].toarray()).digest() for i in range(points.shape[0])])
        elif func == 'sha256':
            digests = tuple([hashlib.sha256(points[i].toarray()).digest() for i in range(points.shape[0])])
        elif func == 'sha512':
            digests = tuple([hashlib.sha512(points[i].toarray()).digest() for i in range(points.shape[0])])
        else:
            digests = tuple()
        return digests

    def index(self, input_points, extra_data=None):
        """ Index input points by adding them to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table.

        :param input_points:
            A sparse CSR matrix. The dimension needs to be N x `input_dim`, N>0.
        :param extra_data:
            (optional) A list of values to associate with the points. Commonly
            this is a target/class-value of some type.
        """

        assert issparse(input_points), "input_points needs to be a sparse matrix"
        assert input_points.shape[1] == self.input_dim, "input_points wrong 2nd dimension"
        assert input_points.shape[0] == 1 or (input_points.shape[0] > 1 and \
               (extra_data is None or (isinstance(extra_data, list) and \
               len(extra_data) == input_points.shape[0]))), \
               "input_points dimension needs to match extra data dimension"

        if input_points.shape[0] == 1:
            for i, table in enumerate(self.hash_tables):
                keys = self._hash(input_points, i)
                # NOTE: there was a bug with 0-equal extra_data
                # we need to allow blank extra_data if it's provided
                # NOTE: value needs to be tuple so it's set-hashable
                value = (input_points[0], extra_data)
                table.append_val(keys[0].tobytes(), value)
        else:
            for i, table in enumerate(self.hash_tables):
                keys = self._hash(input_points, i)
                # NOTE: there was a bug with 0-equal extra_data
                # we need to allow blank extra_data if it's provided
                for j in range(keys.shape[0]):
                    # NOTE: value needs to be tuple so it's set-hashable
                    value = (input_points[j], extra_data[j] if extra_data is not None else None)
                    table.append_val(keys[j].tobytes(), value)


    def query(self, query_points, num_results=None, distance_func=None,
              dist_threshold=None, remove_duplicates=False):
        """ Takes `query_points` which is a sparse CSR matrix of N x `input_dim`,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`. The exact return
        value dimensions depends on if query_points is 1 or more dimensional.

        Example query return with 1-dimensional (single row) query_points:

            ```
            # Dimension 0: Results
            [
                # Dimension 1: Data & distance
                (
                    # Tuple: record, label
                    (<1x7 sparse matrix of type ...>, 'three'),
                    # Distance score
                    1.0
                ),
                (
                    (<1x7 sparse matrix of type ...>, 'three'),
                    1.0
                )
            ]
            ```

        Example query return with 2 query_points:

            ```
            # Dimension 0: Query points
            [
                # Dimension 1: Results
                [
                    # Dimension 2: Data & distance
                    (
                        # Tuple: record, label
                        (<1x7 sparse matrix of type ...>, 'three'),
                        # Distance score
                        1.0
                    ),
                    (
                        (<1x7 sparse matrix of type ...>, 'three'),
                        1.0
                    )
                ]
            ]
            ```

        :param query_points:
            A sparse CSR matrix. The dimension needs to be N x `input_dim`, N>0.
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
            "cosine", "l1norm"). By default "euclidean"
            will used.
        :param dist_threshold:
            (optional) Its type and value depend on the chosen distance function.
            Specifies the distance threshold below which we accept a match. If not
            specified then any distance is accepted.
        """
        assert issparse(query_points), "query_points needs to be sparse"
        assert query_points.shape[0] > 0, "query_points needs to be non-empty"
        assert query_points.shape[1] == self.input_dim, "query_points wrong 2nd dimension"
        assert num_results is None or (isinstance(num_results, int) and num_results > 0), "num_results must be a positive integer"

        if distance_func is None or distance_func == "euclidean":
            d_func = LSH.euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = LSH.euclidean_dist
        elif distance_func == "cosine":
            d_func = LSH.cosine_dist
        elif distance_func == "l1norm":
            d_func = LSH.l1norm_dist
        elif distance_func == "hamming":
            d_func = LSH.hamming_dist
        else:
            raise ValueError(
                "The distance function %s is invalid." % distance_func
            )
        if dist_threshold is not None and \
            (dist_threshold <= 0 or (distance_func == "cosine" and dist_threshold > 1.0)):
            raise ValueError(
                "The distance threshold %s is invalid." % dist_threshold
            )
        if num_results is not None and num_results < 1:
            raise ValueError(
                "The max amount of results %s is invalid." % num_results
            )
        
        # NOTE: Currently this only does exact matching on hash key, the
        # previous version also got the 2 most simlilar hashes and included
        # the contents of those as well. Not sure if this should actually
        # be done since we can change hash size or add more tables to get
        # better accuracy

        # Create a list of lists of candidate neighbors
        candidates = [list() for j in range(query_points.shape[0])]

        for i, table in enumerate(self.hash_tables):
            # get hashes of query points for the specific plane
            keys = self._hash(query_points, i)
            for j in range(keys.shape[0]):
                # TODO: if hamming distance is chosen as the distance_func,
                # go through each hash key in table and check if
                # hamming_distance(table_key, point_key) < dist_threshold

                # Create a sublist of candidate neighbors for each query point
                candidates[j].extend(table.get_list(keys[j].tobytes()))

        # Create a ranked list of lists of candidate neighbors
        ranked_candidates = [list() for j in range(query_points.shape[0])]

        for j in range(query_points.shape[0]):
            # Create a sublist of ranked candidate neighbors for each query point
            if len(candidates[j]) > 0:
                # Transofrm candidate neighbors (without extra_info) into a sparse matrix
                csr = vstack(tuple(zip(*candidates[j]))[0])
                # Calculate distance between the query point and all of its candidate neighbors
                distances = d_func(query_points[j], csr)
                if dist_threshold is not None:
                    # Apply the distance threshold
                    accepted = np.where(distances < dist_threshold)[0]
                else:
                    accepted = np.array([i for i in range(distances.size)])
                # Check if any acceptable candidate neighbors w.r.t. dist_threshold
                if accepted.size > 0:
                    if remove_duplicates:
                        # Get indices of unique acceptable neighbors
                        _, unique_idx = np.unique(self._get_points_digests(csr[accepted]), return_index=True)
                        # Rank unique acceptable neighbors by distance function
                        sorted_idx = np.argsort(distances[accepted[unique_idx]])
                        # Extract unique acceptable neighbors' data
                        idx = accepted[unique_idx[sorted_idx]]
                    else:
                        # Rank acceptable neighbors by distance function
                        sorted_idx = np.argsort(distances[accepted])
                        # Extract unique neighbors' data
                        idx = accepted[sorted_idx]
                    neighbors_sorted = csr[idx]
                    dists_sorted = distances[idx]
                    extra_data_sorted = itemgetter(*list(idx))(list(zip(*candidates[j]))[1])
                    # Add data to list
                    try:
                        ranked_candidates[j] = [tuple((tuple((neighbors_sorted[k], extra_data_sorted[k])), dists_sorted[k])) for k in range(idx.size)]
                    except TypeError:
                        ranked_candidates[j] = [tuple((tuple((neighbors_sorted, extra_data_sorted)), dists_sorted))]

        if query_points.shape[0] == 1:
            if num_results is not None:
                lim = min(len(ranked_candidates[0]), num_results)
                ranked_candidates = ranked_candidates[0][:lim]
            else:
                ranked_candidates = ranked_candidates[0]
        else:
            if num_results is not None:
                for j in range(len(ranked_candidates)):
                    if ranked_candidates[j]:
                        lim = min(len(ranked_candidates[j]), num_results)
                        ranked_candidates[j] = ranked_candidates[j][:lim]

        return ranked_candidates

    ### distance functions
    @staticmethod
    def hamming_dist(x, Y):
        dists = np.zeros(Y.shape[0])
        if issparse(x) and issparse(Y):
            for ix, y in enumerate(Y):
                dists[ix] = (x != y).nnz
        elif type(x) is np.ndarray and type(Y) is np.ndarray:
            for ix, y in enumerate(Y):
                dists[ix] = np.count_nonzero((x != y))
        elif type(x) is type(str) and type(Y) is type(list):
            for ix, y in enumerate(Y):
                dists[ix] = sum(c1 != c2 for c1, c2 in zip(x, y))
        return dists

    @staticmethod
    def euclidean_dist(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        diff = Y - xx
        dists = np.sqrt(diff.dot(diff.T).diagonal()).reshape((1,-1))
        return dists[0]

    @staticmethod
    def euclidean_dist_square(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        diff = Y - xx
        if diff.nnz == 0:
            dists = np.zeros((1, Y.shape[0]))
        else:
            if Y.shape[0] > 1:
                dists = diff.dot(diff.T).diagonal().reshape((1,-1))
            else:
                dists = diff.dot(diff.T).toarray()
        return dists[0]

    @staticmethod
    def l1norm_dist(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        dists = np.abs(Y - xx).sum(axis=1).getA().T
        return dists[0]

    @staticmethod
    def cosine_dist(x, Y):
        dists = cosine_distances(x, Y)
        return dists[0]
