from __future__ import print_function

import os
import numpy as np
from operator import itemgetter
from scipy import sparse
from sklearn.metrics.pairwise import cosine_distances

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

    def _hash(self, planes, input_points):
        """ Generates the binary hashes for `input_points` and returns them.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` x `input_dim`.
        :param input_points:
            A scipy sparse matrix that contains only numbers.
            The dimension needs to be N x `input_dim`, N>0.
        """
        try:
            planes = planes.transpose()
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

        assert sparse.issparse(input_points), "input_points needs to be sparse"
        assert isinstance(extra_data, type(None)) or \
            (not isinstance(extra_data, type(None)) and \
            input_points.shape[0] == len(extra_data)), \
            "input_points dimension needs to match extra data dimension"

        for i, table in enumerate(self.hash_tables):
            keys = self._hash(self.uniform_planes[i], input_points)
            # NOTE: there was a bug with 0-equal extra_data
            # we need to allow blank extra_data if it's provided
            if not isinstance(extra_data, type(None)):
                # NOTE: needs to be tuple so it's set-hashable
                for j in range(keys.shape[0]):
                    if not isinstance(extra_data[j], type(None)):
                        value = (input_points[j], extra_data[j])
                        table.append_val(keys[j].tobytes(), value)
                    else:
                        value = input_points[j]
                        table.append_val(keys[j].tobytes(), value)
            else:
                for j in range(keys.shape[0]):
                    value = input_points[j]
                    table.append_val(keys[j].tobytes(), value)

    def _bytes_string_to_array(self, hash_key):
        """ Takes a hash key (bytes string) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array(list(hash_key))

    def query(self, query_points, num_results=None, distance_func="euclidean", dist_threshold=None):
        """ Takes `query_points` which is a sparse CSR matrix of N x `input_dim`,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

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
        assert sparse.issparse(query_points), "query_points needs to be sparse"

        if distance_func == "euclidean":
            d_func = LSH.euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = LSH.euclidean_dist
        elif distance_func == "cosine":
            d_func = LSH.cosine_dist
        elif distance_func == "l1norm":
            d_func = LSH.l1norm_dist
        elif distance_func == "hamming":
            d_func = LSH.hamming_dist
            raise ValueError(
                "The distance function %s has not been implemented yet." % distance_func
            )
        else:
            raise ValueError(
                "The distance function %s is invalid." % distance_func
            )
        if not isinstance(dist_threshold, type(None)) and \
            (dist_threshold <= 0 or (distance_func == "cosine" and dist_threshold > 1.0)):
            raise ValueError(
                "The distance threshold %s is invalid." % dist_threshold
            )
        if not isinstance(num_results, type(None)) and num_results < 1:
            raise ValueError(
                "The max amount of results %s is invalid." % num_results
            )

        # Create a list of lists of candidate neighbors
        candidates = []
        for i, table in enumerate(self.hash_tables):
            # get hashes of query points for the specific plane
            keys = self._hash(self.uniform_planes[i], query_points)
            for j in range(keys.shape[0]):
                # Create a sublist of candidate neighbors for each query point
                if len(candidates) <= j:
                    candidates.append([])
                new_candidates = table.get_list(keys[j].tobytes())
                if not isinstance(new_candidates, type(None)) and len(new_candidates) > 0:
                    candidates[j].extend(new_candidates)

        # Create a ranked list of lists of candidate neighbors
        ranked_candidates = []

        # If a distance threshold is requested
        if not isinstance(dist_threshold, type(None)):
            for j in range(query_points.shape[0]):
                # Create a sublist of ranked candidate neighbors for each query point
                if len(candidates[j]) > 0:
                    # Remove extra info from candidates to convert them into a matrix
                    cand = list(zip(*candidates[j]))[0]
                    cand_csr = sparse.vstack(cand)
                    distances = d_func(cand_csr, query_points[j])
                    accepted = np.where(distances < dist_threshold)
                    # Check if any suitable candidates
                    if np.any(accepted):
                        accepted = (np.unique(accepted[0]), np.unique(accepted[1]))
                        neighbors = cand_csr[accepted[1],:]
                        dists = distances[accepted[0],accepted[1]]
                        # rank candidates by distance function
                        indices = np.argsort(np.array(dists))
                        neighbors_sorted = neighbors[indices]
                        dists_sorted = dists[indices]
                        if not sparse.issparse(candidates[j][0]):
                            # Sort extra_data by ranked distances
                            extra_data_sorted = itemgetter(*list(indices))(list(zip(*candidates[j]))[1])
                            ranked_candidates.append(tuple((neighbors_sorted, extra_data_sorted, dists_sorted)))
                        else:
                            ranked_candidates.append(tuple((neighbors_sorted, dists_sorted)))
                else:
                    ranked_candidates.append(tuple())
        else:
            for j in range(query_points.shape[0]):
                # Create a sublist of ranked candidate neighbors for each query point
                if len(candidates[j]) > 0:
                    # Remove extra info from candidates to convert them into a matrix
                    cand = list(zip(*candidates[j]))[0]
                    cand_csr = sparse.vstack(cand)
                    distances = d_func(cand_csr, query_points[j])[0]
                    # rank candidates by distance function
                    indices = np.argsort(np.array(distances))
                    neighbors_sorted = cand_csr[indices]
                    dists_sorted = distances[indices]
                    if not sparse.issparse(candidates[j][0]):
                        # Sort extra_data by ranked distances
                        extra_data_sorted = itemgetter(*list(indices))(list(zip(*candidates[j]))[1])
                        ranked_candidates.append(tuple((neighbors_sorted, extra_data_sorted, dists_sorted)))
                    else:
                        ranked_candidates.append(tuple((neighbors_sorted, dists_sorted)))
                else:
                    ranked_candidates.append(tuple())

        if not isinstance(num_results, type(None)):
            for j in range(len(ranked_candidates)):
                if len(ranked_candidates[j]) > 0 and ranked_candidates[j][0].shape[0] > num_results:
                    ranked_candidates[j] = (ranked_candidates[j][0][:num_results], ranked_candidates[j][1][:num_results])

        return ranked_candidates

    ### distance functions
    @staticmethod
    def hamming_dist(x, Y):
        return (Y != x).sum(axis=1)

    @staticmethod
    def euclidean_dist(x, Y):
        diff = Y - x
        return np.sqrt(diff.dot(diff.T))

    @staticmethod
    def euclidean_dist_square(x, Y):
        diff = Y - x
        if diff.nnz == 0:
            return 0.0
        result = diff.dot(diff.T)
        return result.data[0]

    @staticmethod
    def l1norm_dist(x, Y):
        return abs(Y - x).sum(axis=1)

    @staticmethod
    def cosine_dist(x, Y):
        return cosine_distances(Y, x)
