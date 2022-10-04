#!/usr/bin/env python
from __future__ import print_function

import argparse
import math
import operator
import re

import numpy as np
from scipy.sparse import csr_matrix

try:
    # Python 2
    import cPickle as pickle
except ImportError:
    # Python 3
    import pickle
try:
    # Python <= 2.7
    reduce
except NameError:
    # Python 3
    from functools import reduce

from sparselsh.lsh import LSH


def parse_args():
    desc = "An example tool that will build clusters using a LSH index from an input file containing one record per line. The output will be either a list of cluster groups or the within set sum of squared errors for the given hash size (enabling search for optimal hash-size when ran multiple times with different --hashsize params)."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "corpus", type=str,
        help="Path to file, one record per line."
    )
    parser.add_argument(
        "--save-model", dest="save", action="store_true",
        help=(
            "Save model to specified file. By default, index is not "
            "saved."
        )
    )
    parser.add_argument(
        "--hashsize", type=int, default=256,
        help=(
            "Size of hash. Smaller sizes create fewer hash buckets and "
            "more dissimilar things will end up in the same bucket."
        )
    )
    parser.add_argument(
        "--num-tables", dest="tables", type=int, default=1,
        help=(
            "Number of hyperplanes to use. This doubles the potential "
            "memory usage but increases accuracy."
        )
    )
    parser.add_argument(
        "--output", type=str, choices=[
            "clusters", "wssse"
        ], default="clusters", help=(
            "What to output. `cluster`, default, prints out the cluster "
            "names and their items. `wssse` outputs variance inside "
            "clusters and their items. Useful for finding optimal "
            "hashsize."
        )
    )
    args = parser.parse_args()
    return args


def run(corpus_path, save_index=False, hashsize=128, output="clusters",
        num_tables=1):
    rawcorpus = None
    with open(corpus_path, "r") as f:
        rawcorpus = f.readlines()

    corpus = [re.sub("[^0-9a-z]", "", rc.lower()) for rc in rawcorpus]
    maxlen = max([len(x) for x in corpus])
    size = len(corpus)

    # break up text lines into fixed-length int matrix
    X = np.zeros((size, maxlen))
    for i in range(size):
        c = corpus[i]
        for j in range(len(c)):
            X[i][j] = ord(c[j])

    Xs = csr_matrix(X)
    lsh = LSH(
        hashsize, Xs.shape[1], num_hashtables=num_tables
    )
    lsh.index(Xs, extra_data=rawcorpus)

    if save_index:
        with open(save_index, "wb") as f:
            pickle.dump(lsh, f)

    if output == "clusters":
        t = lsh.hash_tables[0]
        print("Group #,Data")
        for k_ix, k in enumerate(t.keys()):
            vals = t.get_val(k)
            print()
            for val in vals:
                print(str(k_ix) + "," + re.sub("\r|\n", "", val[1]).strip())

    elif output == "wssse":
        cluster_keys = lsh.hash_tables[0].keys()

        def mean(cluster):
            cluster_matrices = map(lambda c: c[0], cluster)
            n = len(cluster_matrices)
            if n == 1:
                return cluster_matrices[0]
            return reduce(operator.add, cluster_matrices) / n

        def error(point, cluster_mean):
            return math.sqrt(
                sum([x**2 for x in (point - cluster_mean).toarray()[0]]))

        wssse = 0
        for key in cluster_keys:
            cluster = lsh.hash_tables[0].get_val(key)
            cluster_mean = mean(cluster)
            for point in cluster:
                # matrix is 0th item w/ extra_data 1st
                e = error(point[0], cluster_mean)
                wssse += e

        print(hashsize, wssse)


def cli():
    args = parse_args()
    save = args.save if args.save else False
    hashsize = args.hashsize
    num_tables = args.tables if args.tables else 1
    run(args.corpus, save_index=save, hashsize=hashsize, output=args.output,
        num_tables=num_tables)


if __name__ == "__main__":
    cli()
