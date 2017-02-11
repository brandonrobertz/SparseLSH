#!/usr/bin/env python
from __future__ import print_function
import sparselsh
from scipy.sparse import csr_matrix
import numpy as np
import cPickle as pickle
import argparse
import re
import math
import operator

def parse_args():
    desc = 'Search for optimal hyperparams for a given corpus, saving ' \
        'models as we go.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'corpus', type=str,
        help='Path to file, one line per clusterable entity.')
    parser.add_argument(
        '--save', type=str,
        help='Save model to specified file. By default, index is not saved.')
    parser.add_argument(
        '--hashsize', type=int, default=128,
        help='Size of hash. Smaller sizes create fewer "buckets" and hence ' \
        'more variance between items in the bucket.')
    parser.add_argument(
        '--output', type=str, choices=[
            'clusters','wssse'
        ], default='clusters', help='What to output. Default prints out ' \
        'the cluster names and their items. "wssse" outputs variance inside '\
        'clusters and their items. Useful for finding optimal hashsize.')
    args = parser.parse_args()
    return args

def run(corpus_path, save_index=False, hashsize=128, output='clusters'):
    rawcorpus = None
    with open(corpus_path, 'r') as f:
        rawcorpus = f.readlines()

    corpus = map( lambda rc: re.sub( '[^0-9a-z]', '', rc.lower()), rawcorpus)
    maxlen = max( map( lambda x: len(x), corpus))
    size = len(corpus)

    X = np.zeros((size, maxlen))
    for i in range(size):
        c = corpus[i]
        for j in range(len(c)):
            X[i][j] = ord(c[j])

    Xs = csr_matrix(X)
    lsh = sparselsh.LSH(
        hashsize, Xs.shape[1], num_hashtables=1)
    for i in range(Xs.shape[0]):
        x = Xs.getrow(i)
        c = rawcorpus[i]
        lsh.index( x, extra_data=c)

    if save_index:
        with open(save_index, 'wb') as f:
            pickle.dump( lsh, f)

    if output == 'clusters':
        t = lsh.hash_tables[0]
        for k in t.keys():
            vals = t.get_val(k)
            n_vals = len(vals)
            if n_vals == 1:
                continue
            print('\n', k, n_vals)
            for val in vals:
                print(re.sub('\r|\n', '', val[1]))

    elif output == 'wssse':
        cluster_keys = lsh.hash_tables[0].keys()

        def mean(cluster):
            cluster_matrices = map( lambda c: c[0], cluster)
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


if __name__ == '__main__':
    args = parse_args()
    save = args.save if args.save else False
    hashsize = args.hashsize
    run(args.corpus, save_index=save, hashsize=hashsize, output=args.output)
