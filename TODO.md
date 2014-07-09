Sparse Functions
================

The following might have problems with sparsity:

    n = np.load
    n.items()

    np.savez_compressed


    np.random.randn

Need to convert to sparse:

    np.array   -> sparse.csr_matrix

    np.dot     -> sparse.csr_matrix.dot
    SPARSE_MATRIX1.dot( SPARSE2)

    [1 if i > 0 else 0 for i in arr ] ->
        ((sparse_matrix <= 1) * 1)

    np.asarray

Storage functions used:

    append_val
    get_list
    keys
