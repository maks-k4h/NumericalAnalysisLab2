import numpy as np


def factorize_StDS(A, util=False):
    """

    :param A: symmetric matrix
    :param util: print util info if True
    :return: (S, d)
    """

    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            assert A[i,j] == A.T[i, j]

    S = np.zeros_like(A)
    d = np.zeros(n)

    d[0] = np.sign(A[0, 0])
    S[0, 0] = np.sqrt(np.abs(A[0, 0]))

    for j in range(1, n):
        S[0, j] = (A[0, j]) / (d[0] * S[0, 0])

    if util:
        print('D_1_1 = %d' % d[0])
        for j in range(n):
            print('S_1_%d = %d' % (j + 1, S[0, j]))

    for i in range(1, n):
        d[i] = np.sign( A[i, i] - d[:i] @ np.power(S[:i,i], 2))
        S[i, i] = np.sqrt(np.abs( A[i, i] - d[:i] @ np.power(S[:i,i], 2)))

        for j in range(i + 1, n):
            S[i, j] = (A[i, j] - (d[:i] * S[:i, i]) @ S[:i, j]) / (d[i] * S[i, i])

        if util:
            print('D_%d_%d = %d' % (i+1, i+1, d[i]))
            for j in range(n):
                print('S_%d_%d = %d' % (i+1, j+1, S[i, j]))

    return S, d


def backwards(A, b):
    """
    Solve a system Ax = b
    :param A: upper triangular square matrix
    :param b: vector
    :return: x
    """

    assert A.shape[0] == A.shape[1]

    n = A.shape[0]

    x = np.zeros(n)

    x[-1] = b[-1] / A[-1][-1]

    for i in range(n-2, -1, -1):
        x[i] = (b[i] - A[i][i+1:] @ x[i+1:]) / A[i,i]

    return x


def forward(A, b):
    """
    Solve a system Ax = b
    :param A: lower triangular square matrix
    :param b: vector
    :return: x
    """

    A_rev = A[::-1, ::-1]
    b_rev = b[::-1]

    x_rev = backwards(A_rev, b_rev)

    return x_rev[::-1]


def solve(A, b, util=False):
    if util:
        print('Solving Ax = b for x where\n\n'
              'A:\n', A, '\n\n',
              'b:\n', b, '\n')

    S, d = factorize_StDS(A)
    D = np.diag(d)

    if util:
        print('StDS decomposition\n',
              'S:\n', S, '\n\n',
              'D:\n', D, '\n\n',
              'Check StDS:\n', S.T @ D @ S, '\n')

    y = forward(S.T @ D, b)

    if util:
        print('Solving for StDy = b\n',
              'y:\n', y, '\n')

    x = backwards(S, y)

    if util:
        print('Solving for Sx = y\n',
              'x:\n', x, '\n')

        print('Check Ax:\n', A @ x, '\n')

    return x


def det(A):
    S, d = factorize_StDS(A)
    return np.prod(d) * np.prod(S.diagonal()) ** 2



A = np.array([
    [10, 2, 4],
    [2, 6, 1],
    [4, 1, 7]
], float)

b = np.array([5/2, 7/8, 27/16])

solve(A, b, True)

print('Determinant of A:',det(A))


