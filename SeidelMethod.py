import numpy as np


def solve(A, b, x0, epsilon, util=False, max_n=100):

    x = x0

    if util:
        print('x_0 =', x0, '\n')

    N = A.shape[0]

    for n in range(max_n):

        x_new = x.copy()

        for i in range(N):
            x_new[i] = b[i]

            if i != 0:
                x_new[i] -= x_new[:i] @ A[i, :i]

            if i != n - 1:
                x_new[i] -= x[i+1:] @ A[i, i+1:]

            x_new[i] /= A[i, i]

        if util:
            print('Iteration', n+1)
            print(
                '\t||x_%d|| = %f' % (n, np.linalg.norm(x)),
                '\t||x_%d|| = %f' % (n + 1, np.linalg.norm(x_new)),
                '\t||x_%d - x_%d|| = %f'  % (n + 1, n, np.linalg.norm(x_new - x)),
                '\n',
                '\tx_%d = %s' % (n + 1, str(x_new))
            )

        if np.linalg.norm(x - x_new) < epsilon:
            break

        x = x_new


A = np.array([
    [10, 2, 4],
    [2, 6, 1],
    [4, 1, 7]
], float)

b = np.array([5/2, 7/8, 27/16])

x0 = np.array([0, 0, 0], float)

solve(A, b, x0, 1e-5, True)
