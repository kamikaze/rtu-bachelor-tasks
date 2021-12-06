import numpy as np


def dot(a, b):
    a_shape = np.shape(a)
    b_shape = np.shape(b)

    if a_shape[0] != b_shape[1]:
        raise ValueError(f'Wrong shape of matrix: {np.shape(a)=} {np.shape(b)=}')

    c = np.array(
        tuple(
            sum(x*y for x, y in zip(a[col_idx], b[:, row_idx]))
            for col_idx in range(b_shape[1])
            for row_idx in range(a_shape[0])
        )
    )

    return c.reshape((a_shape[0], b_shape[1]))


def main():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    b = np.array([
        [7, 8],
        [9, 10],
        [11, 12],
    ])
    print(f'{dot(a, b)=} vs {np.dot(a, b)=}')

    a = np.array([
        [1, 2, 3],
    ])
    b = np.array([
        [7, ],
        [9, ],
        [11, ],
    ])
    print(f'{dot(a, b)=} vs {np.dot(a, b)=}')


if __name__ == '__main__':
    main()
