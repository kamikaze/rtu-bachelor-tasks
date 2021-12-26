import numpy as np


def dot(a, b):
    try:
        a_height, a_width = np.shape(a)
    except ValueError:
        a_height, a_width = np.shape(a)[0], 1

    try:
        b_height, b_width = np.shape(b)
    except ValueError:
        b_height, b_width = np.shape(b)[0], 1

    if a_width != b_height:
        raise ValueError(f'Wrong shape of matrix: {np.shape(a)=} {np.shape(b)=}')

    c = np.array(
        tuple(
            sum(x * y for x, y in zip(a[row_idx], b[:, col_idx]))
            for row_idx in range(a_height)
            for col_idx in range(b_width)
        )
    )

    return c.reshape((a_height, b_width))


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

    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    b = np.array([
        [1, ],
        [2, ],
        [3, ]
    ])
    print(f'{dot(a, b)=} vs {np.dot(a, b)=}')


if __name__ == '__main__':
    main()
