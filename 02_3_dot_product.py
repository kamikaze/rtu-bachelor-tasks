import numpy as np


def dot(a, b):
    c_height = np.shape(a)[0]
    c_width = np.shape(b)[1]

    if c_height != c_width:
        raise ValueError(f'Wrong shape of matrix: {np.shape(a)=} {np.shape(b)=}')

    c = np.array(
        tuple(
            sum(x*y for x, y in zip(a[col_idx], b[:, row_idx]))
            for col_idx in range(c_width)
            for row_idx in range(c_height)
        )
    )

    return c.reshape((c_height, c_width))


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
