import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


SEGMENT_COUNT = 3
TARGET_POINT = np.array((-3.0, 0.0))
ANCHOR_POINT = np.array((0.0, 0.0))
IS_RUNNING = True


def rotation(degrees: float):
    """
    :param degrees: Degrees in radians
    :return:
    """
    c = np.cos(degrees)
    s = np.sin(degrees)

    return np.array((
        (c, -s),
        (s, c),
    ))


def d_rotation(degrees: float):
    """
    :param degrees: Degrees in radians
    :return:
    """
    c = np.cos(degrees)
    s = np.sin(degrees)

    return np.array((
        (-s, -c),
        (c, -s),
    ))


def on_button_press(event):
    global TARGET_POINT

    if all([event.xdata, event.ydata]):
        TARGET_POINT = np.array((event.xdata, event.ydata))


def on_key_press(event):
    global IS_RUNNING

    if event.key == 'escape':
        IS_RUNNING = False


def main():
    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', on_button_press)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    joint_length = 2.0
    loss = 0.0
    step = 0.01

    segment = np.array((0.0, 1.0)) * joint_length
    np_joints = np.array([segment.copy() for _ in range(SEGMENT_COUNT + 1)])
    np_joints[0] = np.array((0.0, 0.0))

    rs = [None] * SEGMENT_COUNT
    thetas = [np.deg2rad(-10.0)] * SEGMENT_COUNT

    while IS_RUNNING:
        plt.clf()
        plt.title(f'loss: {loss:.4f} thetas: {tuple(round(np.rad2deg(theta)) for theta in thetas)}')

        for idx in range(SEGMENT_COUNT):
            theta = thetas[idx]
            r = rs[idx] = rotation(theta)
            dr = d_rotation(theta)
            np_joints[idx+1] = np.dot(r, segment) + np_joints[idx]

            x = dr @ segment

            for i in range(idx, 0, -1):
                x = rs[i-1] @ x

            d_theta = np.sum(x * -2 * (TARGET_POINT - np_joints[-1]))
            thetas[idx] -= d_theta * step

        loss = np.sum((TARGET_POINT - np_joints[-1]) ** 2)

        if len(np_joints):
            plt.plot(np_joints[:, 0], np_joints[:, 1])

        plt.scatter(TARGET_POINT[0], TARGET_POINT[1], s=50, c='r')
        plt.xlim(-5, 5)
        plt.ylim(0, 10)
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
