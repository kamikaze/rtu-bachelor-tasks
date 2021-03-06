import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.ion()


SEGMENT_COUNT = 3
JOINT_LENGTH = 2.0
LEARNING_RATE = 0.01
TARGET_POINT = np.array((-3.0, 0.0))
ANCHOR_POINT = np.array((0.0, 0.0))
IS_RUNNING = True


def rotation(theta: float):
    """
    :param theta: in radians
    :return:
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array((
        (c, -s),
        (s, c),
    ))


def d_rotation(theta: float):
    """
    :param theta: in radians
    :return:
    """
    c = np.cos(theta)
    s = np.sin(theta)

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

    # define a single segment vector as a template
    segment = np.array((0.0, 1.0)) * JOINT_LENGTH
    # generate N segments by duplicating previously defined segment template
    np_joints = np.array([segment.copy() for _ in range(SEGMENT_COUNT + 1)])
    # setting root position of an initial (first) segment to 0, 0.
    np_joints[0] = np.array((0.0, 0.0))

    thetas = [np.deg2rad(-10.0) for _ in range(SEGMENT_COUNT)]

    while IS_RUNNING:
        plt.clf()
        prev_r = None

        for segment_idx in range(SEGMENT_COUNT):
            # getting rotation value
            theta = thetas[segment_idx]
            # getting rotation matrix
            r = rotation(theta)
            dr_theta_1 = d_rotation(theta)
            # calculating current segment vector by adding rotated segment template to the tip of the previous segment
            np_joints[segment_idx+1] = np.dot(r, segment) + np_joints[segment_idx]

            # STILL BLACK MAGIC FOR ME
            x = dr_theta_1 @ segment

            if segment_idx:
                x = prev_r @ x

            # is this somehow related to derivative of the loss function?
            d_theta_1 = np.sum(x * -2 * (TARGET_POINT - np_joints[-1]))
            # END OF BLACK MAGIC

            # updating and storing new rotation value for the current segment
            thetas[segment_idx] -= d_theta_1 * LEARNING_RATE

            prev_r = r

        loss = np.sum((TARGET_POINT - np_joints[-1]) ** 2)
        plt.title(f'loss: {loss:.4f} thetas: {tuple(round(np.rad2deg(theta)) for theta in thetas)}')

        if len(np_joints):
            plt.plot(np_joints[:, 0], np_joints[:, 1])

        plt.scatter(TARGET_POINT[0], TARGET_POINT[1], s=50, c='r')
        plt.xlim(-5, 5)
        plt.ylim(0, 10)
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
