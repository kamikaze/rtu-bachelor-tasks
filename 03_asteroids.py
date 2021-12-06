from abc import abstractmethod
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rotation_mat(degrees: float):
    """
    Rotating around Z axis
    :param degrees:
    :return:
    """
    c = np.cos(degrees)
    s = np.sin(degrees)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def translation_mat(dx: float, dy: float):
    t = np.identity(3)

    return t


def scale_mat(sx: float, sy: float):
    s = np.identity(3)

    return s


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


def vec2d_to_vec3d(vec2d):
    i = np.array((
        (1, 0,),
        (0, 1,),
        (0, 0,),
    ))

    return np.dot(i, vec2d[:, None]) + np.array([0, 0, 0, ])


def vec3d_to_vec2d(vec3d):
    i = np.array((
        (1, 0, 0,),
        (0, 1, 0,),
    ))

    return dot(i, vec3d)


class Character:
    def __init__(self):
        self._geometry: Optional[np.array] = None

        self._angle: float = np.random.random() * np.pi
        self._speed: float = 0.1
        self._pos: np.array = np.zeros((2,))
        self._dir_init: np.array = np.array([0.0, 1.0])
        self._dir: np.array = np.array(self._dir_init)

        self._color: str = 'r'

        self._c: np.ndarray = np.identity(3)
        self._r: np.ndarray = np.identity(3)
        self._s: np.ndarray = np.identity(3)
        self._t: np.ndarray = np.identity(3)

        self.generate_geometry()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle: float):
        self._angle = angle
        self._r = rotation_mat(angle)

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed: float):
        self._speed = speed

    @abstractmethod
    def generate_geometry(self):
        pass

    def draw(self):
        if self._geometry is not None:
            x_values = []
            y_values = []

            for vec2d in self._geometry:
                vec2d = vec3d_to_vec2d(dot(self._r, vec2d_to_vec3d(vec2d)))
                x_values.append(vec2d[0])
                y_values.append(vec2d[1])

            plt.plot(x_values, y_values, c=self._color)


class Player(Character):
    def generate_geometry(self):
        self._geometry = np.array((
            (-1, 0,),
            (1, 0,),
            (0, 1,),
            (-1, 0,),
        ))


class Asteroid(Character):
    def generate_geometry(self):
        pass


IS_RUNNING: bool = False
PLAYER: Optional[Player] = None


def on_press(event):
    global IS_RUNNING, PLAYER

    if event.key == 'escape':
        IS_RUNNING = False
    elif event.key == 'left':
        PLAYER.angle += 5
    elif event.key == 'right':
        PLAYER.angle -= 5


def main():
    global IS_RUNNING, PLAYER

    matplotlib.use('TkAgg')
    plt.rcParams['figure.figsize'] = (10, 10,)
    plt.ion()

    PLAYER = Player()
    characters = [PLAYER, Asteroid(), Asteroid()]
    IS_RUNNING = True

    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)

    while IS_RUNNING:
        plt.clf()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        for character in characters:
            character.draw()

            if isinstance(character, Player):
                plt.title(f'Angle: {character.angle}')

        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
