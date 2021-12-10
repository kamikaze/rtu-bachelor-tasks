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
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def translation_mat(dx: float, dy: float):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ])


def scale_mat(sx: float, sy: float):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ])


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

    return dot(i, vec2d[:, None]).transpose()[0] + np.array([0, 0, 1, ])


def vec3d_to_vec2d(vec3d):
    i = np.array((
        (1, 0, 0,),
        (0, 1, 0,),
    ))

    return dot(i, vec3d[:, None])


class Character:
    def __init__(self, pos: Optional[np.array] = None):
        self._geometry: Optional[np.array] = None

        self._angle: float = 0.
        self._speed: float = 0.01
        self._accel_vec: np.array = np.zeros((2,))
        self._pos: np.array = np.zeros((2,)) if pos is None else pos

        self._dir_init: np.array = np.array([0.0, 1.0])
        self._dir: np.array = np.array(self._dir_init)

        self._color: str = 'r'

        self._c: np.ndarray = np.identity(3)
        self._r: np.ndarray = np.identity(3)
        self._s: np.ndarray = np.identity(3)
        self._t: np.ndarray = translation_mat(*self._pos)
        self._update_c()

        self.generate_geometry()

    def _update_c(self):
        self._c = dot(dot(self._s, self._t), self._r)

    def apply_thrust(self, thrust: float):
        angle = np.radians(self._angle + 90)
        thrust_vec = np.array([thrust * np.cos(angle), thrust * np.sin(angle)])
        # TODO: limiting should be done inside circle, not the square. Now max 45deg accel is greater than 1.
        self._accel_vec = np.clip(self._accel_vec + thrust_vec, [-0.1, -0.1], [0.1, 0.1])

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle: float):
        if angle < 0:
            self._angle = angle + 360
        elif angle >= 360:
            self.angle = angle - 360
        else:
            self._angle = angle

        self._r = rotation_mat(angle)
        self._update_c()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed: float):
        self._speed = speed

    @abstractmethod
    def generate_geometry(self):
        pass

    def move(self):
        print(self._pos)
        self._pos += self._accel_vec
        self._t = translation_mat(*self._pos)
        self._update_c()
        # self._pos = self._pos * self._speed * self._angle

    def draw(self):
        if self._geometry is not None:
            x_values = []
            y_values = []

            for vec2d in self._geometry:
                vec3d = vec2d_to_vec3d(vec2d)
                vec3d = np.dot(self._c, vec3d)

                vec2d = vec3d_to_vec2d(vec3d)

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
        self._geometry = np.array((
            (-2, 0,),
            (1, 0,),
            (0, 1,),
            (-2, 0,),
        ))


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
    elif event.key == 'up':
        PLAYER.apply_thrust(0.01)


def main():
    global IS_RUNNING, PLAYER

    matplotlib.use('TkAgg')
    plt.rcParams['figure.figsize'] = (10, 10,)
    plt.ion()

    PLAYER = Player(np.array((0., 0.,)))
    # characters = [PLAYER, Asteroid(np.random.rand(2) * 10), Asteroid(np.random.rand(2) * 10)]
    characters = [PLAYER, ]
    IS_RUNNING = True

    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)

    while IS_RUNNING:
        plt.clf()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        for character in characters:
            character.move()
            character.draw()

            if isinstance(character, Player):
                plt.title(f'Angle: {character.angle}')

        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
