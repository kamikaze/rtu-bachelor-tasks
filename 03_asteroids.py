from abc import abstractmethod
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Character:
    def __init__(self):
        self._geometry: list[float] = []
        self._angle: float = 0.0
        self._speed: float = 0.0
        self._pos: np.array = np.array([0, 0])
        self._dir: np.array = np.array([0, 1])
        self._color: str = 'r'
        self._c: np.ndarray = np.identity(3)
        self._r: np.ndarray = np.identity(3)
        self._t: np.ndarray = np.identity(3)

        self.generate_geometry()

    @property
    def angle(self):
        return

    @angle.setter
    def angle(self, angle: float):
        self._angle = angle

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
        pass


class Player(Character):
    def generate_geometry(self):
        pass


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
