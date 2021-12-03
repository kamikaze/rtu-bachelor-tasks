from typing import Union


class Animal:
    def __init__(self):
        self._hunger_perc: float = 0.5

    def get_hunger_perc(self) -> float:
        return self._hunger_perc

    def eat(self):
        self._hunger_perc = max(0.0, self._hunger_perc - 0.1)

    def move(self):
        pass

    def sleep(self, hours: Union[int, float]):
        self._hunger_perc = min(1.0, self._hunger_perc + 0.1 * hours)


class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.__items_destroyed: int = 0

    def __repr__(self):
        return f'{__class__.__name__}, hunger: {self._hunger_perc}'

    def move(self):
        self._hunger_perc = max(0.0, self._hunger_perc + 0.01)

    @staticmethod
    def meow():
        print('meow')


class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.__bones_hidden: int = 0

    def __repr__(self):
        return f'{__class__.__name__}, hunger: {self._hunger_perc}'

    def move(self):
        self._hunger_perc = max(0.0, self._hunger_perc + 0.1)

    @staticmethod
    def bark():
        print('bark')


class Robot:
    def __init__(self):
        self._battery_percent: float = 1.0

    def __repr__(self):
        return f'{__class__.__name__}, charge: {self._battery_percent}'

    def move(self):
        self._battery_percent = max(0.0, self._battery_percent - 0.1)

    def charge(self, hours: Union[int, float]):
        self._battery_percent = min(1.0, self._battery_percent + 0.1 * hours)


def main():
    who_is_in_the_room = [Dog(), Dog(), Cat(), Robot()]

    for _ in range(10):
        for entity in who_is_in_the_room:
            print(f'Entity: {entity}')

            entity.move()

            if isinstance(entity, Animal):
                entity.eat()

                if entity.get_hunger_perc() <= 0.0:
                    print('Animal is gonna become fat!')

                if isinstance(entity, Dog):
                    entity.bark()
                elif isinstance(entity, Cat):
                    entity.meow()
            elif isinstance(entity, Robot):
                entity.charge(2)


if __name__ == '__main__':
    main()
