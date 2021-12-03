from decimal import Decimal
from functools import reduce
from itertools import repeat
from operator import mul
from typing import Union


def my_pow(number: Union[int, float, Decimal], power: int):
    if power > 0:
        return reduce(mul, repeat(number, power))
    elif power == 0:
        return 1

    return 1 / reduce(mul, repeat(number, abs(power)))


def main():
    print(f'{my_pow(32, 3)=}')
    print(f'{my_pow(32, 0)=}')
    print(f'{my_pow(32, -3)=}')

    print(f'{my_pow(3.2, 2)=}')
    print(f'{my_pow(3.2, 0)=}')
    print(f'{my_pow(3.2, -2)=}')

    print(f'{my_pow(Decimal("3.2"), 2)=}')
    print(f'{my_pow(Decimal("3.2"), 0)=}')
    print(f'{my_pow(Decimal("3.2"), -2)=}')


if __name__ == '__main__':
    main()
