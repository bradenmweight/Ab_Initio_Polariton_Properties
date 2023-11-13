import warnings

from numba import jit

from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import cached_property, partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
    chain,
    compress,
    count,
    cycle,
    dropwhile,
    groupby,
    islice,
    repeat,
    starmap,
    takewhile,
    tee,
    zip_longest,
    product,
)
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le



# Algorithm: https://w.wiki/Qai
#@jit(nopython=True)
def _full(A,size):
    while True:
        # Yield the permutation we have
        yield tuple(A)

        # Find the largest index i such that A[i] < A[i + 1]
        for i in range(size - 2, -1, -1):
            if A[i] < A[i + 1]:
                break
        #  If no such index exists, this permutation is the last one
        else:
            return

        # Find the largest index j greater than j such that A[i] < A[j]
        for j in range(size - 1, i, -1):
            if A[i] < A[j]:
                break

        # Swap the value of A[i] with that of A[j], then reverse the
        # sequence from A[i + 1] to form the new permutation
        A[i], A[j] = A[j], A[i]
        A[i + 1 :] = A[: i - size : -1]  # A[i + 1:][::-1]


# Algorithm: modified from the above
@jit(nopython=True)
def _partial(A, r,size):
    # Split A into the first r items and the last r items
    head, tail = A[:r], A[r:]
    right_head_indexes = range(r - 1, -1, -1)
    left_tail_indexes = range(len(tail))

    while True:
        # Yield the permutation we have
        yield tuple(head)

        # Starting from the right, find the first index of the head with
        # value smaller than the maximum value of the tail - call it i.
        pivot = tail[-1]
        for i in right_head_indexes:
            if head[i] < pivot:
                break
            pivot = head[i]
        else:
            return

        # Starting from the left, find the first value of the tail
        # with a value greater than head[i] and swap.
        for j in left_tail_indexes:
            if tail[j] > head[i]:
                head[i], tail[j] = tail[j], head[i]
                break
        # If we didn't find one, start from the right and find the first
        # index of the head with a value greater than head[i] and swap.
        else:
            for j in right_head_indexes:
                if head[j] > head[i]:
                    head[i], head[j] = head[j], head[i]
                    break

        # Reverse head[i + 1:] and swap it with tail[:r - (i + 1)]
        tail += head[: i - r : -1]  # head[i + 1:][::-1]
        i += 1
        head[i:], tail[:] = tail[: r - i], tail[r - i :]


def distinct_permutations(iterable, r=None):
    """ https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#distinct_permutations """

    items = sorted(iterable)

    size = len(items)
    if r is None:
        r = size

    if 0 < r <= size:
        return _full(items,size) if (r == size) else _partial(items, r,size)

    return iter(() if r else ((),))