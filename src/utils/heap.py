"""
Heap implementations for k-NN search.
"""

import numpy as np
from typing import Tuple, List


class MaxHeap:
    """
    Max heap for maintaining k nearest neighbors.

    Uses a max heap so we can quickly check if a new point
    is closer than the current k-th nearest.

    Parameters
    ----------
    capacity : int
        Maximum number of elements (k).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap: List[Tuple[float, int]] = []  # (distance, index)

    def push(self, distance: float, index: int) -> bool:
        """
        Try to add a new element.

        Returns True if element was added, False if rejected.
        """
        if len(self.heap) < self.capacity:
            self._heap_push((distance, index))
            return True
        elif distance < self.heap[0][0]:
            self._heap_replace((distance, index))
            return True
        return False

    def peek_max(self) -> Tuple[float, int]:
        """Return the maximum element (k-th nearest)."""
        if not self.heap:
            return (float('inf'), -1)
        return self.heap[0]

    def get_sorted(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return sorted arrays of indices and distances."""
        if not self.heap:
            return np.array([]), np.array([])

        sorted_items = sorted(self.heap)
        distances = np.array([d for d, _ in sorted_items])
        indices = np.array([i for _, i in sorted_items])
        return indices, distances

    def _heap_push(self, item: Tuple[float, int]):
        """Push item onto heap."""
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def _heap_replace(self, item: Tuple[float, int]):
        """Replace root with new item and re-heapify."""
        self.heap[0] = item
        self._sift_down(0)

    def _sift_up(self, pos: int):
        """Move item at pos up to maintain heap property."""
        item = self.heap[pos]
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = self.heap[parent_pos]
            if item[0] > parent[0]:  # Max heap: larger goes up
                self.heap[pos] = parent
                pos = parent_pos
            else:
                break
        self.heap[pos] = item

    def _sift_down(self, pos: int):
        """Move item at pos down to maintain heap property."""
        n = len(self.heap)
        item = self.heap[pos]
        child_pos = 2 * pos + 1

        while child_pos < n:
            right_pos = child_pos + 1
            if right_pos < n and self.heap[right_pos][0] > self.heap[child_pos][0]:
                child_pos = right_pos

            if item[0] < self.heap[child_pos][0]:
                self.heap[pos] = self.heap[child_pos]
                pos = child_pos
                child_pos = 2 * pos + 1
            else:
                break

        self.heap[pos] = item

    def __len__(self) -> int:
        return len(self.heap)


class MinHeap:
    """Min heap implementation (for priority queues)."""

    def __init__(self):
        self.heap: List[Tuple[float, any]] = []

    def push(self, priority: float, item: any):
        """Push item with given priority."""
        self.heap.append((priority, item))
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> Tuple[float, any]:
        """Pop item with minimum priority."""
        if not self.heap:
            raise IndexError("pop from empty heap")

        min_item = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return min_item

    def _sift_up(self, pos: int):
        item = self.heap[pos]
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            if item[0] < self.heap[parent_pos][0]:
                self.heap[pos] = self.heap[parent_pos]
                pos = parent_pos
            else:
                break
        self.heap[pos] = item

    def _sift_down(self, pos: int):
        n = len(self.heap)
        item = self.heap[pos]
        child_pos = 2 * pos + 1

        while child_pos < n:
            right_pos = child_pos + 1
            if right_pos < n and self.heap[right_pos][0] < self.heap[child_pos][0]:
                child_pos = right_pos

            if item[0] > self.heap[child_pos][0]:
                self.heap[pos] = self.heap[child_pos]
                pos = child_pos
                child_pos = 2 * pos + 1
            else:
                break

        self.heap[pos] = item

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)
