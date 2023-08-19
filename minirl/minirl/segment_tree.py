import numpy as np


class SegmentTree(object):
    """
              1
           /     \ 
         /         \ 
        2           3
      /    \      /    \ 
     4      5    6      7
    /  \   / \  /  \   / \ 
    8  9  10 11 12 13 14 15
    """

    def __init__(self, size, operation, neutral_element, dtype):
        self.size = size
        self.capacity = 1
        while self.capacity < size:
            self.capacity *= 2
        self.tree = np.full(self.capacity * 2, fill_value=neutral_element, dtype=dtype)
        self.operation = operation
        self.neutral_element = neutral_element

    def __setitem__(self, idx, value):
        # Convert slice to array
        if isinstance(idx, slice):
            idx = np.r_[slice(*idx.indices(self.size))]
        # Get indices of leaf nodes
        leaf_idx = idx + self.capacity
        self.tree[leaf_idx] = value
        if isinstance(idx, int):
            leaf_idx = np.asarray([idx])
        # Update values of parent nodes
        parent_idx = np.unique(leaf_idx // 2)
        while len(parent_idx) > 1 or parent_idx[0] > 0:
            left_child = self.tree[2 * parent_idx]
            right_child = self.tree[2 * parent_idx + 1]
            self.tree[parent_idx] = self.operation(left_child, right_child)
            # Go to one level upper
            parent_idx = np.unique(parent_idx // 2)

    def __getitem__(self, idx):
        # Convert slice to array
        if isinstance(idx, slice):
            idx = np.r_[slice(*idx.indices(self.size))]
        assert np.max(idx) < self.capacity
        assert 0 <= np.min(idx)
        return self.tree[self.capacity + idx]

    def reduction(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
        start (inclusive), end (exclusive)
        """
        if end is None or end > self.size:
            end = self.size
        if end < 0:
            end += self.size
        # Shortcut for full reduction
        if start == 0 and end == self.size:
            return self.tree[1]
        # Get start and end idx in the array
        tree_start, tree_end = start + self.capacity, end + self.capacity
        result = self.neutral_element
        while tree_end - tree_start > 0:
            if (tree_start + 1) % 2 == 0:
                result = self.operation(self.tree[tree_start], result)
            tree_start = (tree_start + 1) // 2
            if tree_end % 2 == 1:
                result = self.operation(self.tree[tree_end - 1], result)
            tree_end //= 2
        return result


class SumSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(
            size=size, operation=np.add, neutral_element=0, dtype=np.float32
        )

    def sum(self, start=0, end=None):
        return super().reduction(start, end)

    def find_prefixsum_idx(self, prefixsum):
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        is_parent = np.ones(len(prefixsum), dtype=bool)

        while np.any(is_parent):  # while not all nodes are leafs
            # Go to left children
            idx[is_parent] = 2 * idx[is_parent]
            # If node value smaller than prefixsum,
            # we update prefixsum and move to its right siblings
            prefixsum_new = np.where(
                self.tree[idx] <= prefixsum, prefixsum - self.tree[idx], prefixsum,
            )
            idx = np.where(
                np.logical_and(self.tree[idx] <= prefixsum, is_parent), idx + 1, idx,
            )
            prefixsum = prefixsum_new
            is_parent = idx < self.capacity
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(
            size=size, operation=np.minimum, neutral_element=np.inf, dtype=np.float32
        )

    def min(self, start=0, end=None):
        return super().reduction(start, end)
