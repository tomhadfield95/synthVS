import numpy as np


class PositionLookup(list):

    def __init__(self, seq=None, eps=1e-3):
        """Helper class for providing a set of coordinates with soft lookup.

        Keys should be space-separated strings of coordinates ('x y z'). The
        precision with which values are retrieved is specified by <eps>
        in the
        constructor. The L2 norm is used to measure distance between an
        unrecognised query and all of the keys in the dictionary. Any
        query more
        than <eps> from all keys will be considered outside of the set.

        Arguments:
            seq: iterable of initial coordinate ndarrays the list
            eps: precision of the lookup
        """
        seq = seq if seq is not None else []
        super().__init__(seq)
        self.eps = eps

    def append(self, __object) -> None:
        list.append(self, list(__object))

    def __contains__(self, key):
        if super().__contains__(list(key)):
            return True
        return self.get_closest_atom(*key)

    def get_closest_atom(self, x, y, z):
        for candidate in self:
            dist = np.linalg.norm(np.array([x, y, z]) - np.array(candidate))
            if dist <= self.eps:
                return True
        return False

    def index(self, __value, __start: int = ..., __stop: int = ...) -> int:
        if not isinstance(__start, int):
            __start = 0
        if not isinstance(__stop, int):
            __stop = self.__len__()
        min_dist = np.inf
        closest_idx = -1
        for i in range(__start, __stop):
            dist = np.linalg.norm(
                np.array(__value) - np.array(self.__getitem__(i)))
            if dist < min_dist:
                closest_idx = i
                min_dist = dist
        if min_dist < self.eps:
            return closest_idx
        raise ValueError('{0} is not within {1} A of any coodinates in the '
                         'PositionLookup object'.format(__value, self.eps))
