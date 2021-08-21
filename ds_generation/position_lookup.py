import numpy as np


class PositionLookup(list):

    def __init__(self, seq, eps=1e-3):
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
        super().__init__(seq)
        self.eps = eps

    def __contains__(self, key):
        if list.__contains__(self, key):
            return True
        return self.get_closest_atom(*key)

    def get_closest_atom(self, x, y, z):
        for candidate in self:
            dist = np.linalg.norm(np.array([x, y, z]) - candidate)
            if dist <= self.eps:
                return True
        return False
