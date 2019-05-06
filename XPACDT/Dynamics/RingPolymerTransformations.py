#  **************************************************************************
#
#  XPACDT, eXtended PolyAtomic Chemical Dynamics Toolkit
#  XPACDT is a program that can treat polyatomic chemical dynamics problems
#  by solving Newtons equations of motion directly or in an
#  extended ring-polymer phase space. Non-Born-Oppenheimer effects are
#  included employ different approaches, including fewest switches surface
#  hopping.
#
#  Copyright (C) 2019
#  Ralph Welsch, DESY, <ralph.welsch@desy.de>
#
#  This file is part of XPACDT.
#
#  XPACDT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  **************************************************************************

""" This module defines transformations of ring polymer coordinates or
momenta to and from normal modes.
"""

import itertools
import numpy as np
import scipy.fftpack as fft
import math

# TODO: Check if numpy.fft is better; the reordering is super slow!
# TODO: Alternative: Code up matrix...


def to_RingPolymer_normalModes(X, i=None):
    """
    Transform to ring polymer normal mode representation.

    Parameters
    ----------
    X : array of arrays of float
        'normal' representation or the ring polymer.
    i : integer, optional, default None
        Only do transformation in i-th degree of freedom.

    Returns
    -------
    NM : array of arrays of float
        Normal mode representation of the ring-polymer.
    """

    assert (isinstance(X, np.ndarray)), "X not a numpy array!"
    assert (X.ndim == 2), "X rray not two-dimensional!"
    assert (X.dtype == 'float64'), "X array not real!"

    n_beads = X.shape[1]

    NM = X.copy()
    if n_beads == 1:
        return NM

    if i is not None:
        NM[i] = _1d_to_nm(X[i], n_beads)
    else:
        for k, x in enumerate(X):
            NM[k] = _1d_to_nm(x, n_beads)

    return NM


def from_RingPolymer_normalModes(NM, i=None):
    """
    Transform from ring polymer normal mode representation.

    Parameters
    ----------
    NM : array of arrays of float
        Normal mode representation or the ring polymer.
    i : integer, optional, default None
        Only do transformation in i-th degree of freedom.

    Returns
    -------
    X : array of arrays of float
        'normal' representation of the ring-polymer.
    """

    assert (isinstance(NM, np.ndarray)), "NM not a numpy array!"
    assert (NM.ndim == 2), "NM rray not two-dimensional!"
    assert (NM.dtype == 'float64'), "NM array not real!"

    n_beads = NM.shape[1]

    X = NM.copy()
    if n_beads == 1:
        return X

    if i is not None:
        X[i] = _1d_from_nm(NM[i], n_beads)
    else:
        for k, nm in enumerate(NM):
            X[k] = _1d_from_nm(nm, n_beads)

    return X


def _1d_to_nm(x, n):
    """
    Transform to ring polymer normal mode representation in one dimension.

    Parameters
    ----------
    x : array of float
        'normal' representation or the ring polymer in one dimension.
    n : integer
        The number of beads.

    Returns
    -------
    array of floats
        Normal mode representation of the ring-polymer in one dimension.
    """

    assert (n == 1 or n % 2 == 0), "Number of beads not 1 or even!"

    if n == 1:
        return x

    after_fft = fft.rfft(x) / math.sqrt(n)
    after_fft[1:-1] *= math.sqrt(2.0)

    reorder_index = [0, 1] + list(range(3, n+1, 2)) + list(range(n-2, 1, -2))

    return after_fft[reorder_index]


def _1d_from_nm(nm, n):
    """
    Transform from ring polymer normal mode representation in one dimension.

    Parameters
    ----------
    nm : array of float
        Normal mode representation or the ring polymer in one dimension.
    n : integer
        The number of beads.

    Returns
    -------
    array of floats
        'normal' representation of the ring-polymer in one dimension.
    """

    assert (n == 1 or n % 2 == 0), "Number of beads not 1 or even!"

    if n == 1:
        return nm

    l1 = list(range(n-1, n//2, -1))
    l2 = list(range(2, n//2+1))
    reorder_index = [0, 1] + list(itertools.chain(*zip(l1, l2)))

    fft_input = nm[reorder_index]
    fft_input *= math.sqrt(n)
    fft_input[1:-1] /= math.sqrt(2.0)

    return fft.irfft(fft_input)
