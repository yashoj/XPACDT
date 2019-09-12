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


class RingPolymerTransformations(object):
    """
    Implementation of transformations from ring polymer coordinates and momenta
    to normal modes, and vice-versa. This can be done using transformation
    matrix given in JCP 133, 124104 (2010) or using fast fourier transform
    (FFT). For smaller number of beads (less than 1000), using matrix
    transformation is recommended whereas for larger bead number, using FFT is
    preferable due to computationally efficiency point of view.

    Parameters
    ----------
    nbeads : list of int
        The number of beads for each degree of freedom. Here all elements
        should be the same.
    transform_type : string
        Type of ring polymer normal mode transformation to be used; this can
        be 'matrix' or 'fft'. Default: 'matrix'


    Attributes:
    -----------
    n_beads
    transform_type
    C_matrix (optional depending upon if transform_type=='matrix')

    """

    def __init__(self, n_beads, transform_type='matrix'):

        self.n_beads = n_beads
        self.transform_type = transform_type

        # Make dictionary of C_matrix for all different nbeads!!!
        if (self.transform_type == 'matrix'):
            self.C_matrix = self.get_normal_mode_matrix()
        else:
            self.C_matrix = None
        return

    @property
    def n_beads(self):
        """int : Number of beads, assuming all dof have the same."""
        return self.__n_beads

    @n_beads.setter
    def n_beads(self, n):
        assert (np.all([(i == n[0]) for i in n])), \
               ("Number of beads not same for all degree of freedom")
        # Only take the first dof beads assuming all of them are the same
        self.__n_beads = n[0]

    @property
    def transform_type(self):
        """string : Type of ring polymer normal mode transformation."""
        return self.__transform_type

    @transform_type.setter
    def transform_type(self, t):
        assert (t == 'matrix' or t == 'fft'), \
               ("Type of ring polymer transformation not found.")
        self.__transform_type = t

    def to_RingPolymer_normalModes(self, X, i=None):
        """
        Transform to ring polymer normal mode representation.

        Parameters
        ----------
        X : (n_dof, n_beads) ndarray of floats.
            'normal' representation or the ring polymer coordinates or momenta.
            The first axis is the physical degrees of freedom, the second one
            the beads.
        i : integer, optional, default None
            Only do transformation in i-th degree of freedom.

        Returns
        -------
        NM : (n_dof, n_beads) ndarray of floats.
            Normal mode representation of the ring polymer coordinates or
            momenta. The first axis is the physical degrees of freedom, the
            second one the beads.
        """

        assert (isinstance(X, np.ndarray)), "X not a numpy array!"
        assert (X.ndim == 2), "X rray not two-dimensional!"
        assert (X.dtype == 'float64'), "X array not real!"

        NM = X.copy()
        if self.n_beads == 1:
            return NM

        if i is not None:
            if self.transform_type == 'matrix':
                NM[i] = self._1d_to_nm_using_matrix(X[i])
            else:
                NM[i] = self._1d_to_nm_using_fft(X[i])
        else:
            if self.transform_type == 'matrix':
                for k, x_k in enumerate(X):
                    NM[k] = self._1d_to_nm_using_matrix(x_k)
            else:
                for k, x_k in enumerate(X):
                    NM[k] = self._1d_to_nm_using_fft(x_k)

        return NM

    def from_RingPolymer_normalModes(self, NM, i=None):
        """
        Transform from ring polymer normal mode representation.

        Parameters
        ----------
        NM : (n_dof, n_beads) ndarray of floats
            Normal mode representation or the ring polymer coordinates or
            momenta. The first axis is the physical degrees of freedom, the
            second one the beads.
        i : integer, optional, default None
            Only do transformation in i-th degree of freedom.

        Returns
        -------
        X : (n_dof, n_beads) ndarray of floats
            'normal' representation of the ring polymer coordinates or momenta.
            The first axis is the physical degrees of freedom, the second one
            the beads.
        """

        assert (isinstance(NM, np.ndarray)), "NM not a numpy array!"
        assert (NM.ndim == 2), "NM rray not two-dimensional!"
        assert (NM.dtype == 'float64'), "NM array not real!"

        X = NM.copy()
        if self.n_beads == 1:
            return X

        if i is not None:
            if self.transform_type == 'matrix':
                X[i] = self._1d_from_nm_using_matrix(NM[i])
            else:
                X[i] = self._1d_from_nm_using_fft(NM[i])
        else:
            if self.transform_type == 'matrix':
                for k, nm in enumerate(NM):
                    X[k] = self._1d_from_nm_using_matrix(nm)
            else:
                for k, nm in enumerate(NM):
                    X[k] = self._1d_from_nm_using_fft(nm)

        return X

    def _1d_to_nm_using_matrix(self, x):
        """
        Transform to ring polymer normal mode representation in one dimension
        using transformation matrix.

        Parameters
        ----------
        x : (n_beads) ndarray of floats
            'normal' representation or the ring polymer in one dimension.

        Returns
        -------
        (n_beads) ndarray of floats
            Normal mode representation of the ring polymer in one dimension.
        """
        return np.matmul(self.C_matrix, x)

    def _1d_from_nm_using_matrix(self, nm):
        """
        Transform from ring polymer normal mode representation in one dimension
        using transformation matrix.

        Parameters
        ----------
        nm : (n_beads) ndarray of floats
            Normal mode representation or the ring polymer in one dimension.

        Returns
        -------
        (n_beads) ndarray of floats
            'normal' representation of the ring polymer in one dimension.
        """
        # Obtaining inverse matrix using the fact that C is unitary and real
        C_inv_mat = np.transpose(self.C_matrix)
        return np.matmul(C_inv_mat, nm)

    def _1d_to_nm_using_fft(self, x):
        """
        Transform to ring polymer normal mode representation in one dimension
        using reordered and scaled FFT.

        Parameters
        ----------
        x : (n_beads) ndarray of floats
            'normal' representation or the ring polymer in one dimension.

        Returns
        -------
        (n_beads) ndarray of floats
            Normal mode representation of the ring polymer in one dimension.
        """
        n = self.n_beads

        if n == 1:
            return x

        after_fft = fft.rfft(x) / math.sqrt(n)
        after_fft[1:-1] *= math.sqrt(2.0)

        reorder_index = [0, 1] + list(range(3, n+1, 2)) + list(range(n-2, 1, -2))

        return after_fft[reorder_index]

    def _1d_from_nm_using_fft(self, nm):
        """
        Transform from ring polymer normal mode representation in one dimension
        using reordered and scaled FFT.

        Parameters
        ----------
        nm : (n_beads) ndarray of floats
            Normal mode representation or the ring polymer in one dimension.

        Returns
        -------
        (n_beads) ndarray of floats
            'normal' representation of the ring polymer in one dimension.
        """

        n = self.n_beads

        if n == 1:
            return nm

        l1 = list(range(n-1, n//2, -1))
        l2 = list(range(2, n//2+1))
        reorder_index = [0, 1] + list(itertools.chain(*zip(l1, l2)))

        fft_input = nm[reorder_index]
        fft_input *= math.sqrt(n)
        fft_input[1:-1] /= math.sqrt(2.0)

        return fft.irfft(fft_input)

    def get_normal_mode_matrix(self):
        """
        Get the transformation matrix to change to ring polymer normal modes in
        one dimension. This is initialized as done in JCP 133, 124104 (2010),
        except that C_kj matrix is created for proper matrix multiplication
        with x_j and that j= 0 to n-1 here instead of j= 1 to n as done in
        paper which simply corresponds to rotating the ring polymer which has
        no effect in the dynamics.

        Returns
        -------
        C_mat : (n_beads, n_beads) ndarray of floats
            Ring polymer normal mode tranformation matrix
        """

        n = self.n_beads

        C_mat = np.zeros((n, n))
        for k in range(n):
            for j in range(n):
                if (k == 0):
                    C_mat[k][j] = math.sqrt(1/n)
                elif ((k >= 1) and (k < n/2)):
                    C_mat[k][j] = np.sqrt(2 / n) * np.cos(2 *np.pi * j * k / n)
                elif(k == n / 2):
                    C_mat[k][j] = np.sqrt(1/n) * (-1)**(j)
                else:
                    C_mat[k][j] = np.sqrt(2 / n) * np.sin(2 *np.pi * j * k / n)

        return C_mat


def sample_free_rp_momenta(nb, mass, beta, centroid=None, 
                           NMtransform_type='matrix'):
    """
    Sample momenta from free ring polymer distribution

    Parameters
    ----------
    nb : int
        Number of beads for a particular degree of freedom
    mass : float
        Mass for one degree of freedom
    beta : float
        Inverse temperature in a.u.
    centroid : float
        Centroid momenta of ring polymer in a.u. Default: None
    NMtransform_type : string
        Type of ring polymer normal mode transformation to be used; this can
        be 'matrix' or 'fft'. Default: 'matrix'

    Returns
    ----------
    (nb) ndarray of floats
        Sampled bead momenta in a.u.
    """
    # Remark: This can also equivalently be done in Cartesian representation
    #         instead of normal mode and then simply shifting by centroid value

    stdev_p = np.sqrt(mass * float(nb) / beta)

    p_nm = [np.random.normal(0, stdev_p) for i in range(1, nb)]
    if centroid is not None:
        p_nm.insert(0, centroid * np.sqrt(nb))
    else:
        p_nm.insert(0, np.random.normal(0, stdev_p))
    p_nm = np.array(p_nm)

    RP_nm_transform = RingPolymerTransformations([nb], NMtransform_type)
    p_arr = RP_nm_transform.from_RingPolymer_normalModes(p_nm.reshape(1, -1))
    return p_arr.flatten()


def sample_free_rp_coord(nb, mass, beta, centroid, NMtransform_type='matrix'):
    """
    Sample coordinates from free ring polymer distribution

    Parameters
    ----------
    nb : int
        Number of beads for a particular degree of freedom
    mass : float
        Mass for one degree of freedom
    beta : float
        Inverse temperature in a.u.
    centroid : float
        Centroid coordinate of ring polymer in a.u.
    NMtransform_type : string
        Type of ring polymer normal mode transformation to be used; this can
        be 'matrix' or 'fft'. Default: 'matrix'

    Returns
    ----------
    (nb) ndarray of floats
        Sampled bead coordinates in a.u.
    """

    # Ring polymer frequency in a.u. with h_bar = 1
    omega_n = float(nb) / beta
    # Array with normal mode frequencies
    omega_nm_arr = np.array([(2 * omega_n * np.sin(i * np.pi / float(nb)))
                             for i in range(0, nb)])

    # Standard deviation for all normal modes except centroid
    stdev_x_nm = [np.sqrt(float(nb) / (beta * mass * omega_nm_arr[i]**2))
                  for i in range(1, nb)]

    nm = [(np.random.normal(0, i)) for i in stdev_x_nm]
    # Adding centroid value from input
    nm.insert(0, centroid * np.sqrt(float(nb)))
    nm = np.array(nm)

    RP_nm_transform = RingPolymerTransformations([nb], NMtransform_type)
    x_arr = RP_nm_transform.from_RingPolymer_normalModes(nm.reshape(1, -1))
    return x_arr.flatten()
