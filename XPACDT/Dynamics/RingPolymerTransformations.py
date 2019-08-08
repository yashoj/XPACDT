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
    nbeads : (n_dof) list of int
        The number of beads for each degree of freedom.
    transform_type : {'matrix', 'fft'}
        Type of ring polymer normal mode transformation to be used.
        Default: 'matrix'

    Attributes:
    -----------
    n_beads
    transform_type
    C_matrices : dictionary or None
        Normal mode transformation matrices for all different number of beads
        present in 'n_beads'; the keys are the number of beads
        (optional depending upon if transform_type=='matrix'; else if
        transform_type=='fft', it is None)

    """

    def __init__(self, n_beads, transform_type='matrix'):

        self.n_beads = n_beads
        self.transform_type = transform_type

        if (self.transform_type == 'matrix'):
            self.C_matrices = self.get_normal_mode_matrix()
        else:
            self.C_matrices = None
        return

    @property
    def n_beads(self):
        """list of ints : Number of beads for each degrees of freedom."""
        return self.__n_beads

    @n_beads.setter
    def n_beads(self, n):
        self.__n_beads = n

    @property
    def transform_type(self):
        """{'matrix', 'fft'} : Type of ring polymer normal mode transformation."""
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
        if np.all([n == 1 for n in self.n_beads]):
            return NM

        if i is not None:
            if self.transform_type == 'matrix':
                NM[i] = self._1d_to_nm_using_matrix(X[i], self.n_beads[i])
            else:
                NM[i] = self._1d_to_nm_using_fft(X[i], self.n_beads[i])
        else:
            if self.transform_type == 'matrix':
                for k, x_k in enumerate(X):
                    NM[k] = self._1d_to_nm_using_matrix(x_k, self.n_beads[k])
            else:
                for k, x_k in enumerate(X):
                    NM[k] = self._1d_to_nm_using_fft(x_k, self.n_beads[k])

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
        if np.all([n == 1 for n in self.n_beads]):
            return X

        if i is not None:
            if self.transform_type == 'matrix':
                X[i] = self._1d_from_nm_using_matrix(NM[i], self.n_beads[i])
            else:
                X[i] = self._1d_from_nm_using_fft(NM[i], self.n_beads[i])
        else:
            if self.transform_type == 'matrix':
                for k, nm in enumerate(NM):
                    X[k] = self._1d_from_nm_using_matrix(nm, self.n_beads[k])
            else:
                for k, nm in enumerate(NM):
                    X[k] = self._1d_from_nm_using_fft(nm, self.n_beads[k])

        return X

    def _1d_to_nm_using_matrix(self, x, nb):
        """
        Transform to ring polymer normal mode representation in one dimension
        using transformation matrix.

        Parameters
        ----------
        x : (nb) ndarray of floats
            'normal' representation or the ring polymer in one dimension.
        nb : int
            Number of beads for a particular degree of freedom

        Returns
        -------
        (nb) ndarray of floats
            Normal mode representation of the ring polymer in one dimension.
        """
        return np.matmul(self.C_matrices[nb], x)

    def _1d_from_nm_using_matrix(self, nm, nb):
        """
        Transform from ring polymer normal mode representation in one dimension
        using transformation matrix.

        Parameters
        ----------
        nm : (nb) ndarray of floats
            Normal mode representation or the ring polymer in one dimension.
        nb : int
            Number of beads for a particular degree of freedom

        Returns
        -------
        (nb) ndarray of floats
            'normal' representation of the ring polymer in one dimension.
        """
        # Obtaining inverse matrix using the fact that C is unitary and real
        C_inv_mat = np.transpose(self.C_matrices[nb])
        return np.matmul(C_inv_mat, nm)

    def _1d_to_nm_using_fft(self, x, nb):
        """
        Transform to ring polymer normal mode representation in one dimension
        using reordered and scaled FFT.

        Parameters
        ----------
        x : (nb) ndarray of floats
            'normal' representation or the ring polymer in one dimension.
        nb : int
            Number of beads for a particular degree of freedom

        Returns
        -------
        (nb) ndarray of floats
            Normal mode representation of the ring polymer in one dimension.
        """
        if nb == 1:
            return x

        after_fft = fft.rfft(x) / math.sqrt(nb)
        after_fft[1:-1] *= math.sqrt(2.0)

        reorder_index = [0, 1] + list(range(3, nb+1, 2)) \
                               + list(range(nb-2, 1, -2))

        return after_fft[reorder_index]

    def _1d_from_nm_using_fft(self, nm, nb):
        """
        Transform from ring polymer normal mode representation in one dimension
        using reordered and scaled FFT.

        Parameters
        ----------
        nm : (nb) ndarray of floats
            Normal mode representation or the ring polymer in one dimension.
        nb : int
            Number of beads for a particular degree of freedom

        Returns
        -------
        (nb) ndarray of floats
            'normal' representation of the ring polymer in one dimension.
        """
        if nb == 1:
            return nm

        l1 = list(range(nb-1, nb//2, -1))
        l2 = list(range(2, nb//2+1))
        reorder_index = [0, 1] + list(itertools.chain(*zip(l1, l2)))

        fft_input = nm[reorder_index]
        fft_input *= math.sqrt(nb)
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
        C_dict : dictionary
            Ring polymer normal mode tranformation matrices for each distinct
            i-th element in 'n_beads' as keys and their transformation matrix
            (n_beads[i], n_beads[i]) ndarray of floats as values
        """
        C_dict = {}

        for n in self.n_beads:
            if n not in C_dict.keys():
                C_mat = np.zeros((n, n))
                for k in range(n):
                    for j in range(n):
                        if (k == 0):
                            C_mat[k][j] = math.sqrt(1/n)
                        elif ((k >= 1) and (k < n/2)):
                            C_mat[k][j] = math.sqrt(2 / n)\
                                          * math.cos(2 * math.pi * j * k / n)
                        elif(k == n / 2):
                            C_mat[k][j] = math.sqrt(1/n) * (-1)**(j)
                        else:
                            C_mat[k][j] = math.sqrt(2 / n)\
                                          * math.sin(2 * math.pi * j * k / n)
                C_dict[n] = C_mat.copy()

        return C_dict

    def sample_free_rp_momenta(self, nb, mass, beta, centroid=None):
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

        Returns
        ----------
        p_arr : (nb) ndarray of floats
            Sampled bead momenta in a.u.
        """
        # Remark: This can also equivalently be done in Cartesian coordinates
        #         instead of normal mode and then shift by centroid value

        stdev_p = math.sqrt(mass * float(nb) / beta)

        p_nm = [np.random.normal(0, stdev_p) for i in range(1, nb)]
        if centroid is not None:
            p_nm.insert(0, centroid * np.sqrt(nb))
        else:
            p_nm.insert(0, np.random.normal(0, stdev_p))
        p_nm = np.array(p_nm)

        if self.transform_type == 'matrix':
            p_arr = self._1d_from_nm_using_matrix(p_nm, nb)
        else:
            p_arr = self._1d_from_nm_using_fft(p_nm, nb)

        return p_arr

    def sample_free_rp_coord(self, nb, mass, beta, centroid):
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

        Returns
        ----------
        x_arr : (nb) ndarray of floats
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

        x_nm = [(np.random.normal(0, i)) for i in stdev_x_nm]
        # Adding centroid value from input
        x_nm.insert(0, centroid * np.sqrt(float(nb)))
        x_nm = np.array(x_nm)

        if self.transform_type == 'matrix':
            x_arr = self._1d_from_nm_using_matrix(x_nm, nb)
        else:
            x_arr = self._1d_from_nm_using_fft(x_nm, nb)
        return x_arr
