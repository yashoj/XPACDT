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

""" This module represents a one dimensional polynomial potential."""

import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate


class OneDPolynomial(itemplate.PotentialInterface):
    """
    One-dimensional polynomial potential of the form:
    :math:`V(x) = \\sum_{i=0}^{N} a_i (x-x_0)^i`.

    Other Parameters
    ----------------
    x0 : float or string of float
        Equilibrium position of the polynomial.
    a : string containing several floats
        The expansion coefficients for the polynomial in ascending order. The
        expansion length is determined by the number of given coefficients
        here.
    """

    def __init__(self, max_n_beads, basis='adiabatic', **kwargs):

        assert (basis == 'adiabatic'), \
        ("Electronic basis for one dimensional polynomial potential should be adiabatic.")

        itemplate.PotentialInterface.__init__(self, "OneDPolynomial", 1,
                                              max_n_beads, 1, basis)

        try:
            self.__x0 = float(kwargs.get('x0', 0.0))
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameter 'x0' for polynomials "
                                   "not convertable to floats. x0 is "
                                   + kwargs.get('x0'))

        assert (isinstance(kwargs.get('a'), str)), \
            "Parameters 'a' for polynomials not given or not given as " \
            "string."
        try:
            self.__as = [float(f) for f in kwargs.get('a').split()]
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameters 'a' for polynomials "
                                   "not convertable to floats."
                                   " a is " + kwargs.get('a'))

    @property
    def a(self):
        """(N) ndarray of floats : Expansion coefficients for the polynomial
        of degree N-1"""
        return self.__as

    @property
    def x0(self):
        """float : The equilibrium position. Default is x0=0. """
        return self.__x0

    def _calculate_all(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system in au. The first axis represents the degrees of freedom and
            the second axis is the beads. `P` is not used in this potential
            and thus defaults to None.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        # centroid part if more than 1 bead
        if self.max_n_beads > 1:
            centroid = np.mean(R, axis=1)
            distance_centroid = centroid[0] - self.x0
            power_centroid = 1.0
            self._gradient_centroid = np.zeros_like(distance_centroid)
            self._energy_centroid = np.zeros_like(distance_centroid) + self.a[0]

        # beads part
        distance = R[0] - self.x0
        power = np.ones_like(distance)
        self._gradient = np.zeros_like(distance)
        self._energy = np.zeros_like(distance) + self.a[0]

        for i, a in enumerate(self.a[1:]):
            # beads part
            self._gradient += float(i+1) * a * power
            power *= distance
            self._energy += a * power

            # centroid part if more than 1 bead
            if self.max_n_beads > 1:
                self._gradient_centroid += float(i+1) * a * power_centroid
                power_centroid *= distance_centroid
                self._energy_centroid += a * power_centroid

        self._energy = self._energy.reshape((1, -1))
        self._gradient = self._gradient.reshape((1, 1, -1))

        if self.max_n_beads == 1:
            self._energy_centroid = self._energy[:, 0]
            self._gradient_centroid = self._gradient[:, :, 0]
        else:
            self._energy_centroid = self._energy_centroid.reshape((-1))
            self._gradient_centroid = self._gradient_centroid.reshape((1, -1))

        return
