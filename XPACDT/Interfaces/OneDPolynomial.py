#  **************************************************************************
#
#  XPACDT, eXtended PolyAtomic Chemical Dynamics Toolkit
#  XPACDT is a program that can treat polyatomic chemical dynamics problems
#  by solving Newtons equations of motion directly or in an
#  extended ring-polymer phase space. Non-Born-Oppenheimer effects are
#  included employ different approaches, including fewest switches surface
#  hopping.
#
#  Copyright (C) 2019, 2020
#  Ralph Welsch, DESY, <ralph.welsch@desy.de>
#  Yashoj Shakya, DESY, <yashoj.shakya@desy.de>
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

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.

    Other Parameters (as given in the input file)
    ----------------
    x0 : float or string of float
        Equilibrium position of the polynomial.
    a : string containing several floats
        The expansion coefficients for the polynomial in ascending order. The
        expansion length is determined by the number of given coefficients
        here.
    """

    def __init__(self, parameters, **kwargs):

        itemplate.PotentialInterface.__init__(self, "OneDPolynomial", 1, 1,
                                              max(parameters.n_beads),
                                              'adiabatic')

        pes_parameters = parameters.get(self.name)

        try:
            self.__x0 = float(pes_parameters.get('x0', 0.0))
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameter 'x0' for polynomials "
                                   "not convertable to floats. x0 is "
                                   + pes_parameters.get('x0'))

        if ('a' not in pes_parameters):
            raise KeyError("\nXPACDT: Parameters 'a' for polynomials not"
                           " given in the input.")
        try:
            self.__as = [float(f) for f in pes_parameters.get('a').split()]
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameters 'a' for polynomials "
                                   "not convertable to floats."
                                   " a is " + pes_parameters.get('a'))

    @property
    def a(self):
        """(N) ndarray of floats : Expansion coefficients for the polynomial
        of degree N-1"""
        return self.__as

    @property
    def x0(self):
        """float : The equilibrium position. Default is x0=0. """
        return self.__x0

    def _calculate_adiabatic_all(self, R, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` representing the
            system in au. The first axis represents the degrees of freedom and
            the second axis is the beads.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        # centroid part if more than 1 bead
        if self.max_n_beads > 1:
            centroid = np.mean(R, axis=1)
            distance_centroid = centroid[0] - self.x0
            power_centroid = 1.0
            self._adiabatic_gradient_centroid = np.zeros(1)
            self._adiabatic_energy_centroid = np.zeros(1) + self.a[0]

        # beads part
        distance = R[0] - self.x0
        power = np.ones(self.max_n_beads)

        self._adiabatic_gradient = np.zeros(self.max_n_beads)
        self._adiabatic_energy = np.zeros(self.max_n_beads) + self.a[0]

        for i, a in enumerate(self.a[1:]):
            # beads part
            self._adiabatic_gradient += float(i+1) * a * power
            power *= distance
            self._adiabatic_energy += a * power

            # centroid part if more than 1 bead
            if self.max_n_beads > 1:
                self._adiabatic_gradient_centroid += float(i+1) * a * power_centroid
                power_centroid *= distance_centroid
                self._adiabatic_energy_centroid += a * power_centroid

        self._adiabatic_energy = self._adiabatic_energy.reshape((1, -1))
        self._adiabatic_gradient = self._adiabatic_gradient.reshape((1, 1, -1))

        if self.max_n_beads == 1:
            self._adiabatic_energy_centroid = self._adiabatic_energy[:, 0]
            self._adiabatic_gradient_centroid = self._adiabatic_gradient[:, :, 0]
        else:
            self._adiabatic_energy_centroid = self._adiabatic_energy_centroid.reshape((-1))
            self._adiabatic_gradient_centroid = self._adiabatic_gradient_centroid.reshape((1, -1))

        return
