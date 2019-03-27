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

import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate


class OneDPolynomial(itemplate.Interface):

    def __init__(self, **kwargs):
        try:
            self.__x0 = float(kwargs.get('x0', 0.0))
        except ValueError:
            print("Parameter 'x0' for polynomials not convertable to floats."
                  " x0 is " + kwargs.get('x0'))
            raise

        assert (isinstance(kwargs.get('a'), str)), \
            "Parameters 'a' for polynomials not given or not given as " \
            "string."
        try:
            self.__as = [float(f) for f in kwargs.get('a').split()]
        except ValueError:
            print("Parameters 'a' for polynomials not convertable to floats."
                  " a is " + kwargs.get('a'))
            raise

        itemplate.Interface.__init__(self, "OneDPolynomial")

    @property
    def a(self):
        """ndarray of floats : Expansion coefficients for the polynomial. """
        return self.__as

    @property
    def x0(self):
        """float : The equilibrium position. Default is x0=0. """
        return self.__x0

    def _calculate(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        R : two-dimensional ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
        P : two-dimensional ndarray of floats, optional
            The momenta of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads. This is not
            used in this potential and thus defaults to None.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        if P is not None:
            assert (isinstance(P, np.ndarray)), "P not a numpy array!"
            assert (P.ndim == 2), "Momentum array not two-dimensional!"
            assert (P.dtype == 'float64'), "Momentum array not real!"

        n = R.shape[1]

        self._energy, self._gradient = self.__polynomial(R)
        self._energy += np.array([self.a[0]]*n)
        self._gradient = self._gradient.reshape((-1, n))

        return

    def __polynomial(self, R):
        """
        Function to calculate the one-D polynomial that can be vectorized.

        Parameters:
        -----------
        R : two-dimensional ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.

        Returns:
        --------
        energy : ndarray of floats
            The potential energy of each bead.
        gradient : ndarray of floats
            The gradient along the one dimension of each bead.
        """

        distance = R[0] - self.x0
        power = np.ones_like(distance)

        gradient = np.zeros_like(distance)
        energy = np.zeros_like(distance)

        for i, a in enumerate(self.a[1:]):
            gradient += float(i+1) * a * power

            power *= distance
            energy += a * power

        return energy, gradient
