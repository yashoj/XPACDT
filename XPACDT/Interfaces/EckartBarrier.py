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

""" This module represents a one dimensional Eckart barrier."""

import math
import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate


class EckartBarrier(itemplate.PotentialInterface):
    """
    One-dimensional Eckart barrier with parameters `A`, `B`, `L` of the form:

    :math:`V(x) = A*y / (1+y) + B*y / (1+y)^2` 

    :math:`y = \exp(x / L)`

    :math:`V(-\infty) = 0; V(\infty) = A`

    :math:`x_{max} = L \ln(-(A+B) / (A-B))`

    :math:`V(x_{max}) = (A+B)^2 / 4B`

    `A` is the energy difference between reactants and products and equals `d`.

    `h` is the barrier height coming from the reactant side. It can be
    calculated as (A+B)^2 / 4B.

    `B` can be obtained from `h` and `d` as :math:`h + (h-d) + \sqrt(h*(h-d))`. As `h`
    is the barrier height from the reactants and `h-d` is the barrier height
    from the products, `B` can be obtaied as the square of the sum of the
    squareroots of the barrier heights. :math:`B = (\sqrt(D V_r) + \sqrt(D V_p))^2`.

    Alternative parameters:
        `w` (barrier frequency), `h` (barrier height), `d` (energy difference for
          reactants and products), `m` (mass of particle)

    `w` is the barrier frequency and can be obtained from the second derivative
    at the potential maximum, which is :math:`F = (2 / L^2) * h * (h-d) / B`. `w` is
    then :math:`w = \sqrt(|F| / m) / 2\pi`. `L` is this obtained from `w` and `m` as
    :math:`L = (\sqrt(h * (h-d)) / \sqrt(B)) / (\sqrt(2) \pi \sqrt(m) w)`.

    Other Parameters
    ----------------
    A, B, L : floats
        Parameters for the Eckart barrier.
        `A`, `B` are energies in au. `L` is a length in bohr.
    w, h, d, m : floats
        Alternative parameters for the Eckart barrier.
        `w` is a frequency in au. `h`, `d` are energies in au. `m` is a mass in au.
    """
    def __init__(self, **kwargs):
        if {'A', 'B', 'L'} <= set(kwargs):
            try:
                self.__A = float(kwargs.get('A'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'A' for Eckart "
                              "barrier not convertable to float. A is "
                              + kwargs.get('A'))
            try:
                self.__B = float(kwargs.get('B'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'B' for Eckart "
                              "barrier not convertable to float. B is "
                              + kwargs.get('B'))
            try:
                self.__L = float(kwargs.get('L'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'L' for Eckart "
                              "barrier not convertable to float. L is "
                              + kwargs.get('L'))

        elif {'w', 'h', 'd', 'm'} <= set(kwargs):
            try:
                w = float(kwargs.get('w'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'w' for Eckart "
                              "barrier not convertable to float. w is "
                              + kwargs.get('w'))
            try:
                h = float(kwargs.get('h'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'h' for Eckart "
                              "barrier not convertable to float. h is "
                              + kwargs.get('h'))
            try:
                d = float(kwargs.get('d'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'd' for Eckart "
                              "barrier not convertable to float. d is "
                              + kwargs.get('d'))

            try:
                m = float(kwargs.get('m'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'm' for Eckart "
                              "barrier not convertable to float. m is "
                              + kwargs.get('m'))

            # conversion here!
            self.__A = d
            self.__B = (math.sqrt(h) + math.sqrt(h-d))**2
            self.__L = math.sqrt(2.0*h*(h-d)) / (w * math.sqrt(m) * math.sqrt(self.__B))
        else:
            raise RuntimeError("XPACDT: Parameters for Eckart barrier not "
                               "given properly. Either give A, B, L or give "
                               "w, h, d.")

        assert(self.__A <= 0.0), "A not zero or less!"
        assert(self.__B > 0.0), "B not positive!"
        assert(self.__L > 0.0), "L not positive!"

        itemplate.PotentialInterface.__init__(self, "EckartBarrier")

    @property
    def A(self):
        """float : A parameter for Eckart barrier."""
        return self.__A

    @property
    def B(self):
        """float : B parameter for Eckart barrier."""
        return self.__B

    @property
    def L(self):
        """float : L parameter for Eckart barrier."""
        return self.__L

    def _calculate_all(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
        P : (n_dof, n_beads) ndarray of floats, optional
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

        # centroid part if more than 1 bead
        if R.shape[1] > 1:
            centroid = np.mean(R, axis=1)
            y = np.exp(centroid / self.L)
            y_plus = 1 + y
            self._gradient_centroid = np.zeros_like(y)
            self._energy_centroid = np.zeros_like(y)

            self._energy_centroid = self.A * y / y_plus + self.B * y / y_plus**2
            self._gradient_centroid = (self.A / self.L) * y / y_plus**2 \
                + (self.B / self.L) * y*(1-y) / y_plus**3

        # beads part
        y = np.exp(R[0] / self.L)
        y_plus = 1 + y
        self._gradient = np.zeros_like(y)
        self._energy = np.zeros_like(y)

        self._energy = self.A * y / y_plus + self.B * y / y_plus**2
        self._gradient = (self.A / self.L) * y / y_plus**2 \
                         + (self.B / self.L) * y*(1-y) / y_plus**3

        self._gradient = self._gradient.reshape((1, -1))

        if R.shape[1] == 1:
            self._energy_centroid = self._energy
            self._gradient_centroid = self._gradient

        return
