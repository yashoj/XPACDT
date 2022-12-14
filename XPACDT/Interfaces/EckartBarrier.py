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

    `B` can be obtained from `h` and `d` as :math:`h + (h-d) + \sqrt(h*(h-d))`.
    As `h` is the barrier height from the reactants and `h-d` is the barrier
    height from the products, `B` can be obtaied as the square of the sum of
    the squareroots of the barrier heights.
    :math:`B = (\sqrt(D V_r) + \sqrt(D V_p))^2`.

    Alternative parameters:
        `w` (barrier frequency), `h` (barrier height), `d` (energy difference
          for reactants and products), `m` (mass of particle)

    `w` is the barrier frequency and can be obtained from the second derivative
    at the potential maximum, which is :math:`F = (2 / L^2) * h * (h-d) / B`.
    `w` is then :math:`w = \sqrt(|F| / m) / 2\pi`. `L` is this obtained from
    `w` and `m` as
    :math:`L = (\sqrt(h * (h-d)) / \sqrt(B)) / (\sqrt(2) \pi \sqrt(m) w)`.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.

    Other Parameters (as given in the input file)
    ----------------
    A, B, L : floats
        Parameters for the Eckart barrier.
        `A`, `B` are energies in au. `L` is a length in bohr.
    w, h, d, m : floats
        Alternative parameters for the Eckart barrier.
        `w` is a frequency in au. `h`, `d` are energies in au.
        `m` is a mass in au.
    """
    def __init__(self, parameters, **kwargs):
        itemplate.PotentialInterface.__init__(self, "EckartBarrier", 1, 1,
                                              max(parameters.n_beads),
                                              'adiabatic')

        pes_parameters = parameters.get(self.name)
        if {'A', 'B', 'L'} <= set(pes_parameters):
            try:
                self.__A = float(pes_parameters.get('A'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'A' for Eckart "
                              "barrier not convertable to float. A is "
                              + pes_parameters.get('A'))
            try:
                self.__B = float(pes_parameters.get('B'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'B' for Eckart "
                              "barrier not convertable to float. B is "
                              + pes_parameters.get('B'))
            try:
                self.__L = float(pes_parameters.get('L'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'L' for Eckart "
                              "barrier not convertable to float. L is "
                              + pes_parameters.get('L'))

        elif {'w', 'h', 'd', 'm'} <= set(pes_parameters):
            try:
                w = float(pes_parameters.get('w'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'w' for Eckart "
                              "barrier not convertable to float. w is "
                              + pes_parameters.get('w'))
            try:
                h = float(pes_parameters.get('h'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'h' for Eckart "
                              "barrier not convertable to float. h is "
                              + pes_parameters.get('h'))
            try:
                d = float(pes_parameters.get('d'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'd' for Eckart "
                              "barrier not convertable to float. d is "
                              + pes_parameters.get('d'))

            try:
                m = float(pes_parameters.get('m'))
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Parameter 'm' for Eckart "
                              "barrier not convertable to float. m is "
                              + pes_parameters.get('m'))

            # conversion here!
            self.__A = d
            self.__B = (math.sqrt(h) + math.sqrt(h-d))**2
            self.__L = math.sqrt(2.0*h*(h-d)) / (w * math.sqrt(m) * math.sqrt(self.__B))
        else:
            raise RuntimeError("XPACDT: Parameters for Eckart barrier not "
                               "given properly. Either give A, B, L or give "
                               "w, h, d.")

        if (self.__A > 0.0):
            raise ValueError("\nXPACDT: A not zero or less!")

        if (self.__B <= 0.0):
            raise ValueError("\nXPACDT: B not positive!")

        if (self.__L <= 0.0):
            raise ValueError("\nXPACDT: L not positive!")

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

    def _calculate_adiabatic_all(self, R, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        # centroid part if more than 1 bead
        if R.shape[1] > 1:
            centroid = np.mean(R, axis=1)
            y = np.exp(centroid / self.L)
            y_plus = 1 + y
            self._adiabatic_gradient_centroid = np.zeros((1, 1))
            self._adiabatic_energy_centroid = np.zeros(1)

            self._adiabatic_energy_centroid[0] = self.A * y / y_plus + self.B * y / y_plus**2
            self._adiabatic_gradient_centroid[0] = (self.A / self.L) * y / y_plus**2 \
                + (self.B / self.L) * y*(1-y) / y_plus**3

        # beads part
        y = np.exp(R[0] / self.L)
        y_plus = 1 + y
        self._adiabatic_gradient = np.zeros_like(y[None, None, :])
        self._adiabatic_energy = np.zeros_like(y[None, :])

        self._adiabatic_energy[0] = self.A * y / y_plus + self.B * y / y_plus**2
        self._adiabatic_gradient[0, 0] = (self.A / self.L) * y / y_plus**2 \
                         + (self.B / self.L) * y*(1-y) / y_plus**3

        if R.shape[1] == 1:
            self._adiabatic_energy_centroid = self._adiabatic_energy[:, 0]
            self._adiabatic_gradient_centroid = self._adiabatic_gradient[:, :, 0]

        return
