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

from XPACDT.Input.Error import XPACDTInputError


class EckartBarrier(itemplate.PotentialInterface):
    r"""
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
    def __init__(self, n_dof=1, **parameters):
        if n_dof != 1:
            raise XPACDTInputError(
                f"Inferred number of degree of freedom is {n_dof}, but "
                "should be 1 for Eckart barrier.",
                section="EckartBarrier")
    
        super().__init__("EckartBarrier",
                         n_dof=1, n_states=1, primary_basis='adiabatic',
                         **parameters)

        pes_parameters = parameters.get(self.name)
        if {'A', 'B', 'L'} <= set(pes_parameters):
            parameters = []
            for key in ('A', 'B', 'L'):
                try:
                    p = float(pes_parameters[key])
                except ValueError as e:
                    raise XPACDTInputError(
                        f"Parameter '{key}' for Eckart barrier not "
                        "convertible to float.",
                        section="EckartBarrier",
                        key=key,
                        given=pes_parameters[key],
                        caused_by=e)
                parameters.append(p)

            self.__A, self.__B, self.__L = parameters

        elif {'w', 'h', 'd', 'm'} <= set(pes_parameters):
            parameters = []
            for key in ('w', 'h', 'd', 'm'):
                try:
                    p = float(pes_parameters[key])
                except ValueError as e:
                    raise XPACDTInputError(
                        f"Parameter '{key}' for Eckart barrier not "
                        "convertible to float.",
                        section="EckartBarrier",
                        key=key,
                        given=pes_parameters[key],
                        caused_by=e)
                parameters.append(p)

            w, h, d, m = parameters

            # conversion here!
            self.__A = d
            self.__B = (math.sqrt(h) + math.sqrt(h-d))**2
            self.__L = math.sqrt(2.0*h*(h-d)) / (w * math.sqrt(m) * math.sqrt(self.__B))
        else:
            raise XPACDTInputError(
                "Parameters for Eckart barrier not given properly. Either "
                "give 'A', 'B', 'L' or give 'w', 'h', 'd' and 'm'.",
                section="EckartBarrier",
                given=pes_parameters)

        if (self.__A > 0.0):
            raise XPACDTInputError(
                "'A' must be non positive. This may happen after conversion "
                "from the 'w', 'h', 'd' and 'm' parameters.",
                section="EckartBarrier",
                key="A",
                given=self.__A)

        if (self.__B <= 0.0):
            raise XPACDTInputError(
                "'B' must be strictly positive. This may happen after "
                "conversion from the 'w', 'h', 'd' and 'm' parameters.",
                section="EckartBarrier",
                key="B",
                given=self.__B)

        if (self.__L <= 0.0):
            raise XPACDTInputError(
                "'L' must be strictly positive. This may happen after "
                "conversion from the 'w', 'h', 'd' and 'm' parameters.",
                section="EckartBarrier",
                key="L",
                given=self.__L)

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
