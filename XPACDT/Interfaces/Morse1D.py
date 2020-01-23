#!/usr/bin/env python3

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

""" This module represents a one dimensional morse potential."""

import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate


class Morse1D(itemplate.PotentialInterface):
    """
    One-dimensional morse potential of the form:
    :math:`V(x) = D_e (1- \text{e}^{-a(x - r_e)})^2 - b`.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.

    Other Parameters
    ----------------
    De : float or string of float
        Defines the well depth of the potential.
    a : float or string of float
        Controls width of potential. Lower `a` value results in wider potential.
    re : float or string of float
        Equilibrium position of the potential.
    b : float or string of float
        Overall vertical shift to the potential.
    """

    def __init__(self, max_n_beads=1, **kwargs):

        itemplate.PotentialInterface.__init__(self, "Morse1D", 1, 1,
                                              max(parameters.n_beads),
                                              'adiabatic')

        pes_parameters = parameters.get(self.name)

        try:
            self.__De = float(pes_parameters.get('De'))
        except (TypeError, ValueError):
            print("\nXPACDT: Parameter 'De' for morse potential not given or "
                  "not convertible to float.\n")
            raise

        try:
            self.__a = float(pes_parameters.get('a'))
        except (TypeError, ValueError):
            print("\nXPACDT: Parameter 'a' for morse potential not given or "
                  "not convertible to float.\n")
            raise

        try:
            self.__re = float(pes_parameters.get('re'))
        except (TypeError, ValueError):
            print("\nXPACDT: Parameter 're' for morse potential not given or "
                  "not convertible to float.\n")
            raise

        try:
            self.__b = float(pes_parameters.get('b'))
        except (TypeError, ValueError):
            print("\nXPACDT: Parameter 'b' for morse potential not given or "
                  "not convertible to float.\n")
            raise

    @property
    def De(self):
        """float : Well depth of the potential."""
        return self.__De

    @property
    def a(self):
        """float : Width of the potential. """
        return self.__a

    @property
    def re(self):
        """float : Equilibrium position of the potential."""
        return self.__re

    @property
    def b(self):
        """float : Overall vertical shift to the potential. """
        return self.__b

    def _calculate_adiabatic_all(self, R, P=None, S=None):
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

        self._adiabatic_energy[0], self._adiabatic_gradient[0] = \
            self._get_V_grad(R)

        # Centroid part
        if self.max_n_beads == 1:
            self._adiabatic_energy_centroid = (
                self._adiabatic_energy.reshape(self.n_states)).copy()
            self._adiabatic_gradient_centroid = (
                self._adiabatic_gradient.reshape((self.n_states, self.n_dof))).copy()
        else:
            r_centroid = np.mean(R, axis=1)
            self._adiabatic_energy_centroid[0], self._adiabatic_gradient_centroid[0] = \
                self._get_V_grad(r_centroid)

        return

    def _get_V_grad(self, R):
        """
        Get adiabatic energy and gradient for either all beads or centroid.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        V : float /or/ (n_beads) ndarrays of floats
            Diagonal diabatic energy term.
        dV : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Diagonal diabatic gradient term.
        """
        exp_term = np.exp(-self.__a * (R - self.__re))
        V = self.__De * (1. - exp_term[0])**2 - self.__b
        dV = 2. * self.__a * self.__De * exp_term * (1. - exp_term)

        return V, dV


#if __name__ == '__main__':
#
#    # Plotting script to visualize the potential.
#    # Runs only if this file is executed on its own by doing: python Morse1D.py
#    import matplotlib.pyplot as plt
#
#    # Input parameters, taken from: J. Chem. Phys. 150, 114105 (2019)
#    nb = 1
#    De = 0.04556
#    a = 1.94  # 0.97
#    re = 2.50  # 1.70
#    b = 0.04556
#    pot = Morse1D(nb, **{'De': De, 'a': a, 're': re, 'b': b})
#
#    # len(linspace) array of positions
#    X = np.linspace(2., 5., num=100)
#
#    V_ad = []
#    dV_ad = []
#
#    for i in X:
#        pot._calculate_adiabatic_all(np.array([[i]]))
#
#        V_ad.append(pot._adiabatic_energy[0, 0])
#        dV_ad.append(pot._adiabatic_gradient[0, 0, 0])
#
#    # Plot all
#    fig, ax = plt.subplots(2)
#    fig.suptitle('Morse potential: ', fontsize=20)
#
#    ax[0].plot(X, V_ad, 'r-', label="V")
#    ax[0].set_xlabel('x')
#    ax[0].set_ylabel('Adiabatic Potential')
#    ax[0].legend(loc='best')
#    # ax[0].set_ylim((-0.001, 0.05))
#
#    ax[1].plot(X, dV_ad, 'r-', label="dV/dx")
#    ax[1].set_xlabel('x')
#    ax[1].set_ylabel('Derivative of Adiabatic Potential')
#    ax[1].legend(loc='best')
#
#    plt.show()
