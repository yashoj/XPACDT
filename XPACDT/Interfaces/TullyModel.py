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

""" This module represents two state Tully model potentials in one dimension."""

import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Input.Inputfile as infile


class TullyModel(itemplate.PotentialInterface):
    """
    Two state Tully model potentials (A, B and C) in one dimension.
    Reference: J. Chem. Phys. 93 (2), 1061 (1990)

    Parameters
    ----------
    max_n_beads : int, optional
        Maximum number of beads from the (n_dof) list of n_beads. Default: 1.

    Other Parameters
    ----------------
    model_type : {'model_A', 'model_B', 'model_C'}
        String denoting model type to be used.
    """

    def __init__(self, max_n_beads=1, **kwargs):

        itemplate.PotentialInterface.__init__(self, "TullyModel", 1, 2,
                                              max_n_beads, 'diabatic')

        assert (isinstance(kwargs.get('model_type'), str)), \
            "Parameter 'model_type' not given or not given as string."
        self.model_type = kwargs.get('model_type')

        # TODO: Need to add 'try' and setting to default warning?
        if (self.model_type == 'model_A'):
            self.__A = float(kwargs.get('A', 0.01))
            self.__B = float(kwargs.get('B', 1.6))
            self.__C = float(kwargs.get('C', 0.005))
            self.__D = float(kwargs.get('D', 1.0))

        elif (self.model_type == 'model_B'):
            self.__A = float(kwargs.get('A', 0.1))
            self.__B = float(kwargs.get('B', 0.28))
            self.__C = float(kwargs.get('C', 0.015))
            self.__D = float(kwargs.get('D', 0.06))
            self.__Eo = float(kwargs.get('Eo', 0.05))

        elif (self.model_type == 'model_C'):
            self.__A = float(kwargs.get('A', 0.0006))
            self.__B = float(kwargs.get('B', 0.1))
            self.__C = float(kwargs.get('C', 0.9))

    @property
    def model_type(self):
        """string : Model number to be used."""
        return self.__model_type

    @model_type.setter
    def model_type(self, m):
        assert (m in ['model_A', 'model_B', 'model_C']),\
               ("Type of Tully model selected not available.")
        self.__model_type = m

    def _calculate_adiabatic_all(self, R, P=None, S=None):
        """
        Calculate and set diabatic and adiabatic (if required) matrices for
        energies and gradients of beads and centroid.

        Parameters:
        ----------
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

        self._calculate_diabatic_all(R)
        self._get_adiabatic_from_diabatic(R)

    def _calculate_diabatic_all(self, R):
        """
        Calculate and set diabatic matrices for energies and gradients for
        beads and centroid.

        Parameters:
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
        """
        # Bead part
        self._diabatic_energy[0, 0], self._diabatic_gradient[0, 0] = \
            self._get_V_dV_11(R)
        self._diabatic_energy[1, 1], self._diabatic_gradient[1, 1] = \
            self._get_V_dV_22(R)

        # Taking into account that the potential matrix is real and Hermitian
        self._diabatic_energy[0, 1], self._diabatic_gradient[0, 1] = \
            self._get_V_dV_12(R)
        self._diabatic_energy[1, 0] = self._diabatic_energy[0, 1].copy()
        self._diabatic_gradient[1, 0] = self._diabatic_gradient[0, 1].copy()

        # Centroid part
        if self.max_n_beads == 1:
            self._diabatic_energy_centroid = (
                self._diabatic_energy.reshape((self.n_states, self.n_states))).copy()
            self._diabatic_gradient_centroid = (
                self._diabatic_gradient.reshape((self.n_states, self.n_states, self.n_dof))).copy()
        else:
            r_centroid = np.mean(R, axis=1)
            self._diabatic_energy_centroid[0, 0], self._diabatic_gradient_centroid[0, 0] = \
                self._get_V_dV_11(r_centroid)
            self._diabatic_energy_centroid[1, 1], self._diabatic_gradient_centroid[1, 1] = \
                self._get_V_dV_22(r_centroid)

            self._diabatic_energy_centroid[0, 1], self._diabatic_gradient_centroid[0, 1] = \
                self._get_V_dV_12(r_centroid)
            self._diabatic_energy_centroid[1, 0] = \
                self._diabatic_energy_centroid[0, 1]
            self._diabatic_gradient_centroid[1, 0] = \
                self._diabatic_gradient_centroid[0, 1].copy()

    def _get_V_dV_11(self, R):
        """
        Get diagonal diabatic energy and gradient term for first state.

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
        x = R[0]

        if (self.model_type == 'model_A'):
            # Note: np.sign() returns 0 if x == 0. That is still fine here
            #       as it recovers proper limit of V11(0) = 0.
            exp_term = np.exp(-self.__B * np.absolute(R))
            V = np.sign(x) * self.__A * (1. - exp_term[0])
            dV = self.__A * self.__B * exp_term
        elif (self.model_type == 'model_B'):
            # 'np.zeros_like' is not used since simple multiplication seems to
            # be faster at least for small ndarrays.
            zeros_term = 0. * R
            V = zeros_term[0]
            dV = zeros_term
        elif (self.model_type == 'model_C'):
            # This construction is done just to get the proper shape compatible
            # with both float and array. This seems to be faster than 'if'
            # statements combined with 'np.ones_like'
            zeros_term = 0. * R
            V = self.__A * (1. + zeros_term[0])
            dV = zeros_term
        return V, dV

    def _get_V_dV_22(self, R):
        """
        Get diagonal diabatic energy and gradient term for second state.

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
        if (self.model_type == 'model_A' or self.model_type == 'model_C'):
            V, dV = self._get_V_dV_11(R)
            V *= -1.
            if (self.model_type == 'model_A'):
                dV *= -1.
        elif (self.model_type == 'model_B'):
            exp_term = np.exp(-self.__B * R * R)
            V = -self.__A * exp_term + self.__Eo
            dV = 2 * R * self.__A * self.__B * exp_term
        return V, dV

    def _get_V_dV_12(self, R):
        """
        Get off-diagonal diabatic energy and gradient term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        V : float /or/ (n_beads) ndarrays of floats
            Off-diagonal diabatic energy term.
        dV : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Off-diagonal diabatic gradient term.
        """
        x = R[0]

        if (self.model_type == 'model_A' or self.model_type == 'model_B'):
            exp_term = np.exp(-self.__D * R * R)
            V = self.__C * exp_term[0]
            dV = -2. * self.__C * self.__D * R * exp_term
        elif (self.model_type == 'model_C'):
            # Note: np.heaviside() returns 0.5 if x == 0. This is used here to
            #       recover proper limit of V12(0) = B.
            exp_term = np.exp(-self.__C * np.absolute(R))
            V = self.__B * (2 * np.heaviside(x, 0.5) - np.sign(x) * exp_term[0])
            dV = self.__B * self.__C * exp_term
        return V, dV


if __name__ == '__main__':

    # Plotting
    import sys
    import matplotlib.pyplot as plt

    nb = 1
    model_type = sys.argv[1]  # 'model_A'
    pot = TullyModel(nb, **{'model_type': model_type})

    # len(linspace) array of positions
    X = np.linspace(-10., 10., num=1000)

    v1 = []
    v2 = []
    k1 = []
    dv1 = []
    dv2 = []
    dk1 = []

    V1_ad = []
    V2_ad = []
    dV1_ad = []
    dV2_ad = []
    nac1 = []

    if (pot.model_type == 'model_A'):
        scaling_nac = 50.
    elif (pot.model_type == 'model_B'):
        scaling_nac = 12.
    elif (pot.model_type == 'model_C'):
        scaling_nac = 1.

    for i in X:
        pot._calculate_adiabatic_all(np.array([[i]]))

        v1.append(pot._diabatic_energy[0, 0, 0])
        v2.append(pot._diabatic_energy[1, 1, 0])
        k1.append(pot._diabatic_energy[0, 1, 0])
        dv1.append(pot._diabatic_gradient[0, 0, 0, 0])
        dv2.append(pot._diabatic_gradient[1, 1, 0, 0])
        dk1.append(pot._diabatic_gradient[0, 1, 0, 0])

        V1_ad.append(pot._adiabatic_energy[0, 0])
        V2_ad.append(pot._adiabatic_energy[1, 0])
        dV1_ad.append(pot._adiabatic_gradient[0, 0, 0])
        dV2_ad.append(pot._adiabatic_gradient[1, 0, 0])
        nac1.append(pot._nac[0, 1, 0, 0])

    # Plot all
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tully ' + model_type, fontsize=20)

    ax[0, 0].plot(X, v1, 'r-', label="V1")
    ax[0, 0].plot(X, v2, 'k-', label="V2")
    ax[0, 0].plot(X, k1, 'b--', label="K1")
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('Diabatic Potential')
    ax[0, 0].legend(loc='best')
    # ax[0, 0].set_ylim((-0.001, 0.05))

    ax[0, 1].plot(X, dv1, 'r-', label="dV1/dx")
    ax[0, 1].plot(X, dv2, 'k-', label="dV2/dx")
    ax[0, 1].plot(X, dk1, 'b--', label="dK1/dx")
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('Derivative of diabatic potential')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(X, V1_ad, 'r-', label="V1")
    ax[1, 0].plot(X, V2_ad, 'k-', label="V2")
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('Adiabatic Potential')
    ax[1, 0].legend(loc='best')
    # ax[1, 0].set_ylim((-0.001, 0.05))

    ax[1, 1].plot(X, dV1_ad, 'r-', label="dV1/dx")
    ax[1, 1].plot(X, dV2_ad, 'k-', label="dV2/dx")
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('Derivative of Adiabatic Potential')
    ax[1, 1].legend(loc='best')

    ax[1, 2].plot(X, np.array(nac1) / scaling_nac, 'r-', label="NAC / "+str(scaling_nac))
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('NAC')
    ax[1, 2].legend(loc='best')
    # ax[1, 2].set_xlim((2, 6))

    plt.show()
