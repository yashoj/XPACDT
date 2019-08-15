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
    Two state Tully model potentials in one dimension.
    Reference: J. Chem. Phys. 93 (2), 1061 (1990)

    !!! Add form of diagonal and off-diagonal terms!!!

    Other Parameters
    ----------------
    model_type
    """

    def __init__(self, max_n_beads, basis, **kwargs):

        if basis == 'diabatic':
            bases_used = 'diabatic'
        elif basis == 'adiabatic':
            bases_used = 'dia2ad'
        else:
            raise ValueError("Electronic state basis representation not available.")

        itemplate.PotentialInterface.__init__(self, "TullyModel", 1,
                                              max_n_beads, 2, bases_used)

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
               ("Type of Tully model not available.")
        self.__model_type = m

    def _calculate_all(self, R, P=None, S=None):
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
        # TODO: Where to place asserts so that they are only checked once in the beginning.
        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        assert (R.shape[0] == self.n_dof), "Degrees of freedom is not one!"
        assert (R.shape[1] == self.max_n_beads), "Number of beads does not match!"

        self._calculate_diabatic_all(R)

        if (self.bases_used == 'dia2ad'):
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
        self._diabatic_energy[0, 0] = self._get_V11(R)
        self._diabatic_energy[1, 1] = self._get_V22(R)
        self._diabatic_gradient[0, 0] = self._get_grad_dV11(R)
        self._diabatic_gradient[1, 1] = self._get_grad_dV22(R)

        # Taking into account that the potential matrix is real and Hermitian
        self._diabatic_energy[0, 1] = self._get_V12(R)
        self._diabatic_energy[1, 0] = self._diabatic_energy[0, 1].copy()

        self._diabatic_gradient[0, 1] = self._get_grad_dV12(R)
        self._diabatic_gradient[1, 0] = self._diabatic_gradient[0, 1].copy()

        # Centroid part
        if self.max_n_beads == 1:
            self._diabatic_energy_centroid = (
                self._diabatic_energy.reshape((self.n_states, self.n_states))).copy()
            self._diabatic_gradient_centroid = (
                self._diabatic_gradient.reshape((self.n_states, self.n_states, self.n_dof))).copy()
        else:
            r_centroid = np.mean(R, axis=1)
            self._diabatic_energy_centroid[0, 0] = self._get_V11(r_centroid)
            self._diabatic_energy_centroid[1, 1] = self._get_V22(r_centroid)
            self._diabatic_gradient_centroid[0, 0] = self._get_grad_dV11(r_centroid)
            self._diabatic_gradient_centroid[1, 1] = self._get_grad_dV22(r_centroid)

            self._diabatic_energy_centroid[0, 1] = self._get_V12(r_centroid)
            self._diabatic_energy_centroid[1, 0] = \
                self._diabatic_energy_centroid[0, 1]

            self._diabatic_gradient_centroid[0, 1] = self._get_grad_dV12(r_centroid)
            self._diabatic_gradient_centroid[1, 0] = \
                self._diabatic_gradient_centroid[0, 1].copy()

    # TODO: how to get rid of these small functions as without them, energies
    # and gradients have to be set twice for beads and centroid
    # Maybe use lambda functions??

    def _get_V11(self, R):
        """
        Get first diagonal diabatic energy term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        float /or/ (n_beads) ndarrays of floats
            Diagonal diabatic energy term.
        """
        x = R[0]

        if (self.model_type == 'model_A'):
            # Note: np.sign() returns the 0 if x == 0. That is still fine here.
            return (np.sign(x) * self.__A * (1. - np.exp(-self.__B * np.absolute(x))))
        elif (self.model_type == 'model_B'):
            return (0. * x)
        elif (self.model_type == 'model_C'):
            # !!!! How to do this more efficiently? if statement faster???
            return (self.__A * (1. + 0. * x))

    def _get_V22(self, R):
        """
        Get second diagonal diabatic energy term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        float /or/ (n_beads) ndarrays of floats
            Diagonal diabatic energy term.
        """
        x = R[0]

        if (self.model_type == 'model_A' or self.model_type == 'model_C'):
            return (-1. * self._get_V11(R))
        elif (self.model_type == 'model_B'):
            return (-self.__A * np.exp(-self.__B * x * x) + self.__Eo)
        
    def _get_V12(self, R):
        """
        Get off-diagonal diabatic energy term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        float /or/ (n_beads) ndarrays of floats
            Off-diagonal diabatic energy term.
        """
        x = R[0]

        if (self.model_type == 'model_A' or self.model_type == 'model_B'):
            return (self.__C * np.exp(-self.__D * x * x))
        elif (self.model_type == 'model_C'):
            return (self.__B * (2 * np.heaviside(x, 0.5) - np.sign(x)
                                * np.exp(-self.__C * np.absolute(x))))

    def _get_grad_dV11(self, R):
        """
        Get diagonal diabatic gradient term for first state.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Diagonal diabatic gradient term.
        """
        if (self.model_type == 'model_A'):
            return (self.__A * self.__B * np.exp(-self.__B * np.absolute(R)))
        elif (self.model_type == 'model_B' or self.model_type == 'model_C'):
            return (0. * R)

    def _get_grad_dV22(self, R):
        """
        Get diagonal diabatic gradient term for second state.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Diagonal diabatic gradient term.
        """
        if (self.model_type == 'model_A'):
            return (-1. * self._get_grad_dV11(R))
        elif (self.model_type == 'model_B'):
            return (2 * R * self.__A * self.__B * np.exp(-self.__B * R * R))
        elif (self.model_type == 'model_C'):
            return (0. * R)

    def _get_grad_dV12(self, R):
        """
        Get off-diagonal diabatic gradient term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Off-diagonal diabatic gradient term.
        """
        if (self.model_type == 'model_A' or self.model_type == 'model_B'):
            return (-2. * self.__C * self.__D * R * np.exp(-self.__D * R * R))
        elif (self.model_type == 'model_C'):
            return (self.__B * self.__C * np.exp(-self.__C * np.absolute(R)))


if __name__ == '__main__':

    # Plotting
    import sys
    import matplotlib.pyplot as plt

    nb = 1
    model_type = sys.argv[1]  # 'model_A'
    pot = TullyModel(nb, 'adiabatic', **{'model_type': model_type})

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
        pot._calculate_all(np.array([[i]]))

        v1.append(pot._diabatic_energy[0, 0, 0])
        v2.append(pot._diabatic_energy[1, 1, 0])
        k1.append(pot._diabatic_energy[0, 1, 0])
        dv1.append(pot._diabatic_gradient[0, 0, 0, 0])
        dv2.append(pot._diabatic_gradient[1, 1, 0, 0])
        dk1.append(pot._diabatic_gradient[0, 1, 0, 0])

        V1_ad.append(pot._energy[0, 0])
        V2_ad.append(pot._energy[1, 0])
        dV1_ad.append(pot._gradient[0, 0, 0])
        dV2_ad.append(pot._gradient[1, 0, 0])
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
