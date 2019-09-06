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

""" This module represents a two or three state morse diabatic potential in
one dimension."""

import numpy as np
import os

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Input.Inputfile as infile


class MorseDiabatic(itemplate.PotentialInterface):
    """
    Two or three state morse diabatic potential in one dimension.
    The diagonal terms are morse potential and off-diagonal couplings are gaussian.
    Reference: Chem. Phys. Lett. 349, 521-529 (2001)

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

        try:
            n_states = int(kwargs.get('n_states', 3))
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameter 'n_states' for morse "
                                   "diabatic not convertable to int. "
                                   "'n_states' is " + kwargs.get('n_states'))
        assert ((n_states == 2) or (n_states == 3)), \
               ("Only 2 or 3 states possible for morse diabatic potential")

        itemplate.PotentialInterface.__init__(self, "MorseDiabatic", 1,
                                              max_n_beads, n_states, bases_used)

        assert (isinstance(kwargs.get('model_type'), str)), \
            "Parameter 'model_type' not given or not given as string."
        self.__model_type = kwargs.get('model_type')

        # Read model parameters from file
        param_file = os.path.join(os.path.dirname(itemplate.__file__),
                                  "model_parameters/morse_diabatic_potential.param")
        all_params = infile.Inputfile(param_file)
        assert (self.model_type in all_params.keys()), \
            "Type of morse diabatic model not found."
        model_params = all_params.get(self.model_type)

        # Setting all the paramters
        # Diagonal terms as list of floats of individual states
        self.__de = [float(model_params.get('de'+str(i+1)))
                     for i in range(self.n_states)]
        self.__beta = [float(model_params.get('beta'+str(i+1)))
                       for i in range(self.n_states)]
        self.__re = [float(model_params.get('re'+str(i+1)))
                     for i in range(self.n_states)]
        self.__c = [float(model_params.get('c'+str(i+1)))
                    for i in range(self.n_states)]
        # Off-diagonal terms as floats
        self.__A12 = float(model_params.get('A12'))
        self.__as12 = float(model_params.get('as12'))
        self.__r12 = float(model_params.get('r12'))
        if self.n_states == 3:
            if self.model_type == 'model_1':
                self.__A23 = float(model_params.get('A23'))
                self.__as23 = float(model_params.get('as23'))
                self.__r23 = float(model_params.get('r23'))
            elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                self.__A13 = float(model_params.get('A13'))
                self.__as13 = float(model_params.get('as13'))
                self.__r13 = float(model_params.get('r13'))

    @property
    def model_type(self):
        """string : Model number to be used."""
        return self.__model_type

    def _calculate_all(self, R, P=None, S=None):
        """
        Calculate and set diabatic and adiabatic (if required) matrices for
        energies and gradients of beads and centroid.

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

        self._calculate_diabatic_all(R)

        if (self.bases_used == 'dia2ad'):
             self._get_adiabatic_from_diabatic(R, self._get_diabatic_energy_matrix)

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
        for i in range(self.n_states):
            self._diabatic_energy[i, i] = self._get_diag_V(R, i)
            self._diabatic_gradient[i, i] = self._get_diag_grad(R, i)

        # Taking into account that the potential matrix is real and Hermitian
        self._diabatic_energy[0, 1] = self._get_off_diag_V(
            R, self.__A12, self.__as12, self.__r12)
        self._diabatic_energy[1, 0] = self._diabatic_energy[0, 1].copy()

        self._diabatic_gradient[0, 1] = self._get_off_diag_grad(
            R, self.__A12, self.__as12, self.__r12)
        self._diabatic_gradient[1, 0] = self._diabatic_gradient[0, 1].copy()

        if self.n_states == 3:
            if self.model_type == 'model_1':
                self._diabatic_energy[1, 2] = self._get_off_diag_V(
                    R, self.__A23, self.__as23, self.__r23)
                self._diabatic_energy[2, 1] = self._diabatic_energy[1, 2].copy()
    
                self._diabatic_gradient[1, 2] = self._get_off_diag_grad(
                    R, self.__A23, self.__as23, self.__r23)
                self._diabatic_gradient[2, 1] = self._diabatic_gradient[1, 2].copy() 

            elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                self._diabatic_energy[0, 2] = self._get_off_diag_V(
                    R, self.__A13, self.__as13, self.__r13)
                self._diabatic_energy[2, 0] = self._diabatic_energy[0, 2].copy()

                self._diabatic_gradient[0, 2] = self._get_off_diag_grad(
                    R, self.__A13, self.__as13, self.__r13)
                self._diabatic_gradient[2, 0] = self._diabatic_gradient[0, 2].copy()

        # Centroid part
        if self.max_n_beads == 1:
            self._diabatic_energy_centroid = (
                self._diabatic_energy.reshape((self.n_states, self.n_states))).copy()
            self._diabatic_gradient_centroid = (
                self._diabatic_gradient.reshape((self.n_states, self.n_states, self.n_dof))).copy()
        else:
            r_centroid = np.mean(R, axis=1)
            for i in range(self.n_states):
                self._diabatic_energy_centroid[i, i] = self._get_diag_V(r_centroid, i)
                self._diabatic_gradient_centroid[i, i] = self._get_diag_grad(r_centroid, i)

            self._diabatic_energy_centroid[0, 1] = self._get_off_diag_V(
                r_centroid, self.__A12, self.__as12, self.__r12)
            self._diabatic_energy_centroid[1, 0] = \
                self._diabatic_energy_centroid[0, 1]

            self._diabatic_gradient_centroid[0, 1] = self._get_off_diag_grad(
                r_centroid, self.__A12, self.__as12, self.__r12)
            self._diabatic_gradient_centroid[1, 0] = \
                self._diabatic_gradient_centroid[0, 1].copy()

            if self.n_states == 3:
                if self.model_type == 'model_1':
                    self._diabatic_energy_centroid[1, 2] = self._get_off_diag_V(
                        r_centroid, self.__A23, self.__as23, self.__r23)
                    self._diabatic_energy_centroid[2, 1] = self._diabatic_energy_centroid[1, 2]

                    self._diabatic_gradient_centroid[1, 2] = self._get_off_diag_grad(
                        r_centroid, self.__A23, self.__as23, self.__r23)
                    self._diabatic_gradient_centroid[2, 1] = self._diabatic_gradient_centroid[1, 2].copy()

                elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                    self._diabatic_energy_centroid[0, 2] = self._get_off_diag_V(
                        r_centroid, self.__A13, self.__as13, self.__r13)
                    self._diabatic_energy_centroid[2, 0] = self._diabatic_energy_centroid[0, 2]

                    self._diabatic_gradient_centroid[0, 2] = self._get_off_diag_grad(
                        r_centroid, self.__A13, self.__as13, self.__r13)
                    self._diabatic_gradient_centroid[2, 0] = self._diabatic_gradient_centroid[0, 2].copy()

    # TODO: how to get rid of these small functions as without them, energies
    # and gradients have to be set twice for beads and centroid
    # Maybe use lambda functions??

    def _get_diag_V(self, R, i):
        """
        Get diagonal diabatic energy term of 'i'-th state.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.
        i: int
            State index.

        Returns:
        ----------
        float /or/ (n_beads) ndarrays of floats
            Diagonal diabatic energy term.
        """

        return (self.__de[i] * (1. - np.exp(-self.__beta[i] * (R[0] - self.__re[i])))**2 + self.__c[i])

    def _get_off_diag_V(self, R, A_ij, as_ij, r_ij):
        """
        Get off-diagonal diabatic energy term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.
        A_ij : float
        as_ij : float
        r_ij : float

        Returns:
        ----------
        float /or/ (n_beads) ndarrays of floats
            Off-diagonal diabatic energy term.
        """

        return (A_ij * np.exp(-as_ij * (R[0] - r_ij)**2))

    def _get_diag_grad(self, R, i):
        """
        Get diagonal diabatic gradient term of 'i'-th state.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.
        i: int
            State index.

        Returns:
        ----------
        (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Diagonal diabatic gradient term.
        """

        return (2. * self.__beta[i] * self.__de[i] * np.exp(-self.__beta[i] * (R - self.__re[i]))
                * (1. - np.exp(-self.__beta[i] * (R - self.__re[i]))))

    def _get_off_diag_grad(self, R, A_ij, as_ij, r_ij):
        """
        Get off-diagonal diabatic gradient term.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.
        A_ij : float
        as_ij : float
        r_ij : float

        Returns:
        ----------
        (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            Off-diagonal diabatic gradient term.
        """

        return (-2. * as_ij * A_ij * (R - r_ij) * np.exp(- as_ij * (R - r_ij)**2))

    def _get_diabatic_energy_matrix(self, R):
        """
        Obtain diabatic energy matrix for beads or centroid.
        This function is needed to pass onto adiabatic transformation and
        should not be used independently.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        V_diabatic : (n_states, n_states) ndarrays of floats
                     /or/ (n_states, n_states, n_beads) ndarrays of floats
            Diabatic energy matrix.
        """
        if len(R.shape) == 1:
            V_diabatic = np.zeros_like(self._diabatic_energy_centroid)
        else:
            V_diabatic = np.zeros_like(self._diabatic_energy)

        for i in range(self.n_states):
            V_diabatic[i, i] = self._get_diag_V(R, i)

        V_diabatic[0, 1] = self._get_off_diag_V(
            R, self.__A12, self.__as12, self.__r12)
        V_diabatic[1, 0] = V_diabatic[0, 1].copy()

        if self.n_states == 3:
            if self.model_type == 'model_1':
                V_diabatic[1, 2] = self._get_off_diag_V(
                    R, self.__A23, self.__as23, self.__r23)
                V_diabatic[2, 1] = V_diabatic[1, 2].copy()

            elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                V_diabatic[0, 2] = self._get_off_diag_V(
                    R, self.__A13, self.__as13, self.__r13)
                V_diabatic[2, 0] = V_diabatic[0, 2].copy()

        return V_diabatic


if __name__ == '__main__':

    # !!! Should these plotting scripts be left here for future plotting?

    import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad
    pot = MorseDiabatic(4, 'adiabatic', **{'n_states': '3', 'model_type': 'model_3'})

    # R = np.array([[3.3, 3.4,  3.5, 3.6]])
    R = np.array([[2., 3.5, 4., 5.]])

    pot._calculate_all(R)

    # print(pot._energy)
    # print(pot._gradient, '\n\n')
    print(pot._nac, '\n\n')

    # print(dia2ad.get_adiabatic_energy(pot._diabatic_energy))
    # print(dia2ad.get_adiabatic_gradient(R, pot._get_diabatic_energy_matrix, pot.DERIVATIVE_STEPSIZE))
    # print(dia2ad.get_NAC(pot._diabatic_energy, pot._diabatic_gradient), '\n\n')

    print(np.allclose(dia2ad.get_adiabatic_energy(pot._diabatic_energy), pot._energy))
    print(np.allclose(dia2ad.get_adiabatic_gradient(R, pot._get_diabatic_energy_matrix, pot.DERIVATIVE_STEPSIZE), pot._gradient))
    print(np.allclose(dia2ad.get_NAC(pot._diabatic_energy, pot._diabatic_gradient), pot._nac, atol=1e-5))


    # Plotting
    import sys
    import matplotlib.pyplot as plt

    model_type = sys.argv[1]  # 'model_3'
    n_states = int(sys.argv[2])  # 2
    nb = 1
    pot = MorseDiabatic(nb, 'adiabatic', **{'n_states': str(n_states),
                                            'model_type': model_type})

    bead_ind = 0  # Bead to be used for plotting

    # len(linspace) array of positions
    X = np.linspace(1.8, 12., num=1000)
    v1 = []
    v2 = []
    v3 = []
    k1 = []
    k2 = []

    dv1 = []
    dv2 = []
    dv3 = []
    dk1 = []
    dk2 = []

    V1_ad = []
    V2_ad = []
    V3_ad = []
    dV1_ad = []
    dV2_ad = []
    dV3_ad = []

    nac12 = []
    nac13 = []
    nac23 = []

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
        nac12.append(pot._nac[0, 1, 0, 0])

        if (n_states == 3):
            v3.append(pot._diabatic_energy[2, 2, 0])
            dv3.append(pot._diabatic_gradient[2, 2, 0, 0])
            V3_ad.append(pot._energy[2, 0])
            dV3_ad.append(pot._gradient[2, 0, 0])
            nac13.append(pot._nac[0, 2, 0, 0])
            nac23.append(pot._nac[1, 2, 0, 0])

            if model_type == 'model_1':
                k2.append(pot._diabatic_energy[1, 2, 0])
                dk2.append(pot._diabatic_gradient[1, 2, 0, 0])
            else:
                k2.append(pot._diabatic_energy[0, 2, 0])
                dk2.append(pot._diabatic_gradient[0, 2, 0, 0])

    # Plot all
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Morse diabatic potential: ' + model_type, fontsize=20)

    ax[0, 0].plot(X, v1, 'r-', label="V1")
    ax[0, 0].plot(X, v2, 'k-', label="V2")
    ax[0, 0].plot(X, k1, 'b--', label="K1")
    if (n_states == 3):
        ax[0, 0].plot(X, v3, 'g-', label="V3")
        ax[0, 0].plot(X, k2, 'g--', label="K2")
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('Diabatic Potential')
    ax[0, 0].legend(loc='best')
    ax[0, 0].set_ylim((-0.001, 0.05))

    ax[0, 1].plot(X, dv1, 'r-', label="dV1/dx")
    ax[0, 1].plot(X, dv2, 'k-', label="dV2/dx")
    ax[0, 1].plot(X, dk1, 'b--', label="dK1/dx")
    if (n_states == 3):
        ax[0, 1].plot(X, dv3, 'g-', label="dV3/dx")
        ax[0, 1].plot(X, dk2, 'g--', label="dK2/dx")
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('Derivative of diabatic potential')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(X, V1_ad, 'r-', label="V1")
    ax[1, 0].plot(X, V2_ad, 'k-', label="V2")
    if (n_states == 3):
        ax[1, 0].plot(X, V3_ad, 'g-', label="V3")
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('Adiabatic Potential')
    ax[1, 0].legend(loc='best')
    ax[1, 0].set_ylim((-0.001, 0.05))

    ax[1, 1].plot(X, dV1_ad, 'r-', label="dV1/dx")
    ax[1, 1].plot(X, dV2_ad, 'k-', label="dV2/dx")
    if (n_states == 3):
        ax[1, 1].plot(X, dV3_ad, 'g-', label="dV3/dx")
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('Derivative of Adiabatic Potential')
    ax[1, 1].legend(loc='best')

    ax[1, 2].plot(X, nac12, 'r-', label="NAC_12")
    if (n_states == 3):
        ax[1, 2].plot(X, nac13, 'b-', label="NAC_13")
        ax[1, 2].plot(X, nac23, 'g-', label="NAC_23")
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('NAC')
    ax[1, 2].legend(loc='best')
    ax[1, 2].set_xlim((2, 6))

    plt.show()
