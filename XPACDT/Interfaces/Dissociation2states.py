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

""" This module represents a two state dissociation potential in one dimension."""

import numpy as np
import os

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Input.Inputfile as infile


class Dissociation2states(itemplate.PotentialInterface):
    """
    Two state morse diabatic potential in one dimension.
    The diagonal terms are morse potential and dissociative potential,
    and off-diagonal coupling is gaussian.
    Reference: J. Chem. Phys. 150, 114105 (2019)
    Please note the change in variables compared to the paper: E -> De,
    qo -> re, qo12 -> r12c.

    TODO: Add form of diagonal and off-diagonal terms; and aliases from paper!

    Parameters
    ----------
    basis : {'adiabatic', 'diabatic'}
        Electronic state basis representations to be used. Default: 'adiabatic'.
    max_n_beads : int, optional
        Maximum number of beads from the (n_dof) list of n_beads. Default: 1.

    Other Parameters
    ----------------
    model_type : {'strong_coupling', 'weak_coupling'}
        String denoting model type to be used.
    """

    def __init__(self, basis, max_n_beads=1, **kwargs):

        if basis == 'diabatic':
            bases_used = 'diabatic'
        elif basis == 'adiabatic':
            bases_used = 'dia2ad'

        itemplate.PotentialInterface.__init__(self, "Dissociation2states", 1,
                                              2, max_n_beads, bases_used)

        assert (isinstance(kwargs.get('model_type'), str)), \
            "Parameter 'model_type' not given or not given as string."
        self.__model_type = kwargs.get('model_type')

        # Read model parameters from file
        param_file = os.path.join(os.path.dirname(itemplate.__file__),
                                  "model_parameters/dissociation_potential.param")
        all_params = infile.Inputfile(param_file)
        assert (self.model_type in all_params.keys()), \
            "Type of morse diabatic model not found."
        model_params = all_params.get(self.model_type)

        # Setting all the paramters
        # Diagonal terms as list of floats of individual states
        self.__a = [float(model_params.get('a'+str(i+1)))
                    for i in range(self.n_states)]
        self.__re = [float(model_params.get('re'+str(i+1)))
                     for i in range(self.n_states)]
        self.__De = [float(model_params.get('De'+str(i+1)))
                     for i in range(self.n_states)]
        self.__b = [float(model_params.get('b'+str(i+1)))
                    for i in range(self.n_states)]
        # Off-diagonal terms as floats
        self.__V12c = float(model_params.get('V12c'))
        self.__a12c = float(model_params.get('a12c'))
        self.__r12c = float(model_params.get('r12c'))

    @property
    def model_type(self):
        """string : Model number to be used."""
        return self.__model_type

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
        for i in range(self.n_states):
            self._diabatic_energy[i, i] = self._get_diag_V(R, i)
            self._diabatic_gradient[i, i] = self._get_diag_grad(R, i)

        # Taking into account that the potential matrix is real and Hermitian
        self._diabatic_energy[0, 1] = self._get_off_diag_V(R)
        self._diabatic_energy[1, 0] = self._diabatic_energy[0, 1].copy()

        self._diabatic_gradient[0, 1] = self._get_off_diag_grad(R)
        self._diabatic_gradient[1, 0] = self._diabatic_gradient[0, 1].copy()

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

            self._diabatic_energy_centroid[0, 1] = self._get_off_diag_V(r_centroid)
            self._diabatic_energy_centroid[1, 0] = \
                self._diabatic_energy_centroid[0, 1]

            self._diabatic_gradient_centroid[0, 1] = self._get_off_diag_grad(r_centroid)
            self._diabatic_gradient_centroid[1, 0] = \
                self._diabatic_gradient_centroid[0, 1].copy()

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
        exp_term = np.exp(-self.__a[i] * (R[0] - self.__re[i]))
        if (i == 0):
            return (self.__De[i] * (1. - exp_term)**2 - self.__b[i])
        else:
            return (self.__De[i] * exp_term + self.__b[i])

    def _get_off_diag_V(self, R):
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
        return (self.__V12c * np.exp(-self.__a12c * (R[0] - self.__r12c)**2))

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
        exp_term = np.exp(-self.__a[i] * (R - self.__re[i]))
        if (i == 0):
            return (2. * self.__a[i] * self.__De[i] * exp_term * (1. - exp_term))
        else:
            return (-self.__a[i] * self.__De[i] * exp_term)

    def _get_off_diag_grad(self, R):
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
        return (-2. * self.__a12c * self.__V12c * (R - self.__r12c) * np.exp(-self.__a12c * (R - self.__r12c)**2))


if __name__ == '__main__':

    # Plotting
    import matplotlib.pyplot as plt
    nb = 1
    model_type = 'weak_coupling'  # 'strong_coupling'
    pot = Dissociation2states(nb, 'adiabatic', **{'model_type': model_type})

    # len(linspace) array of positions
    # X = np.linspace(1.5, 5., num=1000)
    X = np.linspace(1., 10., num=1000)

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
    fig.suptitle('Dissociation potential: ' + model_type, fontsize=20)

    ax[0, 0].plot(X, v1, 'r-', label="V1")
    ax[0, 0].plot(X, v2, 'k-', label="V2")
    ax[0, 0].plot(X, np.array(k1) * 10, 'b--', label="K1 * 10")
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('Diabatic Potential')
    ax[0, 0].legend(loc='best')
    # ax[0, 0].set_ylim((-0.001, 0.05))

    ax[0, 1].plot(X, dv1, 'r-', label="dV1/dx")
    ax[0, 1].plot(X, dv2, 'k-', label="dV2/dx")
    ax[0, 1].plot(X, np.array(dk1) * 10, 'b--', label="dK1/dx * 10")
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

    ax[1, 2].plot(X, nac1, 'r-', label="NAC_1")
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('NAC')
    ax[1, 2].legend(loc='best')
    ax[1, 2].set_xlim((2, 6))

    plt.show()
