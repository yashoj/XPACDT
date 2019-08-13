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

import math
import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Input.Inputfile as infile


class MorseDiabatic(itemplate.PotentialInterface):
    """
    Three state morse diabatic potential in one dimension.
    The diagonal terms are morse potential and off-diagonal couplings are gaussian.
    Reference: Chem. Phys. Lett. 349, 521-529 (2001)

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
            

        itemplate.PotentialInterface.__init__(self, "MorseDiabatic3states", 1,
                                              n_states, max_n_beads, bases_used)

        assert (isinstance(kwargs.get('model_type'), str)), \
            "Parameter 'model_type' not given or not given as string."
        self.__model_type = kwargs.get('model_type')

        # Read model parameters from file
        all_params = infile.Inputfile("model_parameters/morse_diabatic_potential.params")
        assert (self.model_type in all_params.keys()), \
            "Type of morse diabatic model not found."
        model_params = all_params.get(self.model_type)


        # Setting all the paramters
        # Diagonal terms as list of floats of individual states
        self.__de = [float(model_params.get('de'+str(i+1))) for i in range(self.n_states)]
        self.___beta = [float(model_params.get('beta'+str(i+1))) for i in range(self.n_states)]
        self.___re = [float(model_params.get('re'+str(i+1))) for i in range(self.n_states)]
        self.___c = [float(model_params.get('c'+str(i+1))) for i in range(self.n_states)]
        # Off-diagonal terms as floats
        self.___A12 = float(model_params.get('A12'))
        self.___as12 = float(model_params.get('as12'))
        self.___r12 = float(model_params.get('r12'))
        if self.n_states == 3:
            if self.model_type == 'model_1':
                self.___A23 = float(model_params.get('A23'))
                self.___as23 = float(model_params.get('as23'))
                self.___r23 = float(model_params.get('r23'))
            elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                self.___A13 = float(model_params.get('A13'))
                self.___as13 = float(model_params.get('as13'))
                self.___r13 = float(model_params.get('r13'))
        
    
    @property
    def model_type(self):
        """string : Model number to be used."""
        return self.__model_type

    def _calculate_all(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

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
            # Adiabatize here
            if self.n_states == 2:
                import DiabaticToAdiabatic_2states as dia2ad
                self._gradient = dia2ad.get_adiabatic_gradient(
                        self._diabatic_energy, self._diabatic_gradient)
                self._gradient_centroid = dia2ad.get_adiabatic_gradient(
                        self._diabatic_energy_centroid, self._diabatic_gradient_centroid)

            elif self.n_states == 3:
                import DiabaticToAdiabatic_Nstates as dia2ad
                self._gradient = dia2ad.get_adiabatic_gradient(
                    R, self.__get_diabatic_energy_3states,
                    self.DERIVATIVE_STEPSIZE)

                r_centroid = np.mean(R, axis=1)
                self._gradient_centroid = dia2ad.get_adiabatic_gradient(
                    r_centroid, self.__get_diabatic_energy_3states,
                    self.DERIVATIVE_STEPSIZE)

            self._energy = dia2ad.get_adiabatic_energy(self._diabatic_energy)
            self._energy_centroid = dia2ad.get_adiabatic_energy(self._diabatic_energy_centroid)

            self._nac = dia2ad.get_NAC(self._diabatic_energy, self._diabatic_gradient)
            self._nac_centroid = dia2ad.get_NAC(self._diabatic_energy_centroid, self._diabatic_gradient_centroid)


    def _calculate_diabatic_all(self, R):
        """
        Calculate and set diabatic matrices for energies and gradients for
        beads and centroid. 

        Bead energies are (n_states, n_states, n_beads) ndarrays of floats,
        bead gradients are (n_states, n_states, n_dof, n_beads) ndarrays of floats,

        centroid energies are (n_states, n_states) ndarrays of floats and
        centroid gradients are (n_states, n_states, n_dof) ndarrays of floats
        
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
        self._diabatic_energy[0, 1] = self._get_off_diag_V(R, self.__A12, self.__as12, self.__r12)
        self._diabatic_energy[1, 0] = self._diabatic_energy[0, 1].copy()

        self._diabatic_gradient[0, 1] = self._get_off_diag_grad(R, self.__A12, self.__as12, self.__r12)
        self._diabatic_gradient[1, 0] = self._diabatic_gradient[0, 1].copy()

        if self.n_states == 3:
            if self.model_type == 'model_1':
                self._diabatic_energy[1, 2] = self._get_off_diag_V(R, self.__A23, self.__as23, self.__r23)
                self._diabatic_energy[2, 1] = self._diabatic_energy[1, 2].copy()
    
                self._diabatic_gradient[1, 2] = self._get_off_diag_grad(R, self.__A23, self.__as23, self.__r23)
                self._diabatic_gradient[2, 1] = self._diabatic_gradient[1, 2].copy() 
    
            elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                self._diabatic_energy[0, 2] = self._get_off_diag_V(R, self.__A13, self.__as13, self.__r13)
                self._diabatic_energy[2, 0] = self._diabatic_energy[0, 2].copy()
    
                self._diabatic_gradient[0, 2] = self._get_off_diag_grad(R, self.__A13, self.__as13, self.__r13)
                self._diabatic_gradient[2, 0] = self._diabatic_gradient[0, 2].copy()

        # Centroid part
        if self.max_n_beads == 1:
            self._diabatic_energy_centroid = (self._diabatic_energy.reshape((self.n_states, self.n_states))).copy()
            self._diabatic_gradient_centroid = (self._diabatic_gradient.reshape((self.n_states, self.n_states, self.n_dof))).copy()
        else:
            r_centroid = np.mean(R, axis=1)
            for i in range(self.n_states):
                self._diabatic_energy_centroid[i, i] = self._get_diag_V(r_centroid, i)
                self._diabatic_energy_centroid[i, i] = self._get_diag_grad(r_centroid, i)

            self._diabatic_energy_centroid[0, 1] = self._get_off_diag_V(r_centroid, self.__A12, self.__as12, self.__r12)
            self._diabatic_energy_centroid[1, 0] = self._diabatic_energy_centroid[0, 1]
    
            self._diabatic_gradient_centroid[0, 1] = self._get_off_diag_grad(Rr_centroid self.__A12, self.__as12, self.__r12)
            self._diabatic_gradient_centroid[1, 0] = self._diabatic_gradient_centroid[0, 1].copy()

            if self.n_states == 3:
                if self.model_type == 'model_1':
                    self._diabatic_energy_centroid[1, 2] = self._get_off_diag_V(Rr_centroid self.__A23, self.__as23, self.__r23)
                    self._diabatic_energy_centroid[2, 1] = self._diabatic_energy_centroid[1, 2]
    
                    self._diabatic_gradient_centroid[1, 2] = self._get_off_diag_grad(r_centroid, self.__A23, self.__as23, self.__r23)
                    self._diabatic_gradient_centroid[2, 1] = self._diabatic_gradient_centroid[1, 2].copy() 
        
                elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
                    self._diabatic_energy_centroid[0, 2] = self._get_off_diag_V(r_centroid, self.__A13, self.__as13, self.__r13)
                    self._diabatic_energy_centroid[2, 0] = self._diabatic_energy_centroid[0, 2]
        
                    self._diabatic_gradient_centroid[0, 2] = self._get_off_diag_grad(r_centroid, self.__A13, self.__as13, self.__r13)
                    self._diabatic_gradient_centroid[2, 0] = self._diabatic_gradient_centroid[0, 2].copy()
            
    # TODO: how to get rid of these small functions as without them, energies
    # and gradients have to be set twice for beads and centroid
    # Maybe use lambda functions??

    # These functions take (Ndof) ndarray or (Ndof, Nbeads) ndarray of position

    def _get_diag_V(self, R, i):
        return (self.__de[i] * (1. - math.exp(-self.__beta[i] * (R[0] - self.__re[i])))**2 + self.__c[i])

    def _get_off_diag_V(self, R, A_ij, as_ij, r_ij):
        return (A_ij * math.exp(-as_ij * (R[0] - r_ij)**2))

    def _get_diag_grad(self, R, i):
        return (2. * self.__beta[i] * self.__de[i] * math.exp(-self.__beta[i] * (R - self.__re[i]))
                * (1. - math.exp(-self.__beta[i] * (R - self.__re[i]))))

    def _get_off_diag_grad(self, R, A_ij, as_ij, r_ij):
        return (-2. * as_ij * A_ij * (R - r_ij) * math.exp(- as_ij * (R - r_ij)**2))

    def __get_diabatic_energy_3states(self, R):
        """
        Obtain diabatic energy matrices of 3 state model for beads or centroid. 
        This function is needed to pass onto adiabatic transformation and
        should not be used independently.
        
        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Function to get diabatic energies of shape 
        or  ndarrays of floats to be compatible
        with 'get_adiabatic_energy'. Should take 'R' as parameter.
        
        Returns:
        ----------
        V_diabatic : (n_states, n_states) ndarrays of floats 
                     /or/ (n_states, n_states, n_beads) ndarrays of floats
            Diabatic energy matrix.
        """
        if len(R) == 1:
            V_diabatic = np.zeros_like(self._diabatic_energy_centroid)
        else:
            V_diabatic = np.zeros_like(self._diabatic_energy)

        for i in range(self.n_states):
            V_diabatic[i, i] = self._get_diag_V(R, i)
            
        V_diabatic[0, 1] = self._get_off_diag_V(R, self.__A12, self.__as12, self.__r12)
        V_diabatic[1, 0] = V_diabatic[0, 1].copy()
        
        if self.model_type == 'model_1':
            V_diabatic[1, 2] = self._get_off_diag_V(R, self.__A23, self.__as23, self.__r23)
            V_diabatic[2, 1] = V_diabatic[1, 2].copy()

        elif (self.model_type == 'model_2') or (self.model_type == 'model_3'):
            V_diabatic[0, 2] = self._get_off_diag_V(R, self.__A13, self.__as13, self.__r13)
            V_diabatic[2, 0] = V_diabatic[0, 2].copy()

        return V_diabatic
