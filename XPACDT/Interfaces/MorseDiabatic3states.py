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

""" This module represents a three state morse diabatic potential in 
one dimension."""

import numpy as np

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Input.Inputfile as infile


class MorseDiabatic3states(itemplate.PotentialInterface):
    """
    Three state morse diabatic potential in one dimension.
    The diagonal terms are morse potential and off-diagonal couplings are gaussian.
    Reference: Chem. Phys. Lett. 349, 521-529 (2001)

    Other Parameters
    ----------------
    x0 : float or string of float
        Equilibrium position of the polynomial.
    a : string containing several floats
        The expansion coefficients for the polynomial in ascending order. The
        expansion length is determined by the number of given coefficients
        here.
    """

    def __init__(self, max_n_beads, basis, **kwargs):

        if basis == 'diabatic':
            bases_used = 'diabatic'
        elif basis == 'adiabatic':
            bases_used = 'dia2ad'
        else:
            raise ValueError("Electronic state basis representation not available.")

        itemplate.PotentialInterface.__init__(self, "MorseDiabatic3states", 1,
                                              3, max_n_beads, bases_used)

        assert (isinstance(kwargs.get('model_type'), str)), \
            "Parameter 'model_type' not given or not given as string."
        self.__model_type = kwargs.get('model_type')

        # Read model parameters from file
        all_params = infile.Inputfile("model_parameters/morse_diabatic_potential.params")
        assert (self.model_type in all_params.keys()), \
            "Type of morse diabatic model not found."
        model_params = all_params.get(self.model_type)

        # TODO: maybe make this into list since all parameters are the same?
        # That way can accomodate 2, 3 or more states properly

        # Setting all the paramters
        # Diagonal terms
        self.___de1 = float(model_params.get('de1'))
        self.___beta1 = float(model_params.get('beta1'))
        self.___re1 = float(model_params.get('re1'))
        self.___c1 = float(model_params.get('c1'))
        self.___de2 = float(model_params.get('de2'))
        self.___beta2 = float(model_params.get('beta2'))
        self.___re2 = float(model_params.get('re2'))
        self.___c2 = float(model_params.get('c2'))
        self.___de3 = float(model_params.get('de3'))
        self.___beta3 = float(model_params.get('beta3'))
        self.___re3 = float(model_params.get('re3'))
        self.___c3 = float(model_params.get('c3'))
        # Off-diagonal terms
        self.___A12 = float(model_params.get('A12'))
        self.___as12 = float(model_params.get('as12'))
        self.___r12 = float(model_params.get('r12'))
        self.___A23 = float(model_params.get('A23'))
        self.___as23 = float(model_params.get('as23'))
        self.___r23 = float(model_params.get('r23'))
        
        # Allocate proper sizes for energy and gradient matrices
        
    
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
        
    def _calculate_diabatic_all(self, R):
        """
        Get back diabatic matrices for energies and gradients for beads and centroid. 
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
        # TODO: Where to place asserts so that they are only checked once in the beginning.
        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        assert (R.shape[0] == self.n_dof), "Degrees of freedom is not one!"
        assert (R.shape[1] == self.max_n_beads), "Number of beads does not match!"

        r_centroid = np.mean(R, axis=1)

        self._diabatic_energy[0, 0] = self._get_diagonal_V(R, self.__de1, self.__beta1, self.__re1, self.__c1)
        self._diabatic_energy[1, 1] = self._get_diagonal_V(R, self.__de1, self.__beta1, self.__re1, self.__c1)
        self._diabatic_energy[2, 2] = self._get_diagonal_V(R, self.__de1, self.__beta1, self.__re1, self.__c1)
        self._diabatic_energy[0, 1] = self._get_off_diagonal_V(R, self.__de1, self.__beta1, self.__re1, self.__c1)
        self._diabatic_energy[1, 2] = self._get_diagonal_V(R, self.__de1, self.__beta1, self.__re1, self.__c1)


        
    # TODO: how to get rid of these small functions as without them, energies
    # and gradients have to be set twice for beads and centroid
    # Maybe use lambda functions??
    def _get_diagonal_V(self, R, de, beta, re, c):
        p = de1 * (1 - np.exp(-beta * (R[0] - re)))**2 + c
        return p
        
    def _get_off_diagonal_V(self, R, de, beta, re, c):
        p = de1 * (1 - np.exp(-beta * (R[0] - re)))**2 + c
        return p
        
    def _V11_diab(self, R):
        p = self.__de1 * (1 - np.exp(-self.__beta1 * (R[0] - self.__re1)))**2 + self.__c1
        return p

    def _V22_diab(self, R):
        p = self.__de2 * (1 - np.exp(-self.__beta2 * (R[0] - self.__re2)))**2 + self.__c2
        return p

    def _V33_diab(self, R):
        p = self.__de3 * (1 - np.exp(-self.__beta3 * (R[0] - self.__re3)))**2 + self.__c3
        return p

    def _V12_diab(self, R):
        p = self.__A12 * np.exp(-self.__as12 * (R[0] - self.__r12)**2)
        return p
    
    def _V23_diab(self, R):
        p = self.__A23 * np.exp(-self.__as23 * (R[0] - self.__r23)**2)
        return p

    # Functions to return derivatives of diabatic potential as an array with 3 elements representing 
    # each axis, however setting 0 for 2nd and 3rd axis (just needed to interface with CDTK)
    # Input : x  - Ndof array or Ndof * Nbeads matrix of position
    # Output: dp - same dimensions as input x with derivative values

    def _dV1_morse_diab(self, x):
        dp = 2* self.beta1 *self.de1 *np.exp(-self.beta1*(x[0] - self.re1)) \
             *(1 -np.exp(-self.beta1*(x[0] -self.re1)) )
        return np.array([dp, 0.*x[1], 0.*x[2]])
        
    def _dV2_morse_diab(self, x):
        dp = 2* self.beta2 *self.de2 *np.exp(-self.beta2*(x[0] - self.re2)) \
             *(1 -np.exp(-self.beta2*(x[0] -self.re2)) )
        return np.array([dp, 0.*x[1], 0.*x[2]])
        
    def _dV12_morse_diab(self, x):
        dp = -2 *self.aw12 *self.a12 *(x[0] -self.r12) * np.exp(-self.aw12*(x[0] -self.r12)**2)  
        return np.array([dp, 0.*x[1], 0.*x[2]])
        return

        