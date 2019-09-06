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

"""This is an implementation of fewest switches surface hopping (FSSH) for
electronic dynamics, extended to ring polymer molecular dynamics to give
ring polymer surface hopping (RPSH)."""

import numpy as np

import XPACDT.System.Electrons as electrons


class SurfaceHoppingElectrons(electrons.Electrons):
    """
    Tully's fewest switches surface hopping (FSSH) from his original paper [1]_
    is implemented for electronic dynamics with modifications to incorporate
    nuclear quantum effects using ring polymer molecular dynamics. This
    extension, termed ring polymer surface hopping (RPSH), follows from 
    another paper from Tully [2]_ and some own modifications to it.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    
    References
    ----------
    .. [1] J. Chem. Phys. 93, 1061 (1990)
    .. [2] J. Chem. Phys. 137, 22S549 (2012)
    
    """

    def __init__(self, parameters):

        electronic_parameters = parameters.get("SurfaceHoppingElectrons")
        basis = electronic_parameters.get("basis", "adiabatic")

        electrons.Electrons.__init__(self, "SurfaceHoppingElectrons", parameters, basis)
        print("Initiating surface hopping in ", self.basis, " basis")

        # TODO: decide what should be the default, for nb=1 does that shouldn't matter though
        self.rpsh_type = electronic_parameters.get("rpsh_type", "bead")
        self.rpsh_rescaling = electronic_parameters.get("rpsh_rescaling", "bead")
        self.rescaling_type = electronic_parameters.get("rescaling_type", "nac")
        
        self.__masses_nuclei = parameters.masses
        positions = parameters.coordinates
        momenta = parameters.momenta

        n_states = self.pes.n_states
        max_n_beads = self.pes.max_n_beads
        try:
            initial_state = int(parameters.get("SurfaceHoppingElectrons").get("initial_state"))
            assert ((initial_state >= 0) and (initial_state < n_states)), \
                ("Initial state is either less 0 or exceeds total number of "
                 "states")
        except TypeError:
            print("Initial state for surface hopping not given or not "
                  "convertible to int. Setting to default value: 0")
            initial_state = 0

        self.current_state = initial_state

        if (self.rpsh_type == 'density_matrix'):
            self._c_coeff = np.zeros((max_n_beads, n_states), dtype=complex)
            self._D = np.zeros((max_n_beads, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((max_n_beads, n_states, n_states), dtype=complex)
            # Check if picture is interaction, also in propagation and density matrix creation in hopping
            # This is only needed in interaction picture
            self._phase = np.zeros(max_n_beads, n_states, n_states, dtype=complex)
        else:
            self._c_coeff = np.zeros((1, n_states), dtype=complex)
            self._D = np.zeros((1, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((1, n_states, n_states), dtype=complex)
            self._phase = np.zeros((1, n_states, n_states), dtype=complex)

        self.c_coeff[:, self.current_state] = 1.0 + 0.0j
        

    @property
    def current_state(self):
        """Int : Current electronic state of the system. All beads are in same
        state for now."""
        # TODO: change to list for each bead if rpsh_type = 'individual' added.
        return self.__current_state

    # Is this setter needed?
    @current_state.setter
    def current_state(self, s):
        self.__current_state = s

    @property
    def masses_nuclei(self):
        """(n_dof) ndarray of floats : The masses of each nuclear degree of
           freedom in au."""
        return self.__masses_nuclei

    @property
    def rpsh_type(self):
        """{'bead', 'centroid', 'density_matrix'} : Type of ring polymer
        surface hopping (RPSH) to be used."""
        # TODO: Possibly add 'individual'; does having individual bead hops make sense?
        return self.__rpsh_type

    @rpsh_type.setter
    def rpsh_type(self, r):
        assert (r in ['bead', 'centroid', 'density_matrix']),\
               ("Ring polymer surface hopping (RPSH) type not available.")
        self.__rpsh_type = r

    @property
    def rpsh_rescaling(self):
        """{'bead', 'centroid'} : Type of RPSH rescaling to be used;
        this can be either to conserve bead or centroid energy."""
        return self.__rpsh_rescaling

    @rpsh_rescaling.setter
    def rpsh_rescaling(self, r):
        assert (r in ['bead', 'centroid']),\
               ("RPSH rescaling type not available.")
        self.__rpsh_rescaling = r

    @property
    def rescaling_type(self):
        """{'nac', 'gradient'} : Type of velocity rescaling to be used."""
        # TODO: possibly add 'nac_with_momenta_reversal' if needed
        return self.__rescaling_type

    @rescaling_type.setter
    def rescaling_type(self, r):
        assert (r in ['nac', 'gradient']),\
               ("Velocity rescaling type not available.")
        self.__rescaling_type = r

    def energy(self, R, centroid=False):
        """Return the electronic energy at the current geometry and active
        state as defined by the systems PES. This is a diagonal term in the
        energy matrix.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_beads) ndarray of float /or/ float
        The energy of the systems PES at each bead position or at the centroid
        in hartree.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.energy(R, self.current_state, centroid=centroid)
        else:
            return self.pes.diabatic_energy(R, self.current_state, self.current_state, centroid=centroid)

    def gradient(self, R, centroid=False):
        """Calculate the gradient of the electronic energy at the current
        geometry and active state as defined by the systems PES. This is a
        diagonal term in the gradient matrix.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
        The gradient of the systems PES at each bead position or at the
        centroid in hartree/au.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.gradient(R, self.current_state, centroid=centroid)
        else:
            return self.pes.diabatic_gradient(R, self.current_state, self.current_state, centroid=centroid)

    def step(self, time, **kwargs):
        """ Propagate electronic wavefunction coefficients 
        """
        # Need phase matrix, susbtraction of large diagonal term and getting the density matrix back
        # Solve coefficient eq using rk4 or scipy solve_ivp or scipy ode?
        # Check for interaction rep.
        # If evolution_pic == 'interaction'
        # Get phase matrix and propagate that
        # Propagation equation is fnc: -1j*np.matmul(H, np.matmul(phase, c))
        # Use linear interpolation to get H matrix at particular times
        
        return

    def _get_kinetic_coupling_matrix(self, P):
        # v . d_kj
        np.matmul()
        return

    def _get_H_matrix(self):
        # Give V - i D matrix
        return
    
    def _get_phase_matrix(self):
        # Tile adiabatic energies and
        return
