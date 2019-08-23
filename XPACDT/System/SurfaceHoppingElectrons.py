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

"""This is a basic implementation of adiabatic electrons, i.e., no electron
dynamics is taken into account."""

import numpy as np

import XPACDT.System.Electrons as electrons


class SurfaceHoppingElectrons(electrons.Electrons):
    """ Surface hopping electrons, i.e., no electron dynamics and only the PES
    gradients and energies are passed on.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    """

    def __init__(self, parameters, n_beads):

        print('Initiating surface hopping.')
        basis = parameters.get("SurfaceHoppingElectrons").get("basis", "Adiabatic")
        self.rpsh_type = parameters.get("SurfaceHoppingElectrons").get("rpsh_type")
        self.rpsh_rescaling = parameters.get("SurfaceHoppingElectrons").get("rpsh_rescaling")
        self.rescaling_type = parameters.get("SurfaceHoppingElectrons").get("rescaling_type")

        electrons.Electrons.__init__(self, parameters, n_beads, basis)

        n_states = self.pes.n_states
        self.current_state = parameters.get("SurfaceHoppingElectrons").get("initial_state")
        # Check initial state is within n_states

        if (self.rpsh_type == 'density_matrix'):
            self._c_coeff = np.zeros((self.pes.max_n_beads, n_states), dtype=complex)
            self._H_e = np.zeros((self.pes.max_n_beads, n_states, n_states), dtype=complex)
        else:
            self._c_coeff = np.zeros((1, n_states), dtype=complex)
            self._H_e = np.zeros((1, n_states, n_states), dtype=complex)

        self.c_coeff[:, self.current_state] = 1.0 + 0.0j

    @property
    def current_state(self):
        """Int : Current electronic state of the system. All beads are in same
        state for now."""
        # TODO: extend to list for each bead if rpsh_type = 'individual' added.
        return self.__current_state

    @current_state.setter
    def current_state(self, s):
        self.__current_state = s

    @property
    def rpsh_type(self):
        """{'beads', 'centroid', 'density_matrix'} : Type of ring polymer
        surface hopping (RPSH) to be used."""
        # TODO: Possibly add 'individual'; does having individual bead hops make sense?
        return self.__rpsh_type

    @rpsh_type.setter
    def rpsh_type(self, r):
        assert (r in ['beads', 'centroid', 'density_matrix']),\
               ("Ring polymer surface hopping (RPSH) type not available.")
        self.__rpsh_type = r

    @property
    def rpsh_rescaling(self):
        """{'beads', 'centroid'} : Type of RPSH rescaling to be used;
        this can be either to conserve bead or centroid energy."""
        return self.__rpsh_rescaling

    @rpsh_rescaling.setter
    def rpsh_rescaling(self, r):
        assert (r in ['beads', 'centroid']),\
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
        """Return the electronic energy at the current geometry as defined
        by the systems PES.

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
        return self.pes.energy(R, self.current_state, centroid=centroid)

    def gradient(self, R, centroid=False):
        """Calculate the gradient of the electronic energy at the current
        geometry as defined by the systems PES.

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

        return self.pes.gradient(R, self.current_state, centroid=centroid)

    def step(self, time, **kwargs):
        """ Dummy implementation of the step, as adiabatic electrons have no
        explicit time-dependence.
        """

        return
