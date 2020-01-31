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

"""This is a basic implementation of adiabatic electrons, i.e., no electron
dynamics is taken into account."""

import XPACDT.System.Electrons as electrons


class AdiabaticElectrons(electrons.Electrons):
    """ Adiabatic electrons, i.e., no electron dynamics and only the PES
    gradients and energies are passed on.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    masses_nuclei : (n_dof) ndarray of floats
        The masses of each nuclear degree of freedom in au. This is not needed
        for this particular electron subclass.
    R, P : (n_dof, n_beads) ndarray of floats
        The (ring-polymer) positions `R` and momenta `P` representing the
        system nuclei in au. These are not needed for this particular electron
        subclass.
    """

    def __init__(self, parameters, masses_nuclei=None, R=None, P=None):
        electrons.Electrons.__init__(self, "AdiabaticElectrons", parameters,
                                     "adiabatic")

    def step(self, R, P, time_propagate, **kwargs):
        """ Dummy implementation of the step, as adiabatic electrons have no
        explicit time-dependence.
        """

        return

    @property
    def current_state(self):
        """ Int. Here we are always in the lowest state."""
        return 0

    def energy(self, R, centroid=False):
        """Calculate the electronic energy at the current geometry as defined
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
        in au.
        """
        return self.pes.adiabatic_energy(R, centroid=centroid)

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

        return self.pes.adiabatic_gradient(R, centroid=centroid)

    def get_population(self, proj, basis_requested):
        """ Here we are always in the lowest state. Thus we will return 1 for
        `proj` = 0 and 0 else.

        Parameters
        ----------
        proj : int
            State to be projected onto in the basis given by `basis_requested`.
        basis_requested : str
            Electronic basis to be used. Can be "adiabatic" or "diabatic".

        Returns
        -------
        float
            Electronic population value.
        """

        if proj == 0:
            return 1.0
        else:
            return 0.0
