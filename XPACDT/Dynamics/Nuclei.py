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

import numpy as np
# import scipy as sp

# TODO: Design decision - how to roder R, P, etc. beads x DOF or DOF x beads
# TODO: add more quantities calculated for the nuclei!


class Nuclei(object):
    """
    This class represents the nuclar degrees of freedom.

    Parameters:
    -----------
    degrees_of_freedom : int
        The number of nuclear degrees of freedom present.
    coordinate : two-dimensional ndarray of floats
        The positions of all beads in the system. The first axis is the degrees
        of freedom and the second axis the beads.
    momenta : two-dimensional ndarray of floats
        The momenta of all beads in the system. The first axis is the degrees
        of freedom and the second axis the beads.
    propagator : VelocityVerlet object, optional
        The propagator used to integrate the equations of motion for this
        system. Default: None.

    # TODO: Do we need those two???
    xyz_atoms : bool, optional
                Whether this is a molecule with xyz-coordinates for each atom.
                Default: False
    n_beads : list of int, optional
              The number of beads per degree of freedom. If one element is
              given it is assumed that each degree of freedom has that many
              beads. Default: [1]

    Attributes:
    -----------
    n_dof
    n_beads
    positions
    momenta
    """

    def __init__(self, degrees_of_freedom, coordinates, momenta,
                 propagator=None, xyz_atoms=False, n_beads=[1], **kwargs):
        self.n_dof = degrees_of_freedom

        if xyz_atoms:
            assert ((self.n_dof % 3) == 0), "Assumed atoms, but number of \
 deegrees of freedom not a multiple of 3."
            self.n_atoms = self.n_dof / 3

        self.n_beads = n_beads

        self.positions = coordinates
        self.momenta = momenta

        self.log = []

        self.__propagator = propagator
        return

    @property
    def n_dof(self):
        """int : The number of degrees of freedom in this system."""
        return self.__n_dof

    @n_dof.setter
    def n_dof(self, i):
        assert (i > 0), "Number of degrees of freedom less than 1!"
        self.__n_dof = i

    @property
    def n_beads(self):
        """list of int : The number of beads for each degree of freedom."""
        return self.__n_beads

    @n_beads.setter
    def n_beads(self, l):
        # TODO: add check for atoms to give only one bead number per atom.
        assert (len(l) == 1 or len(l) == self.n_dof), "Wrong number of \
beads given."

        if len(l) == 1:
            self.__n_beads = l * self.n_dof
        else:
            self.__n_beads = l
        # TODO: add check for multiple of twos

    @property
    def positions(self):
        """two-dimensional ndarray of floats : The positions of all beads in
            the system. The first axis is the degrees of freedom and the
            second axis the beads."""
        return self.__positions

    @positions.setter
    def positions(self, a):
        self.__positions = a.copy()

    @property
    def x_centroid(self):
        """ Array of floats : The centroid of each coordinate. """
        return np.mean(self.positions, axis=1)

    @property
    def momenta(self):
        """two-dimensional ndarray of floats : The momenta of all beads in
            the system. The first axis is the degrees of freedom and the
            second axis the beads."""
        return self.__momenta

    @momenta.setter
    def momenta(self, a):
        self.__momenta = a.copy()

    @property
    def p_centroid(self):
        """ Array of floats : The centroid of each momentum. """
        return np.mean(self.momenta, axis=1)

    def propagate(self, time):
        """ This functions advances the positions and momenta of the nuclei
        for a given time using the proapgator assigned.

        Parameters
        ----------
        time : float
            The time to advance the nuclei.
        """

        self.positions, self.momenta = \
            self.__propagator.propagate(self.positions, self.momenta, time)

        self.log.append(self.positions.copy())

        return
