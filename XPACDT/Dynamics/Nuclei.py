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
import sys
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
    parameters : XPACDT.Input.Inputfile
    pes : XPACDT.Interfaces.InterfaceTemplate


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

    def __init__(self, degrees_of_freedom, parameters, pes):

        self.n_dof = degrees_of_freedom
        self.pes = pes

        # coordinates, masses from input - reshape and test some consistency
        # TODO: This should be put into Inputfile.py!
        self.masses = parameters.masses
        if parameters._c_type == 'mass-value':
            self.positions = parameters.coordinates.reshape((self.n_dof, -1))
        elif parameters._c_type == 'xyz':
            self.positions = parameters.coordinates.T.reshape((self.n_dof, -1))

            assert ((self.n_dof % 3) == 0), "Assumed atoms, but number of \
 deegrees of freedom not a multiple of 3."
            self.n_atoms = self.n_dof / 3

        self.n_beads = [self.positions.shape[1]]

        try:
            self.momenta = parameters._momenta.reshape(self.positions.shape)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Number of given momenta and "
                          "coordinates does not match!")

        # set up propagator and attach
        if 'propagator' in parameters:
            self.attach_propagator(parameters)

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
    def propagator(self):
        """ The propagator used to advance the nuclei in time. """
        return self.__propagator

    @propagator.setter
    def propagator(self, p):
        self.__propagator = p

    @property
    def x_centroid(self):
        """ Array of floats : The centroid of each coordinate. """
        return np.mean(self.positions, axis=1)

    @property
    def masses(self):
        """one-dimensional ndarray of floats : The masses of each degree of
        freedom in au."""
        return self.__masses

    @masses.setter
    def masses(self, a):
        self.__masses = a.copy()

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

    @property
    def energy(self):
        """ float : Total energy of the nuclei including the spring term.
        i.e. \frac{1}{n}(\sum_i \sum_j (p^2_ij)(2m_j)) + SPINGS + V). TODO:
        write out properly."""
        return self.kinetic_energy + self.spring_energy + self.potential_energy

    @property
    def kinetic_energy(self):
        """ float TODO, incorrect currently! Need to be changed when
        refactoring."""
        return 0.5*np.sum(self.momenta * self.momenta)

    @property
    def spring_energy(self):
        """ floatTODO, incorrect currently! Need to be changed when
        refactoring."""
        return 0.0

    @property
    def potential_energy(self):
        """ floatTODO, incorrect currently! Need to be changed when
        refactoring."""
        return self.pes.energy(self.positions)

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

        return

    def attach_propagator(self, parameters):
        prop_parameters = parameters.get('propagator')
        if 'rpmd' in parameters:
            assert('beta' in parameters.get("rpmd")), "No beta " \
                    "given for RPMD."
            prop_parameters['beta'] = parameters.get("rpmd").get('beta')

        method = prop_parameters.get('method')
        __import__("XPACDT.Dynamics." + method)
        self.propagator = getattr(sys.modules["XPACDT.Dynamics." + method],
                                  method)(self.pes, self.masses,
                                          **prop_parameters)

        if 'thermostat' in parameters:
            self.propagator.attach_thermostat(parameters, self.masses)
