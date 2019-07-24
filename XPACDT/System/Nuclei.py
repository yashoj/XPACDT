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

# TODO: add more quantities calculated for the nuclei!


class Nuclei(object):
    """
    This class represents the nuclar degrees of freedom.

    Parameters:
    -----------
    degrees_of_freedom : int
        The number of nuclear degrees of freedom present.
    input_parameters : XPACDT.Input.Inputfile
    pes : XPACDT.Interfaces.InterfaceTemplate


    # TODO: Do we need this or should it be in the input file?
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

    def __init__(self, degrees_of_freedom, input_parameters, pes):

        self.n_dof = degrees_of_freedom
        self.pes = pes

        # coordinates, masses from input
        self.masses = input_parameters.masses
        self.positions = input_parameters.coordinates
        self.momenta = input_parameters.momenta

        self.n_beads = [self.positions.shape[1]]

        # set up propagator and attach
        if 'nuclei_propagator' in input_parameters:
            self.attach_nuclei_propagator(input_parameters)

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

    def getCOM(self, dofs, quantity='x'):
        """ Get the center of mass position, momentum or velocity for a list
        of degree of freedoms.

        Parameters
        ----------
        dofs : list of ints
            Degrees of freedoms to calculate the center of mass quantity for.
        quantity : character, default 'x'
            Type of quantity to calulate. 'x': positions, 'p': momentum,
            'v': velocity

        Returns
        -------
        Currently a NotImplementedError!
        float :
            Center of mass position, momentum or velocity.
        """
        raise NotImplementedError("GetCOM called but not implemented!")

    def parse_dof(self, dof_string, quantity='x', beads=False):
        """ Obtain positions, momenta or velocities for a list of degrees of
        freedom.

        Parameters
        ----------
        dof_string : string
            Degrees of freedoms to calculate the quantity for. This can be a
            comma-separated list of degrees of freedom. Then an array of the
            quantity for each degree of freedom is returned. Or it can be a
            'm' followed by a comma-separated list of degrees of freedom. Then
            the center of mass of that quantity is returned.
        quantity : character, default 'x'
            Type of quantity to calulate. 'x': positions, 'p': momentum,
            'v': velocity
        beads : bool, default: False
            Whether the ring polymer beads should be used (True) or the
            respective centroids (False).

        Returns
        -------
        ndarray of floats
           Array of position, momentum or velocity values requested.
        """
        dofs = dof_string.split(',')

        if dofs[0] == 'm':
            values = self.getCOM([int(v) for v in dofs[1:]], quantity)

        else:
            values = []
            for dof in [int(d) for d in dofs]:
                if quantity == 'x':
                    if beads:
                        values.append(self.positions[dof])
                    else:
                        values.append(self.x_centroid[dof])
                if quantity == 'p':
                    if beads:
                        values.append(self.momenta[dof])
                    else:
                        values.append(self.p_centroid[dof])
                if quantity == 'v':
                    if beads:
                        values.append(self.momenta[dof] / self.masses[dof])
                    else:
                        values.append(self.x_centroid[dof] / self.masses[dof])

        return np.array(values)

    def attach_nuclei_propagator(self, parameters):
        """ Create and attach a propagator to this nuclei representation. If
        required, a thermostatt is added to the propagator as well.

        Parameters
        ----------
        parameters: XPACDT.Inputfile
            The inputfile object containing all input parameters.
        """

        prop_parameters = parameters.get('nuclei_propagator')
        if 'rpmd' in parameters:
            assert('beta' in parameters.get("rpmd")), "No beta " \
                   "given for RPMD."
            prop_parameters['beta'] = parameters.get("rpmd").get('beta')

        prop_method = prop_parameters.get('method')
        __import__("XPACDT.Dynamics." + prop_method)
        self.propagator = getattr(sys.modules["XPACDT.Dynamics." + prop_method],
                                  prop_method)(self.pes, self.masses,
                                               **prop_parameters)

        if 'thermostat' in parameters:
            self.propagator.attach_thermostat(parameters, self.masses)

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
