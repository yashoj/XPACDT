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
        Representation of the input file.
    time : float
        The initial time of the system in au.

    Attributes:
    -----------
    n_dof
    n_beads
    time
    positions
    momenta
    electrons
    """

    def __init__(self, degrees_of_freedom, input_parameters, time):

        self.n_dof = degrees_of_freedom
        self.time = time

        # coordinates, masses from input
        self.masses = input_parameters.masses
        self.positions = input_parameters.coordinates
        self.momenta = input_parameters.momenta

        self.n_beads = input_parameters.n_beads

        # Set up electrons
        self.init_electrons(input_parameters)

        # set up propagator and attach
        if 'nuclei_propagator' in input_parameters:
            self.attach_nuclei_propagator(input_parameters)

        return

    @property
    def electrons(self):
        """ XPACDT.System.Electrons : The electrons used in the propagation.
            Default: AdiabaticElectrons."""
        return self.__electrons

    @property
    def propagator(self):
        """ The propagator used to advance the nuclei in time. """
        return self.__propagator

    @propagator.setter
    def propagator(self, p):
        self.__propagator = p

    @property
    def time(self):
        """float : Current time of the system in au."""
        return self.__time

    @time.setter
    def time(self, f):
        self.__time = f

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
        """(n_dof) list of int : The number of beads for each degree of freedom."""
        return self.__n_beads

    @n_beads.setter
    def n_beads(self, l):
        self.__n_beads = l

    @property
    def masses(self):
        """(n_dof) ndarray of floats : The masses of each degree of
           freedom in au."""
        return self.__masses

    @masses.setter
    def masses(self, a):
        self.__masses = a.copy()

    @property
    def positions(self):
        """(n_dof, n_beads) ndarray of floats : The positions of all beads in
            the system. The first axis is the degrees of freedom and the
            second axis the beads."""
        return self.__positions

    @positions.setter
    def positions(self, a):
        self.__positions = a.copy()

    @property
    def x_centroid(self):
        """ (n_dof) ndarray of floats : The centroid of each coordinate. """
        return np.mean(self.positions, axis=1)

    @property
    def momenta(self):
        """(n_dof, n_beads) ndarray of floats : The momenta of all beads in
            the system. The first axis is the degrees of freedom and the
            second axis the beads."""
        return self.__momenta

    @momenta.setter
    def momenta(self, a):
        self.__momenta = a.copy()

    @property
    def p_centroid(self):
        """(n_dof) ndarray of floats : The centroid of each momentum. """
        return np.mean(self.momenta, axis=1)

    @property
    def energy(self):
        """ float : Total energy of the nuclei including the spring term.
        i.e. :math:`\\frac{1}{n}(\\sum_i \\sum_j (p^2_{ij})(2m_j)) + SPINGS + V)`.
        TODO: write out properly."""
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
        return self.electrons.energy(self.positions)

    def __eq__(self, other):
        """Test if an object is equal to the current nuclei object. A nuclei
        object is assumed to be equal to another nuclei object if they have
        the same number of degrees of freedom, the same number of beads,
        the same positions, momenta and masses.

        Parameters:
        -----------
        other : any object
            Object to compare to.

        Returns:
        -------
        bool
            Returns True if both objects have the same number of degrees of
            freedom, the same number of beads, thesame positions, momenta
            and masses. False else.
        """
        return (self.n_dof == other.n_dof
                and self.n_beads == other.n_beads
                and (self.positions == other.positions).all()
                and (self.momenta == other.momenta).all()
                and (self.masses == other.masses).all())

    def init_electrons(self, parameters):
        """ Initialize the representation of the electrons in the system.

        Parameters
        ----------
        parameters: XPACDT.Inputfile
            The inputfile object containing all input parameters.
        """

        electronic_method = parameters.get('system').get('electrons',
                                                         'AdiabaticElectrons')
        __import__("XPACDT.System." + electronic_method)
        if (electronic_method != "AdiabaticElectrons"):
            assert(electronic_method in parameters), \
                  ("No input parameters for chosen electronic method.")

        self.__electrons = getattr(sys.modules["XPACDT.System." + electronic_method],
                                   electronic_method)(parameters, self.n_beads,
                                                      self.masses, self.positions, self.momenta)

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
        (selected dof) ndarray of floats if beads is False;
        (selected dof, nbeads) ndarray of floats else
            Array of positions, momenta or velocity values requested.
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
                elif quantity == 'p':
                    if beads:
                        values.append(self.momenta[dof])
                    else:
                        values.append(self.p_centroid[dof])
                elif quantity == 'v':
                    if beads:
                        values.append(self.momenta[dof] / self.masses[dof])
                    else:
                        values.append(self.p_centroid[dof] / self.masses[dof])
                else:
                    raise RuntimeError("XPACDT: Requested quantity not"
                                       " implemented: " + quantity)

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
            prop_parameters['rp_transform_type'] = parameters.get('rpmd').get(
                    "nm_transform", "matrix")

        prop_method = prop_parameters.get('method')
        __import__("XPACDT.Dynamics." + prop_method)
        self.propagator = getattr(sys.modules["XPACDT.Dynamics." + prop_method],
                                  prop_method)(self.electrons, self.masses,
                                               self.n_beads, **prop_parameters)

        if 'thermostat' in parameters:
            self.propagator.attach_thermostat(parameters, self.masses)

    def propagate(self, time_propagate):
        """ This functions advances the positions and momenta of the nuclei
        for a given time using the proapgator assigned. The electronic
        subsystem is advanced for the same time.

        Parameters
        ----------
        time_propagate : float
            The time to advance the nuclei and electrons in au.
        """
        # TODO : choose a better name than 'step_index'. Also do we need to
        #        pass both R and P for all electronic methods?
        self.electrons.step(self.positions, self.momenta, time_propagate, **{'step_index': 'first'})

        self.positions, self.momenta = \
            self.__propagator.propagate(self.positions, self.momenta,
                                        time_propagate)

        self.electrons.step(self.positions, self.momenta, time_propagate, **{'step_index': 'last'})

        self.time += time_propagate

        return
