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

"""This module defines the state of the nuclei treated in XPACDT."""

import math
import numpy as np
import warnings
import sys

import XPACDT.Tools.Memory as mem


class Nuclei(object):
    """
    This class represents the nuclei. It also holds the XPACDT.System.Electrons
    object for the current physical system.

    Parameters:
    -----------
    input_parameters : XPACDT.Input.Inputfile
        Representation of the input file.
    time : float
        The initial time of the system in au.

    Attributes:
    -----------
    n_dof
    n_beads
    beta
    time
    positions
    x_centroid
    momenta
    p_centroid
    electrons
    propagator
    energy
    energy_centroid
    kinetic_energy
    spring_energy
    potential_energy
    kinetic_energy_centroid
    potential_energy_centroid
    """

    def __init__(self, input_parameters, time):
        # coordinates, masses from input
        self.__masses = input_parameters.masses
        self.__n_beads = input_parameters.n_beads
        self.__n_dof = input_parameters.n_dof
        self.__beta = input_parameters.beta

        self.positions = input_parameters.coordinates
        self.momenta = input_parameters.momenta

        # Set up electrons
        self.init_electrons(input_parameters)

        # set up propagator and attach
        if 'nuclei_propagator' in input_parameters:
            self.attach_nuclei_propagator(input_parameters)

        self.time = time

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
        warnings.warn("\nXPACDT: Setting the propagator with the setter"
                      " should only be used in the"
                      " UnitTests. If you are currently not running a"
                      " UnitTest, something has gone WRONG!",
                      category=RuntimeWarning)

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

    @property
    def n_beads(self):
        """(n_dof) list of int : The number of beads for each degree of
        freedom."""
        return self.__n_beads

    @property
    def beta(self):
        """ float or np.nan: Inverse temperature for ring polymer springs in
        a.u. It is NaN if not given in the case of 1 bead for each degree of
        freedom."""
        return self.__beta

    @beta.setter
    def beta(self, f):
        self.__beta = f

    @property
    def masses(self):
        """(n_dof) ndarray of floats : The masses of each degree of
           freedom in au."""
        return self.__masses

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
        if a is not None:
            self.__momenta = a.copy()
        else:
            self.__momenta = None

    @property
    def p_centroid(self):
        """(n_dof) ndarray of floats : The centroid of each momentum. """
        return np.mean(self.momenta, axis=1)

    @property
    def energy(self):
        """ float : Total energy of the nuclei of the ring polymer including
        the spring term in au.
        i.e. :math:`\\frac{1}{n}(\\sum_i \\sum_j (p^2_{ij})(2m_j)) + SPINGS + V)`.
        TODO: write out properly."""
        return self.kinetic_energy + self.spring_energy + self.potential_energy

    @property
    def energy_centroid(self):
        """ float : Total energy of the nuclei of the centroid in au.
        i.e. :math:`(\\sum_i \\sum_j (p^2_{ij})(2m_j)) + V)`.
        TODO: write out properly."""
        return self.kinetic_energy_centroid + self.potential_energy_centroid

    @property
    def kinetic_energy(self):
        """ float : 'Kinetic energy' of the ring polymer in au.
        i.e. :math:`(\\sum_\\alpha \\sum_A (p^2_{A}^{(\\alpha))})(2m_A)))`."""
        return (0.5 * np.sum(np.sum((self.momenta * self.momenta), axis=1)
                             / self.masses))

    @property
    def spring_energy(self):
        """ float : Energy due to spring terms in the ring polymer in au.
        TODO : write equation properly"""
        if np.all([i == 1 for i in self.n_beads]):
            return 0.0
        else:
            prefactor = 0.5 * (float(max(self.n_beads)) / self.beta)**2
            return (prefactor * np.sum(self.masses
                                       * np.sum((self.positions
                                                 - np.roll(self.positions, -1, axis=1))**2, axis=1)))

    @property
    def potential_energy(self):
        """ float : Potential energy of the ring polymer in au."""
        return np.sum(self.electrons.energy(self.positions))

    @property
    def kinetic_energy_centroid(self):
        """ float : Kinetic energy of the centroid in au."""
        return (0.5 * np.dot(self.p_centroid, (self.p_centroid / self.masses)))

    @property
    def potential_energy_centroid(self):
        """ float : Potential energy of the centroid in au."""
        return self.electrons.energy(self.positions, centroid=True)

    def __eq__(self, other):
        """Test if an object is equal to the current nuclei object. A nuclei
        object is assumed to be equal to another nuclei object if they have
        the same number of degrees of freedom, the same number of beads,
        the same beta, the same positions, momenta and masses.

        Parameters:
        -----------
        other : XPACDT.System.Nuclei object
            Nuclei object to compare to.

        Returns:
        -------
        bool
            Returns True if both objects have the same number of degrees of
            freedom, the same number of beads, thesame positions, momenta
            and masses. False else.
        """
        # 'beta' is tested using 'is' since it can have the value np.nan
        # and np.nan == np.nan is False, so need another way to test it.
        return (self.n_dof == other.n_dof
                and self.n_beads == other.n_beads
                and self.beta is other.beta
                and self.time == other.time
                and (self.positions == other.positions).all()
                and (self.momenta == other.momenta).all()
                and (self.masses == other.masses).all())

    def make_sparse(self):
        """ Decrease size of the object for sparse logging.
        Right now this removes the propagator object.
        """

        self.__propagator = None

    def init_electrons(self, parameters):
        """ Initialize the representation of the electrons in the system. This
        creates a XPACDT.System.Electrons object.

        Parameters
        ----------
        parameters: XPACDT.Inputfile
            The inputfile object containing all input parameters.
        """

        electronic_method = parameters.get('system').get('electrons',
                                                         'AdiabaticElectrons')
        __import__("XPACDT.System." + electronic_method)
        if (electronic_method != "AdiabaticElectrons"):
            if electronic_method not in parameters:
                raise KeyError("\nXPACDT: No input parameters for chosen"
                               "electronic method.")

        self.__electrons = getattr(sys.modules["XPACDT.System." + electronic_method],
                                   electronic_method)(parameters,
                                                      self.masses,
                                                      self.positions,
                                                      self.momenta)

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

    def get_selected_quantities(self, dof_string, quantity='x', beads=False):
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
                    raise RuntimeError("\nXPACDT: Requested quantity not"
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
            prop_parameters['rp_transform_type'] = parameters.get('rpmd').get(
                    "nm_transform", "matrix")

        prop_method = prop_parameters.get('method')
        __import__("XPACDT.Dynamics." + prop_method + "Propagator")
        self.__propagator = getattr(sys.modules["XPACDT.Dynamics." + prop_method + "Propagator"],
                                    prop_method)(self.electrons, self.masses,
                                                 self.n_beads, self.beta,
                                                 **prop_parameters)

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
        # 1e-8 is added because of floating point representation issue needed
        # for proper floor division or modulo operation. This value is chosen
        # since it is greater than machine error and less than typical
        # propagation timesteps.
        time_plus = time_propagate + 1e-8
        timestep = self.propagator.timestep
        n_steps = int(time_plus // timestep)

        # This is needed since nuclear propagator has this fixed timestep.
        # 'time_propagate' can be output time in propagation or sampling time
        # in thermostated sampling.
        if not math.isclose(n_steps*timestep, time_propagate, abs_tol=1e-6):
            raise RuntimeError("\nXPACDT: Propagation time is not multiple of"
                               " nuclear timestep.")

        for i in range(n_steps):
            self.electrons.step(self.positions, self.momenta, timestep,
                                **{'step_index': 'before_nuclei'})
            self.positions, self.momenta = \
                self.__propagator.propagate(self.positions, self.momenta,
                                            timestep, self.time + i*timestep)
            self.electrons.step(self.positions, self.momenta, timestep,
                                **{'step_index': 'after_nuclei'})

        self.time += time_propagate

        return

    def print_size(self):
        """ Print the size of the nuclei object and some of its members.
        """

        print("Nuclei is {: .2f} KB".format(mem.getsize(self) / 1024))
        print("\t Consisting of:")
        print("\t n_dof {: .2f} KB".format(mem.getsize(self.n_dof) / 1024))
        print("\t time {: .2f} KB".format(mem.getsize(self.time) / 1024))
        print("\t beta {: .2f} KB".format(mem.getsize(self.beta) / 1024))
        print("\t n_beads {: .2f} KB".format(mem.getsize(self.n_beads) / 1024))
        print("\t masses {: .2f} KB".format(mem.getsize(self.masses) / 1024))
        print("\t positions {: .2f} KB".format(mem.getsize(self.positions) / 1024))
        print("\t momenta {: .2f} KB".format(mem.getsize(self.momenta) / 1024))
        print("\t propagator {: .2f} KB".format(mem.getsize(self.propagator) / 1024))
        print("\t electrons {: .2f} KB".format(mem.getsize(self.electrons) / 1024))
