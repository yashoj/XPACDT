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

import copy
from molmod.units import parse_unit
import sys

import XPACDT.Dynamics.VelocityVerlet as vv
import XPACDT.Dynamics.Nuclei as nuclei


class System(object):
    """This class is the main representation of the system treated with XPACDT.
    It sets up the calculation, stores all important objects.

    Parameters
    ----------


### TODO: stuff we need:
    - number of dof
    - masses/atoms

    - Interface (own classes)

    - Nuclear part (own class)
      -> # beads
      -> positions
      -> velocities
      -> propagator

    - Electronic part (own class)
      -> expansion coefficients
    - -> electron dynamics part

    - restrictions
    - pes where


    """

    def __init__(self, parameters):

        assert('Interface' in parameters.get_section("system")), "Interface " \
            "not specified!"
        assert('dof' in parameters.get_section("system")), "Number of " \
            "degrees of freedom not specified!"

        self.n_dof = parameters.get_section("system").get("dof")
        time_string = parameters.get_section("system").get("time", "0 fs")
        self.time = float(time_string[0]) * parse_unit(time_string[1])

        # Set up potential
        pes_name = parameters.get_section("system").get("Interface", None)
        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(**parameters.get_section(pes_name))

        # TODO: init nuclei
        self._init_nuclei(parameters)
        # TOOD: init electrons
        self._init_electrons(parameters)

        self.__parameters = parameters

        self.log(init=True)

    @property
    def n_dof(self):
        """int : The number of degrees of freedom in this system."""
        return self.__n_dof

    @n_dof.setter
    def n_dof(self, i):
        assert (int(i) > 0), "Number of degrees of freedom less than 1!"
        self.__n_dof = int(i)

    @property
    def time(self):
        """float : Current time of the system in au."""
        return self.__time

    @time.setter
    def time(self, f):
        self.__time = f

    @property
    def parameters(self):
        """XPACDT.Input.Inputfile : The parameters from the input file."""
        return self.__parameters

    @property
    def nuclei(self):
        """XPACDT.Dynamics.Nuclei : The nuclei in this system."""
        return self.__nuclei

    @property
    def pes(self):
        """InterfaceTemplate : Potential energy interface of the system."""
        return self.__pes

    def _init_nuclei(self, parameters):
        """ Function to set up the nuclei of a system including all associated
        objects like propagators.

        Parameters
        ----------
        parameters : XPACDT input file
            Dictonary-like presentation of the input file.
        """

        # coordinates, masses from input - reshape and test some consistency
        self.masses = parameters._masses
        if parameters._c_type == 'mass-value':
            coordinates = parameters._coordinates.reshape((self.n_dof, -1))
        elif parameters._c_type == 'xyz':
            coordinates = parameters._coordinates.T.reshape((self.n_dof, -1))

        try:
            momenta = parameters._momenta.reshape(coordinates.shape)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Number of given momenta and "
                          "coordinates does not match!")

        self.__nuclei = nuclei.Nuclei(self.n_dof, coordinates, momenta,
                                      n_beads=[coordinates.shape[1]])

        # set up propagator and attach
        if parameters.get_section('propagator') is not None:
            prop_parameters = parameters.get_section('propagator')
            if parameters.get_section("rpmd"):
                assert('beta' in parameters.get_section("rpmd")), "No beta " \
                        "given for RPMD."
                prop_parameters['beta'] = parameters.get_section("rpmd")

            method = prop_parameters.get('method')
            __import__("XPACDT.Dynamics." + method)
            propagator = getattr(sys.modules["XPACDT.Dynamics." + method],
                                 method)(self.__pes, self.masses,
                                         **prop_parameters)

            self.__nuclei.propagator = propagator

    def _init_electrons(self, parameters):
        """ Function to set up the electrons of a system including all
        associated objects like propagators. Not yet implemented.

        Parameters
        ----------
        parameters : XPACDT input file
            Dictonary-like presentation of the input file.
        """

        self.__electrons = None

    def step(self, time):
        """ Step in time."""
        # TODO: more advanced here.
        # TODO: add electrons
        # TODO: add logging
        # TODO: split in timesteps
        self.__nuclei.propagate(time)
        self.time += time
        self.log()

    def log(self, init=False):
        if init:
            self._log = []
        self._log.append([self.time, copy.deepcopy(self.nuclei)])
