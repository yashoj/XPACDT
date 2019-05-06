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

""" Module that defines the physical system treated in XPACDT. This is the
core of XPACDT."""

import copy
from molmod.units import parse_unit
import sys

import XPACDT.System.Nuclei as nuclei


class System(object):
    """This class is the main class representing the system state. It stores
    all important objects.

    Parameters
    ----------
    input_parameters : XPACDT.Inputfile
        Represents all the input for the simulation.

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

        assert('Interface' in parameters.get("system")), "Interface " \
            "not specified!"
        assert('dof' in parameters.get("system")), "Number of " \
            "degrees of freedom not specified!"

        self.n_dof = parameters.get("system").get("dof")
        time_string = parameters.get("system").get("time", "0 fs").split()
        self.time = float(time_string[0]) * parse_unit(time_string[1])

        # Set up potential
        pes_name = parameters.get("system").get("Interface", None)
        __import__("XPACDT.Interfaces." + pes_name)
        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(**parameters.get(pes_name))

        # Set up nuclei
        self.__nuclei = nuclei.Nuclei(self.n_dof, parameters, self.pes)

        # TOOD: Set up electrons
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

    def reset(self):
        """ Reset to original values. """
        self.time = self._log[0][0]
        self.nucei = copy.deepcopy(self._log[0][1])

    def clear_log(self):
        """ Set the current state as initial state and clear everything else."""
        self.time = self._log[0][0]
        self.nucei = copy.deepcopy(self._log[-1][1])
        self.log(True)

    def log(self, init=False):
        """ Log the system state. """
        if init:
            self._log = []
        self._log.append([self.time, copy.deepcopy(self.nuclei)])
