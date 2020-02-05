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

""" Module that defines the physical system treated in XPACDT."""

import copy

import XPACDT.System.Nuclei as nuclei
import XPACDT.Tools.Units as units


class System(object):
    """This class is the main class representing the system state. It stores
    the nuclei object and takes care of logging.

    Parameters
    ----------
    input_parameters : XPACDT.Inputfile
        Represents all the input parameters for the simulation given in the
        input file.

    Attributes:
    -----------
    log
    parameters
    nuclei
    """

    def __init__(self, input_parameters):

        self.__parameters = input_parameters
        time = units.parse_time(self.parameters.get("system").
                                get("time", "0 fs"))

        # Set up nuclei
        self.__nuclei = nuclei.Nuclei(self.parameters, time)

        # Set up log
        self.do_log(init=True)

    @property
    def log(self):
        """list of XPACDT.System.Nuclei : Log of the system history as a list
        of the states of the nuclei, which also carry the information on the
        electrons, times, etc."""
        return self.__log

    @property
    def parameters(self):
        """XPACDT.Input.Inputfile : The parameters from the input file."""
        return self.__parameters

    @parameters.setter
    def parameters(self, new_parameters):
        # TODO: See Ticket XPACDT-65
        pass

    def optimize_geometry(self):
        self.nuclei.optimize_geometry()
        # Reset log with optimized geometry as initial state
        self.do_log(init=True)

    @property
    def nuclei(self):
        """XPACDT.Dynamics.Nuclei : The nuclei in this system."""
        return self.__nuclei

    def step(self, time_propagate, sparse=False):
        """ Step the whole system forward in time. Also keep a log of the
        system state at these times.

        Parameters
        ----------
        time_propagate : float
            Time to advance the system in au.
        sparse : bool, optional, default: False
            Whether to keep a sparse (less memory consuming) log or not
        """
        self.__nuclei.propagate(time_propagate)
        self.do_log(sparse=sparse)

    def reset(self, time=None):
        """ Reset the system state to its original values and clear everything
        else in the log. Optionally, the time can be set to a given value.

        Parameters
        ----------
        time : float, optional, default None
            System time to be set, if given.
        """

        self.__nuclei = copy.deepcopy(self.__log[0])
        if time is not None:
            self.__nuclei.time = time
        self.do_log(True)

    def clear_log(self):
        """ Set the current system state as initial state and clear everything
        else in the log."""

        self.__nuclei = copy.deepcopy(self.__log[-1])
        self.do_log(True)

    def do_log(self, init=False, sparse=False):
        """ Log the system state to a list called __log. Each entry is a
        XPACDT.System.Nuclei object, which contains all important quantities
        like time, position, momenta, energies, etc.

        Parameters
        ----------
        init : bool, optional
            If the log has to be initialized or not. Default False.
        sparse : bool, optional, default: False
            Whether to keep a sparse (less memory consuming) log or not
        """

        if init:
            self.__log = []

        self.__log.append(copy.deepcopy(self.nuclei))

        # Use a sprase log by removing the propagator object
        if sparse:
            self.__log[-1].propagator = None
