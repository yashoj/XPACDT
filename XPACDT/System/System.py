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

import XPACDT.System.Nuclei as nuclei
import XPACDT.Tools.Units as units


class System(object):
    """This class is the main class representing the system state. It stores
    the nuclei and takes care of logging.

    Parameters
    ----------
    input_parameters : XPACDT.Inputfile
        Represents all the input parameters for the simulation given in the
        input file.
    """

    def __init__(self, input_parameters):

        self.__parameters = input_parameters

        assert('dof' in self.parameters.get("system")), "Number of " \
            "degrees of freedom not specified!"

        self.n_dof = self.parameters.get("system").get("dof")
        time = units.parse_time(self.parameters.get("system").get("time", "0 fs"))

        # Set up nuclei
        self.__nuclei = nuclei.Nuclei(self.n_dof, self.parameters, time)

        self.do_log(init=True)

    @property
    def log(self):
        """list of XPACDT.System.Nuclei : Log of the system history as a list
        of the states of the nuclei, which also carry the information on the
        electrons, times, etc."""
        return self.__log

    @property
    def n_dof(self):
        """int : The number of degrees of freedom in this system."""
        return self.__n_dof

    @n_dof.setter
    def n_dof(self, i):
        assert (int(i) > 0), "Number of degrees of freedom less than 1!"
        self.__n_dof = int(i)

    @property
    def parameters(self):
        """XPACDT.Input.Inputfile : The parameters from the input file."""
        return self.__parameters

    @property
    def nuclei(self):
        """XPACDT.Dynamics.Nuclei : The nuclei in this system."""
        return self.__nuclei

    def step(self, time_propagate):
        """ Step the whole system forward in time. Also keep a log of the
        system state at these times.

        Parameters
        ----------
        time_propagate : float
            Time to advance the system in au.
        """
        self.__nuclei.propagate(time_propagate)
        self.do_log()

    def reset(self, time=None):
        """ Reset the system state to its original values and clear everything
        else in the log. Optionally, the time can be set to a given value.

        Parameters
        ----------
        time : float, optional, default None
            System time to be set, if given.
        """
        # TODO: after updating new parameters, initialize e- and propagator for 1st log
        self.__nuclei = copy.deepcopy(self.__log[0])
        if time is not None:
            self.__nuclei.time = time
        self.do_log(True)

    def clear_log(self):
        """ Set the current system state as initial state and clear everything
        else in the log."""

        self.__nuclei = copy.deepcopy(self.__log[-1])
        self.do_log(True)

    def do_log(self, init=False):
        """ Log the system state to a list called __log. Each entry is a
        dictonary containing the logged quantities. Currently this logs
        the system time and the nuclei object.

        Parameters
        ----------
        init : bool, optional
            If the log has to be initialized or not. Default False.
        """

        if init:
            self.__log = []

        self.__log.append(copy.deepcopy(self.nuclei))
        # TODO: remove certain parts to not consume too much memory
