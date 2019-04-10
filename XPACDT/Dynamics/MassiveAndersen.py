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

""" This module holds the defines te massive Andersen Thermostat.
"""

import molmod.constants as const
from molmod.units import parse_unit
import numpy as np

# TODO also some options for only thermostatting internal DOF


class MassiveAndersen(object):
    """Implementation of the massive Andersen thermostat. With a given
    timescale all momenta of the sytem are re-drawn from the Maxwell-Boltzman
    distribution at a given temperature.

    Parameter
    ---------
    parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    masses : ndarray of floats
        The atomic masses of the current system in au.

    Other Parameters
    ----------------
    Within the inputfile the timescale for resampling and the temperature
    need to be given. Either as 'time' (in fs or au) and 'temperature' (in K)
    in the thermostat or the sampling sections.
    """
    def __init__(self, parameters, masses):
        # TODO: basic argument parsing here

        thermo_parameters = parameters.get('thermostat')
        sampling_parameters = parameters.get('sampling')

        if 'time' in thermo_parameters:
            timestring = thermo_parameters.get("time").split()
        elif 'time' in sampling_parameters:
            timestring = sampling_parameters.get("time").split()
        else:
            raise RuntimeError("No timescale given for MassiveAndersen!")

        self.time = float(timestring[0]) * parse_unit(timestring[1])

        if 'temperature' in thermo_parameters:
            self.temperature = float(thermo_parameters.get('temperature').split()[0])
        elif 'temperature' in sampling_parameters:
            self.temperature = float(sampling_parameters.get('temperature').split()[0])
        else:
            raise RuntimeError("No temperature given for MassiveAndersen!")

        self.beta = 1.0 / (self.temperature * const.boltzmann)
        self.mass = masses

    def apply(self, R, P, state):
        """Apply the thermostat. All Ps are redrawn from a Maxwell-Boltzman
        distribution.

        Parameters
        ----------
        R : two-dimensional ndarray of floats
            Positions of the system. Ignored here.
        P : two-dimensional ndarray of floats
            Momenta of the system that will be reset.
        state : int
            Step within the propagation, where the function is called.
            For Velocity-Verlet this is:
            0: After propagating for a certain time interval
                (this is where this thermostat is applied).
            1: After the first velocity step.
            2: After the 'verlet' step.
            3. After the second velocity step.

        Returns
        --------
        Nothing, but the new momenta are set into the given P array.
        """

        if state != 0:
            return
        else:
            n_beads = P.shape[1]
            sigmas = np.sqrt(np.ones_like(P)*self.mass[:, None] * n_beads / self.beta)
            P[:, :] = np.random.normal(np.zeros_like(sigmas), sigmas)
        return