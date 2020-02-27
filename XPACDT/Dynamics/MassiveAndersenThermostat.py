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

""" This module defines the massive Andersen Thermostat.
"""

import numpy as np
import warnings

import XPACDT.Tools.Units as units


class MassiveAndersen(object):
    """Implementation of the massive Andersen thermostat. With a given
    timescale all momenta of the sytem are re-drawn from the Maxwell-Boltzman
    distribution at a given temperature.
    The Andersen thermostat is described in
    https://aip.scitation.org/doi/10.1063/1.439486, but for the 'massive'
    variant, the momenta are not redrawn each timestep but at infrequent
    steps.

    Parameter
    ---------
    input_parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    masses : (n_dof) ndarray of floats
        The nuclear masses of the current system in au.
    """
    def __init__(self, input_parameters, masses):
        thermo_parameters = input_parameters.get('thermostat')
        sampling_parameters = input_parameters.get('sampling')

        if 'temperature' in thermo_parameters:
            self.temperature = float(thermo_parameters.
                                     get('temperature').split()[0])

            # Consistency check
            if sampling_parameters is not None and 'temperature' in sampling_parameters:
                sampling_temperature = float(sampling_parameters.
                                             get('temperature').split()[0])

                # Avoid round-off errors; There should be really no difference
                # in results if temperatures given deviate by 1e-4
                if abs(self.temperature - sampling_temperature) > 1e-4:
                    raise RuntimeError("Temperatures given in thermostat"
                                       "(" + str(self.temperature) + " K) and"
                                       " in samping ("
                                       + str(sampling_temperature) + " K)"
                                       " do not match!")
        else:
            raise RuntimeError("No temperature given for MassiveAndersen!")

        
        if 'time' not in thermo_parameters:
            raise KeyError("\nXPACDT: No time given for Massive Andersen "
                           "thermostat.")

        self.__timescale = units.parse_time(thermo_parameters.get("time"))
        if sampling_parameters is not None and 'time' in sampling_parameters:
            sampling_time = units.parse_time(sampling_parameters.get('time'))
            if abs(self.__timescale - sampling_time) > 1e-4:
                warnings.warn(f"\nXPACDT: Sampling time ({sampling_time})"
                              f" and timescale for thermostat ({self.__timescale})"
                              f"  do not match. Is this desired?",
                              category=RuntimeWarning)

        self.beta = 1.0 / (self.temperature * units.boltzmann)
        self.mass = masses

    def apply(self, R, P, state, time):
        """Apply the thermostat. All Ps are redrawn from a Maxwell-Boltzman
        distribution.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            Positions of the system. Ignored here.
        P : (n_dof, n_beads) ndarray of floats
            Momenta of the system that will be reset.
        state : int
            Step within the propagation, where the function is called.
            For Velocity-Verlet this is:
            0: After propagating for a certain time interval
                (this is where this thermostat is applied).
            1: After the first velocity step.
            2: After the 'verlet' step.
            3. After the second velocity step.
        time : float
            The current absolute time of the propagation in au.

        Returns
        --------
        Nothing, but the new momenta are set into the given P array.
        """

        if state != 0:
            return
        else:
            # Check if it is time for applying the thermostat
            mod_time = (time + 1e-8) % self.__timescale
            if mod_time < 1e-6:
                n_beads = P.shape[1]
                sigmas = np.sqrt(np.ones_like(P) * self.mass[:, None] * n_beads
                                 / self.beta)
                # Make sure that the momenta are actually changed in the calling
                # modules. P = ... won't work.
                P[:, :] = np.random.normal(np.zeros_like(sigmas), sigmas)
        return
