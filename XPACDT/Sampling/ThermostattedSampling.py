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

""" Module implementing thermostatted sampling, i.e., drawing samples from a
long, thermostatted (MD) trajectory.
"""


import copy

import XPACDT.Tools.Units as units

from XPACDT.Input.Error import XPACDTInputError


def do_Thermostatted_sampling(system, parameters, n_sample):
    """ Perform sampling by running a long, thermostatted trajectory and
    periodically saving the state as a new sample.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        A representation of the basic system with potential interface set up
        and a valid starting geometry.
    parameters : XPACDT input file
        Dictonary-like presentation of the input file.
    n_sample : int
        Actual number of samples required.

    Returns
    -------
    systems : (n_sample) list of XPACDT.Dynamics.System
        A list of systems located at the sampled phase-space points.
    """

    if system.nuclei.momenta is None:
        raise XPACDTInputError(
            "Momenta not provided in system or input file, but required in "
            "thermostatted sampling.",
            section="momenta")

    sample_parameters = parameters.get('sampling')

    if 'time' not in sample_parameters:
        raise XPACDTInputError(
            "Time for each sampling run required, but not given.",
            section="sampling",
            key="time")

    if 'thermostat' not in parameters:
        raise XPACDTInputError(
            "A thermostat is required for thermostatted sampling.",
            section="thermostat")

    sampling_time = units.parse_time(sample_parameters.get('time', '0.0 fs'))

    systems = []
    for i in range(n_sample):
        system.step(sampling_time)

        system.clear_log()
        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.time = 0.0

    return systems
