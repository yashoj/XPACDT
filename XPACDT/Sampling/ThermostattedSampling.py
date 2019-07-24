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

import XPACDT.Tools.Units as units


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

    Other Parameters
    ----------------
    TODO

    Returns
    -------
    systems : (n_sample) list of XPACDT.Dynamics.System
        A list of systems located at the sampled phase-space points.
    """

    sample_parameters = parameters.get('sampling')
    assert('samples' in sample_parameters), "Number of " \
        "samples required, but not given."
    assert('time' in sample_parameters), "Time for each sampling run " \
        "required, but not given."

    sampling_time = units.parse_time(sample_parameters.get('time', '0.0 fs'))

    systems = []
    for i in range(n_sample):
        system.step(sampling_time)

        system.clear_log()
        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.propagator.thermostat = None

    return systems
