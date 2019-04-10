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


def do_Thermostatted_sampling(system, parameters):
    """ Perform sampling by running a long, thermostatted trajectory and
    periodically saving the state as a new sample.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        A representation of the basic system with potential interface set up
        and a valid starting geometry.
    parameters : XPACDT input file
        Dictonary-like presentation of the input file.

    Other Parameters
    ----------------
    TODO
    """
    assert('samples' in parameters.get('sampling')), "Number of " \
        "samples required not given."

    n_sample = int(parameters.get("sampling").get('samples'))
    sampling_time = system.nuclei.propagator.thermostat.time
    systems = []
    for i in range(n_sample):
        system.step(sampling_time)

        system.clear_log()
        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.propagator.thermostat = None

    return systems
