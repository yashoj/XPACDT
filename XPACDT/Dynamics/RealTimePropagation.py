#!/usr/bin/env python3

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

import pickle

# import XPACDT.Dynamics.System as xSystem
# import XPACDT.Input.Inputfile as infile


def propagate(system, parameters):
    """ Propagate the system."""

    picklefile_name = parameters.get_section('system').\
        get('picklefile', 'pickle.dat')
    if parameters.get_section('system').get('restart') is not None:
        system = pickle.load(open(picklefile_name, 'rb'))

    time_end = float(parameters.get_section('propagation').get('time_end'))
    timestep_output = parameters.get_section('propagation').get('time_output')
    if timestep_output is not None:
        timestep_output = float(timestep_output)
    else:
        timestep_output = float(parameters.get_section('propagation').
                                get('timestep_nuclei'))

    print(system.time, system.nuclei.x_centroid[0])
    while(system.time < time_end):
        system.step(timestep_output)
        print(system.time, system.nuclei.x_centroid[0])

    pickle.dump(system, open(picklefile_name, 'wb'), -1)
