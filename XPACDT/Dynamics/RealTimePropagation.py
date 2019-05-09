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

""" This module defines all required routines to propagate a given system in
real time."""

from molmod.units import parse_unit
import os
import pickle


def propagate(system, input_parameters):
    """ Propagate the system as given in the input file. The system state is
    saved in a pickle file.

    TODO: Talk more about
    possibilities for given things in the input file.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        System that defines the initial geometry and the potential.
    input_parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    """

    # TODO: put time parsing into function?!

    prop_parameters = input_parameters.get('propagation')
    sys_parameters = input_parameters.get('system')

    assert('time_end' in prop_parameters), "No endtime " \
        "given for the propagation."

    if 'continue' not in sys_parameters:
        system.reset()

        # set initial time
        time_string = prop_parameters.get('time_start', '0.0 fs').split()
        system.time = float(time_string[0]) * parse_unit(time_string[1])

    # Set desired propagator
    system.nuclei.attach_nuclei_propagator(input_parameters)

    # Obtain times for propagation and output
    time_string = prop_parameters.get('time_end').split()
    time_end = float(time_string[0]) * parse_unit(time_string[1])

    timestep_output = prop_parameters.get('time_output')
    if timestep_output is not None:
        time_string = timestep_output.split()
        timestep_output = float(time_string[0]) * parse_unit(time_string[1])
    else:
        timestep_output = system.nuclei.propagator.timestep

    # set up pickle file
    name_folder = sys_parameters.get('folder')
    name_file = sys_parameters.get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)
    while(system.time < time_end):
        system.step(timestep_output)

        # TODO: Learn how to append in pickle and maybe do that
        if 'intermediate_write' in sys_parameters:
            pickle.dump(system, open(path_file, 'wb'), -1)

    pickle.dump(system, open(path_file, 'wb'), -1)
