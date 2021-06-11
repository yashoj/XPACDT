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

""" This module implements all required routines to propagate a given system in
real time."""

import os
import sys
import pickle

import XPACDT.Tools.Units as units


def propagate(system, input_parameters, initiated=False):
    """ Propagate the system as given in the input file. The system history is
    saved to a pickle file during or after propagation.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        System that defines the initial geometry and the potential.
    input_parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    initiated : bool, optional, default: False
        True if the system has been recently initiated for the real time propagation
        from an input file (in xpacdt.py)
        False if the system read from file/been sampled before.
    """

    if system.nuclei.momenta is None:
        raise RuntimeError("\nXPACDT: Momenta not provided in system or input"
                           " file, but required in real time propagation.")

    prop_parameters = input_parameters.get('propagation')
    sys_parameters = input_parameters.get('system')

    assert('time_end' in prop_parameters), "No endtime " \
        "given for the propagation."

    if 'continue' not in prop_parameters:
        # set initial time and reset log
        system.reset(time=units.parse_time(prop_parameters.get('time_start',
                                                               '0.0 fs')))
        # reset the electrons if not already initiated
        if not initiated:
            system.nuclei.init_electrons(input_parameters)
        system.do_log(init=True)

    # Reset beta; Set desired propagator
    system.nuclei.beta = input_parameters.beta
    system.nuclei.attach_nuclei_propagator(input_parameters)

    # Obtain times for propagation and output
    time_end = units.parse_time(prop_parameters.get('time_end'))

    timestep_output = prop_parameters.get('time_output')
    if timestep_output is not None:
        timestep_output = units.parse_time(timestep_output)
    else:
        timestep_output = system.nuclei.propagator.timestep

    # set up pickle file
    name_folder = sys_parameters.get('folder')
    if not os.path.isdir(name_folder):
        try:
            os.mkdir(name_folder)
        except OSError:
            sys.stderr.write("Creation of trajectory folder failed!")
            raise
    name_file = sys_parameters.get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)

    while(system.nuclei.time < time_end):
        # TODO: making sparse should be an input option with default True.
        system.step(timestep_output, sparse=True)

        if 'intermediate_write' in prop_parameters:
            pickle.dump(system, open(path_file, 'wb'), -1)

    pickle.dump(system, open(path_file, 'wb'), -1)
