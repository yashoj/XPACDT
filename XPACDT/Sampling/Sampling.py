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

""" This module defines all required routines to generate an initial set of
systems for real-time propagation."""

import bz2
import glob
import os
import pickle
import shutil
import sys
import warnings

import XPACDT.Tools.Xtools as xtools


def sample(system, parameters, do_return=False):
    """
    Basic sampling method. This function creates the folder given in the input
    to put the sampled data to. If the folder already exists, either overwrite
    or add has to be given for removing all old trajectories or adding to the
    existing trajectories, respectively. If none of the two is given, a
    RuntimeError is raised.
    The function calls the given sampling method and obtains a list of systems,
    which it then employs pickle to write to folders named 'traj_XXX',
    with XXX numbering the trajectories.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        System that defines the initial geometry and the potential.
    parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    do_return: Bool, optional, default:False
        If True then the list of samples systems is returned instead of written
        to pickle files.

    Return
    ------
    If `do_return` is True, the list of samples systems is returned.
    """

    sampling_parameters = parameters.get('sampling')
    system_parameters = parameters.get('system')

    if 'folder' not in system_parameters:
        raise KeyError("\nXPACDT: No folder for trajectories given.")
    if 'method' not in sampling_parameters:
        raise KeyError("\nXPACDT: No sampling method specified.")
    if 'samples' not in sampling_parameters:
        raise KeyError("\nXPACDT: Number of samples required not given.")

    # Create or handle trajectory folder.
    n_samples = int(sampling_parameters.get('samples'))
    n_samples_required = n_samples
    compressed = 'compressed_pickle' in system_parameters

    if compressed:
        default_file_name = 'pickle.bz2'
    else:
        default_file_name = 'pickle.dat'
    name_folder = system_parameters.get('folder')
    file_name = system_parameters.get('picklefile', default_file_name)
    if not os.path.isdir(name_folder):
        try:
            os.mkdir(name_folder)
        except OSError as e:
            raise type(e)(str(e) + "\nXPACDT: Creation of trajectory folder"
                          " failed!")
    else:
        trj_folder_list = glob.glob(os.path.join(name_folder, 'trj_*'))
        trj_folder_list.sort()

        if 'overwrite' in sampling_parameters:
            if trj_folder_list is not None:
                for folder in trj_folder_list:
                    shutil.rmtree(folder)

        elif 'add' in sampling_parameters:
            n_samples_required -= len(trj_folder_list)
            warnings.warn("\nXPACDT: Please make sure to use different"
                          " seeds in the input file when adding"
                          " trajectories.")
        else:
            raise RuntimeError("The trajectory folder already exists and no "
                               "directive for overwriting old data or adding "
                               "trajectories is given.")

    # Run sampling method
    method = sampling_parameters.get('method')
    __import__("XPACDT.Sampling." + method + "Sampling")
    sampled_systems = getattr(sys.modules["XPACDT.Sampling." + method + "Sampling"],
                              "do_" + method + "_sampling")(system, parameters,
                                                            n_samples_required)

    # Shift centroid position and/or momenta if requested
    if parameters.positionShift is not None:
        for system in sampled_systems:
            system.nuclei.positions += parameters.positionShift[:, None]
            system.do_log(init=True)

    if parameters.momentumShift is not None:
        for system in sampled_systems:
            system.nuclei.momenta += parameters.momentumShift[:, None]
            system.do_log(init=True)

    if do_return is True:
        # Add the existing samples if they exists.
        if 'add' in sampling_parameters:
            dirs = xtools.get_directory_list(name_folder, file_name)
            for system in xtools.get_systems(dirs, file_name, None,
                                             compressed):
                sampled_systems.append(system)

        return sampled_systems

    # Save stuff to pickle files. Iterate over all possible folders
    shift = 0
    for i in range(n_samples):
        trj_folder = os.path.join(name_folder, 'trj_{0:07}'.format(i))

        # Check if this folder already exists;
        # if not, create and save the next system
        if not os.path.isdir(trj_folder):
            try:
                os.mkdir(trj_folder)
            except OSError as e:
                raise type(e)(str(e) + "\nXPACDT: Creation of trajectory"
                              " folder " + trj_folder + " failed!")

            filename_w_path = os.path.join(trj_folder, file_name)

            if compressed:
                out_file = bz2.BZ2File(filename_w_path, 'wb')
            else:
                out_file = open(filename_w_path, 'wb')

            pickle.dump(sampled_systems[shift], out_file, -1)
            out_file.close()

            shift += 1

    return
