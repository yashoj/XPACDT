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

""" This module defines all required routines to generate an initial set of
systems for real-time propagation."""

import glob
import os
import pickle
import re
import shutil
import sys


REGEXP_FOLDER_NUMBER = re.compile('trj_(\d+)')


def sample(system, parameters):
    """
    Basic sampling method. This function creates the folder given in the input
    to put the sampled data to. If the folder already exists, either override
    or add has to be given for removing all old trajectories or adding to the
    existing trajectories, respectively. If none of the two is given, a
    RuntimeError is raised.
    The function calls the given sampling method and obtains a list of systems,
    which it then employs pickle to write to folders named 'traj_XXX',
    with XXX numbering the trajectories.

    TODO: Basic analysis if required.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        System that defines the initial geometry and the potential.
    parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    """

    assert('folder' in parameters.get('system')), "No folder for " \
        "trajectories given."
    assert('method' in parameters.get('sampling')), "No sampling " \
        "method specified."
    assert('samples' in parameters.get('sampling')), "Number of " \
        "samples required not given."

    # Create or handle trajectory folder.
    folder_shift = 0
    name_folder = parameters.get('system').get('folder')
    if not os.path.isdir(name_folder):
        try:
            os.mkdir(name_folder)
        except OSError:
            sys.stderr.write("Creation of trajectory folder failed!")
            raise
    else:
        trj_folders = glob.glob(os.path.join(name_folder, 'trj_*'))

        if 'override' in parameters.get('system'):
            for folder in trj_folders:
                shutil.rmtree(folder)

        elif 'add' in parameters.get('system'):
            for folder in trj_folders:

                try:
                    folder_number = int(REGEXP_FOLDER_NUMBER.
                                        search(folder).group(1))

                    if folder_number > folder_shift + 1:
                        folder_shift = folder_number + 1
                except AttributeError:
                    pass

        else:
            raise RuntimeError("The trajectory folder already exists and no "
                               "directive for overriding old data or adding "
                               "trajectories is given.")

    # Run sampling method
    method = parameters.get('sampling').get('method')
    __import__("XPACDT.Dynamics." + method + "Sampling")
    systems = getattr(sys.modules["XPACDT.Sampling." + method + "Sampling"],
                      "do_" + method + "_sampling")(system, parameters)

    # TODO: Add linear shifts in position or momentum!

    # Save stuff to pickle files.
    for i, s in enumerate(systems):
        trj_folder = os.path.join(name_folder,
                                  'trj_{0:07}'.format(i+folder_shift))
        os.mkdir(trj_folder)
        file_name = parameters.get('system').get('picklefile', 'pickle.dat')
        pickle.dump(s, open(os.path.join(trj_folder, file_name), 'wb'), -1)

# TODO: maybe do some anaylis here!!
