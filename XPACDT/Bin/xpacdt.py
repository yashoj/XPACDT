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

"""XPACDT main executable. Please refer to the general documentation.
"""

import argparse
import git
import inspect
import numpy as np
import os
import pickle
import random
import time

import XPACDT.Dynamics.RealTimePropagation as rt
import XPACDT.Sampling.Sampling as sampling
import XPACDT.Tools.Analysis as analysis
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile


def start():
    """Start any XPACDT calculation."""

    # TODO find possibility to include something with pyinstaller
    # Save version used for later reference
    current_path = os.path.abspath(inspect.getsourcefile(lambda: 0))
    repo = git.Repo(path=current_path, search_parent_directories=True)
    version_file = open('.version', 'w')
    version_file.write("Branch: " + repo.active_branch.name + " \n")
    version_file.write("Commit: " + repo.head.object.hexsha + " \n")
    version_file.close()
    print("Branch: " + repo.active_branch.name)
    print("Commit: " + repo.head.object.hexsha)

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    i_help = "Name of the XPACDT input file. Please refer to the general " \
             "documentation for instructions on how this has to be structured."
    parser.add_argument("-i", "--input", type=str, dest="InputFile",
                        required=True, help=i_help)

    # TODO: Add more command line arguments as fit
    args = parser.parse_args()

    # Get input file
    print("The inputfile '" + args.InputFile + "' is read! \n")
    input_parameters = infile.Inputfile(args.InputFile)

    # Initialize random number generators
    seed = int(input_parameters.get('system').get('seed', time.time()))
    random.seed(seed)
    np.random.seed(seed)

    # Analysis or plotting should be called right away
    job = input_parameters.get("system").get("job")
    if job == "analyze":
        analysis.do_analysis(input_parameters)
        return
    elif job == "plot":
        #    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, False)
        #    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, True)
        raise NotImplementedError("Plotting of PES needs to be implemented!")
        return

    # read from pickle file if exists
    name_folder = input_parameters.get('system').get('folder')
    name_file = input_parameters.get('system').get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)
    if os.path.isfile(path_file):
        print("Reading system state from pickle file!")
        system = pickle.load(open(path_file, 'rb'))
    else:
        system = xSystem.System(input_parameters)

    # Run job
    if job == "sample":
        sampling.sample(system, input_parameters)
    elif job == "propagate":
        rt.propagate(system, input_parameters)
    else:
        raise NotImplementedError("Requested job type not implemented: " + job)

    return


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
