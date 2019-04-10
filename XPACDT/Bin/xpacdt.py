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

"""XPACDT main executable. Please refer to the gneral documentation.
"""

import argparse
import git
from inspect import getsourcefile
import numpy as np
import os
import pickle
import random
# import sys
import time

import XPACDT.Dynamics.RealTimePropagation as rt
import XPACDT.Dynamics.Sampling as sampling
import XPACDT.Tools.Analysis as analysis
import XPACDT.Dynamics.System as xSystem
# import XPACDT.Dynamics.VelocityVerlet as vv
# import XPACDT.Dynamics.WignerSampling as wigner

import XPACDT.Input.Inputfile as infile

# import XPACDT.Interfaces.OneDPolynomial as oneDP
# import XPACDT.Interfaces.InterfaceTemplate as template


def start():
    """Start any XPACDT calculation."""

    # Save version used for later reference
    current_path = os.path.abspath(getsourcefile(lambda: 0))
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
    parameters = infile.Inputfile(**{'filename': args.InputFile})

    # Initialize random number generators
    seed = int(parameters.get('system').get('seed', time.time()))
    random.seed(seed)
    np.random.seed(seed)

    # Analysis needs to be called right away
    job = parameters.get("system").get("job")
    if job == "analyze":
        analysis.do_analysis(parameters)
        return

    # read from pickle file if exists
    name_folder = parameters.get('system').get('folder')
    name_file = parameters.get('system').get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)
    if os.path.isfile(path_file):
        print("Reading system state from pickle file!")
        system = pickle.load(open(path_file, 'rb'))
    else:
        system = xSystem.System(parameters)

    # Run job
    if job == "sample":
        sampling.sample(system, parameters)
    elif job == "propagate":
        rt.propagate(system, parameters)
    else:
        raise NotImplementedError("Requested job type not implemented: " + job)

#    # Example usage for potentials
#    pes = oneDP.OneDPolynomial(**parameters.get("OneDPolynomial"))
#    print(isinstance(pes, oneDP.OneDPolynomial))
#    print(isinstance(pes, template.Interface))
#    print(pes.name)
#    print(pes.energy(np.array([[0.0]])))
#    print(pes.energy(np.array([[0.0]])))
#    print(pes.energy(np.array([[1.0]])))
#    print(pes.minimize(np.array([0.1])))
#
#    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, False)
#    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, True)
#
#    # Example usage for propagator - not meant to be used like this later!!
#    propagator = vv.VelocityVerlet(0.0001, pes, np.array([1.0]), **{'beta': 8.0})
#    r = np.random.rand(4).reshape(1, 4) + 1.0
#    p = np.array([[0.0]*4])
#    print(r, p)
#    outfile = open("/tmp/blah.dat", 'w')
#    for i in range(101):
#        outfile.write(str(i*0.1) + " ")
#        outfile.write(str(np.mean(r[0])) + " ")
#        outfile.write(str(np.mean(p[0])) + " ")
#        outfile.write("\n")
#        r, p = propagator.propagate(r, p, 0.1)
#    outfile.close()
#    pass
    return


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
