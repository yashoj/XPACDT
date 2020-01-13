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
import sys

import XPACDT.Dynamics.RealTimePropagation as rt
import XPACDT.Sampling.Sampling as sampling
import XPACDT.Tools.Analysis as analysis
import XPACDT.Tools.Operations as operations
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile


def resource_path(relativePath):
    """ Get absolute path to a folder below the directory, where the currently executed 
    xpacdt.py is stored. This works with PyInstaller and also 'normal' python. This is 
    used to access files stored in the PyInstaller folder.

    Parameters
    ----------
    relativePath: string
        Relative path of the required file to the file currently executed.

    Returns
    -------
    Absolute path to the folder below the current working folder.
    """
    base_path = getattr(sys, '_MEIPASS',
                        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relativePath)


def start():
    """Start any XPACDT calculation."""

    # Save version used for later reference; either from git repository or from
    # .version file included by the PyInstaller program
    try:
        current_path = os.path.abspath(inspect.getsourcefile(lambda: 0))
        repo = git.Repo(path=current_path, search_parent_directories=True)
        branch_name = repo.active_branch.name
        hexsha = repo.head.object.hexsha
    except:  # TODO: better specific errors!
        with open(resource_path("") + '.version', 'r') as input_file:
            branch_name = input_file.readline().split()[1]
            hexsha = input_file.readline().split()[1]

    version_file = open('.version', 'w')
    version_file.write("Branch: " + branch_name + " \n")
    version_file.write("Commit: " + hexsha + " \n")
    version_file.close()
    print("Branch: " + branch_name)
    print("Commit: " + hexsha)

    # Parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-h', '--help', nargs='?',
                        type=str, dest="help", const='nothing',
                        help='Prints this help page and additional information for certain keywords: Analysis, TODO_MORE.')

    i_help = "Name of the XPACDT input file. Please refer to the general " \
             "documentation for instructions on how this has to be structured."
    parser.add_argument("-i", "--input", type=str, dest="InputFile",
                        required=False, help=i_help)

    # TODO: Add more command line arguments as fit
    args = parser.parse_args()

    if args.help is not None:
        parser.print_help()
        if args.help == 'Analysis':
            print()
            print()
            print("Printing additional help for " + args.help + ":")
            print()
            print("Module to perform analysis on a set of XPACDT.Systems. The most general cases that can be calculated are: \n \t Expectation values <A(t)>, \n \t Correlation functions <B(0)A(t)>, \n \t One- and Two-dimensional histograms \n\nIn the input file one defines: \n\t A(t), B(0):  Quantities of interest, e.g., the position of a certain atom, a bond length, the charge, etc. The quantities can be obtained as multiplications of more basic quantities.\n\t f(x): A function to be calculated over the quantities obtained from all trajectories, i.e., the mean or standard devitaion, a histogram. \n\nThe analysis then iterates over all XPACDT.Systems and calculates A(t), B(0) for each system. Then the function f(x) is evaluated, i.e., the mean of the quantity is obtained or a histogram of the quantity is obtained. The standard error of the obtain results is evaluated employing bootstrapping. \n\nResults are printed to file for easy plotting with gnuplot. \n\nPlease note that for each quantity one wishes to obtain, an individual 'command'-block has to be defined in the input file. If n operation, i.e. A(t), B(0), returns more than one value, they all together enter the function f(x) and are treated as independet in the bootstrapping. This might be desired behavior for obtaining mean positions of the beads or obtaining a density plot of the ring polymer, but for most scenarios, this is not desired. Thus, whenever a command returns more than one value, a RuntimeWarning is printed for the first system and timestep.\n\n\n")
            print("An Example input block for a position-position correlation function looks like:\n\n$commandCxx\nop0 = +pos\nop = +pos\nformat = time\nvalue = mean\n\n$end \n\n")
            print("An Example input block for a histogram of the positions of a one-d system looks like:\n\n$commandPos\nop = +pos\nvalue = histogram -3.0 3.0 10\nformat = value\n$end \n\n")
            print("An Example input block for a histogram of the velocities for all positions greater than 0 of a one-d system looks like:\n\n$commandPos\nop = +vel +pos -p<0\nvalue = histogram -3.0 3.0 10\nformat = value\n$end \n\n")
            print("Please refer to the example input files for more options!\n\n")
            print("Possible operations for A and B are: \n")
            operations.position(["-h"], None)
            print()
            operations.momentum(["-h"], None)
        elif args.help == 'nothing':
            pass
        else:
            print("Incorect keyword given to -h :" + args.help)
        return

    # Get input file
    if args.InputFile is None:
        print("Input file required!")
        return
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
        pes_name = input_parameters.get("system").get("Interface", None)
        assert (pes_name != None), "No potential interface name given in input."
        __import__("XPACDT.Interfaces." + pes_name)
        pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                      pes_name)(**input_parameters.get(pes_name))

        dof = [int(i) for i in input_parameters.get("plot").get("dof").split()]
        start = [float(i) for i in input_parameters.get("plot").get("start").split()]
        end = [float(i) for i in input_parameters.get("plot").get("end").split()]
        step = [float(i) for i in input_parameters.get("plot").get("step").split()]

        assert(len(dof) == len(start)), "Number of degrees of freedom and start values for grids differ!" + str(len(dof)) + " != " + str(len(start))
        assert(len(dof) == len(end)), "Number of degrees of freedom and end values for grids differ!" + str(len(dof)) + " != " + str(len(end))
        assert(len(dof) == len(step)), "Number of degrees of freedom and step values for grids differ!" + str(len(dof)) + " != " + str(len(step))

        if len(dof) == 1:
            pes.plot_1D(input_parameters.coordinates[:, 0], dof[0],
                        start[0], end[0], step[0],
                        relax=("optimize" in input_parameters.get("plot")),
                        internal=("internal" in input_parameters.get("plot")))
        elif len(dof) == 2:
            pes.plot_2D(input_parameters.coordinates, dof[0], dof[1],
                        start, end, step,
                        relax=("optimize" in input_parameters.get("plot")),
                        internal=("internal" in input_parameters.get("plot")))

        else:
            raise RuntimeError("Cannot plot PES other than for 1 or 2 degrees of freedom.")
        return

    # read from pickle file if exists
    name_folder = input_parameters.get('system').get('folder')
    name_file = input_parameters.get('system').get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)
    if os.path.isfile(path_file):
        print("Reading system state from pickle file!")
        system = pickle.load(open(path_file, 'rb'))
        # TODO: update new input parameters
    else:
        assert(input_parameters.momenta is not None), \
            "Momenta not provided in input file."
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
