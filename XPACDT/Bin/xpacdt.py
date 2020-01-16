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

"""XPACDT main executable. Please refer to the general documentation.
"""

import argparse
import datetime
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
    """ Get absolute path to a folder below the directory, where the currently
    executed xpacdt.py is stored. This works with PyInstaller and also 'normal'
    python. This is used to access files stored in the PyInstaller folder.

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


def print_helpfile(filename):
    """ Output the contents of a help file to stdout.

    Parameters
    ----------
    filename : string
       Name of the file to be printed.
    """

    help_path = resource_path("helptext")

    with open(os.path.join(help_path, filename), 'r') as helpfile:
        for line in helpfile:
            print(line, end='')


def start():
    """Start any XPACDT calculation."""

    # Save version used for later reference; either from git repository or from
    # .version file included by the PyInstaller program
    try:
        current_path = os.path.abspath(inspect.getsourcefile(lambda: 0))
        repo = git.Repo(path=current_path, search_parent_directories=True)
        branch_name = repo.active_branch.name
        hexsha = repo.head.object.hexsha
    except git.exc.NoSuchPathError:
        with open(resource_path("") + '.version', 'r') as input_file:
            branch_name = input_file.readline().split()[1]
            hexsha = input_file.readline().split()[1]

    version_file = open('.version', 'w')
    version_file.write("Branch: " + branch_name + " \n")
    version_file.write("Commit: " + hexsha + " \n")
    version_file.close()
    print("Branch: " + branch_name)
    print("Commit: " + hexsha)
    now = datetime.datetime.now()
    print(now)

    # Parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-h', '--help', nargs='?',
                        type=str, dest="help", const='nothing',
                        help='Prints this help page and additional'
                        ' information for certain keywords:'
                        ' analysis, plot, sampling, propagation.')

    i_help = "Name of the main XPACDT input file. The input file is required" \
             " for any calculation. Please refer to the general " \
             "documentation for instructions on how this has to be structured."
    parser.add_argument("-i", "--input", type=str, dest="InputFile",
                        required=False, help=i_help)

    i_help = "Name of an additional XPACDT input file used for real time " \
             " propagation. Please refer to the general " \
             "documentation for instructions on how this has to be structured."
    parser.add_argument("-p", "--propagation_input", type=str,
                        dest="PropagationInputFile",
                        required=False, help=i_help)

    i_help = "Name of an additional XPACDT input file used for analysis. " \
             "This input file is only used if job = full in the -i inputfile" \
             "or if -p is given. " \
             "Please refer to the general documentation for instructions " \
             " on how this has to be structured."
    parser.add_argument("-a", "--analysis_input", type=str,
                        dest="AnalysisInputFile",
                        required=False, help=i_help)

    args = parser.parse_args()

    if args.help is not None:
        parser.print_help()

        if args.help.lower() == 'analysis':
            print()
            print()
            print("Printing additional help for " + args.help + ":")
            print()
            print_helpfile("analysis.txt")
            operations.position(["-h"], None)
            print()
            operations.momentum(["-h"], None)
            print()
            operations.electronic_state(["-h"], None)
        elif (args.help.lower() == 'plot'
              or args.help.lower() == 'sampling'
              or args.help.lower() == 'propagation'):
            print()
            print()
            print("Printing additional help for " + args.help + ":")
            print()
            print_helpfile(args.help.lower() + ".txt")
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
    start_time = time.time()
    job = input_parameters.get("system").get("job")
    if job == "analyze":
        print("Running Analysis...", end='', flush=True)
        analysis.do_analysis(input_parameters)
        print("...Analysis done in {: .2f} s.".format(time.time() - start_time), flush=True)
        return

    elif job == "plot":
        print("Running plotting...", end='', flush=True)
        pes_name = input_parameters.get("system").get("Interface", None)

        if pes_name is None:
            raise KeyError("\nXPACDT: No potential interface name"
                           " given in input.")
        __import__("XPACDT.Interfaces." + pes_name)
        pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                      pes_name)(**input_parameters.get(pes_name))

        plot_params = input_parameters.get("plot")
        # Parse grid parameters
        dof = [int(i) for i in plot_params.get("dof").split()]
        start = [float(i) for i in plot_params.get("start").split()]
        end = [float(i) for i in plot_params.get("end").split()]
        step = [float(i) for i in plot_params.get("step").split()]

        if len(dof) != len(start):
            raise ValueError("\nXPACDT: Number of degrees of freedom and"
                             " start values for grids differ!"
                             + str(len(dof)) + " != " + str(len(start)))

        if len(dof) != len(end):
            raise ValueError("\nXPACDT: Number of degrees of freedom and"
                             " end values for grids differ!"
                             + str(len(dof)) + " != " + str(len(start)))

        if len(dof) != len(step):
            raise ValueError("\nXPACDT: Number of degrees of freedom and"
                             " step values for grids differ!"
                             + str(len(dof)) + " != " + str(len(start)))

        # Parse electronic parameters
        state = int(plot_params.get("state", "0"))
        picture = 'diabatic' if 'diabatic' in plot_params else 'adiabatic'

        if len(dof) == 1:
            pes.plot_1D(input_parameters.coordinates[:, 0], dof[0],
                        start[0], end[0], step[0],
                        relax=("optimize" in plot_params),
                        internal=("internal" in plot_params),
                        S=state, picture=picture)
        elif len(dof) == 2:
            pes.plot_2D(input_parameters.coordinates, dof[0], dof[1],
                        start, end, step,
                        relax=("optimize" in plot_params),
                        internal=("internal" in plot_params),
                        S=state, picture=picture)

        else:
            raise ValueError("\nXPACDT: Cannot plot PES other than for 1 or"
                             " 2 degrees of freedom.")

        print("...plotting done in {: .2f} s.".format(time.time() - start_time), flush=True)
        return

    # read from pickle file if exists
    name_folder = input_parameters.get('system').get('folder')
    name_file = input_parameters.get('system').get('picklefile', 'pickle.dat')
    path_file = os.path.join(name_folder, name_file)
    if os.path.isfile(path_file):
        print("Reading system state from pickle file!")
        system = pickle.load(open(path_file, 'rb'))
        # Updating input parameters appropriately
        system.parameters = input_parameters
    else:
        system = xSystem.System(input_parameters)

    # Run job
    if job == "full" or args.PropagationInputFile is not None:
        # run sampling first
        print("Running Sampling...", end='', flush=True)
        systems = sampling.sample(system, input_parameters, do_return=True)
        print("...Samping done in {: .2f} s.".format(time.time() - start_time), flush=True)

        # loop and propagate
        print("Running Real time propagation...", end='', flush=True)
        start_time = time.time()
        # Read new input file if given
        if args.PropagationInputFile is not None:
            print("The inputfile '" + args.PropagationInputFile +
                  "' is read! \n")
            input_parameters = infile.Inputfile(args.PropagationInputFile)
        else:
            # Remove thermostat if exists
            input_parameters.pop('thermostat', None)

        for i, system_i in enumerate(systems):
            # create folder
            trj_folder = os.path.join(name_folder, 'trj_{0:07}'.format(i))
            if not os.path.isdir(trj_folder):
                try:
                    os.mkdir(trj_folder)
                except OSError:
                    sys.stderr.write("Creation of trajectory folder "
                                     + trj_folder + " failed!")
                    raise
            # set folder in input parameters
            input_parameters['system']['folder'] = trj_folder

            # Updating input parameters appropriately if second input file is given
            if args.PropagationInputFile is not None:
                system_i.parameters = input_parameters

            # run
            rt.propagate(system_i, input_parameters)
        print("...real time propagation done in {: .2f} s.".format(time.time() - start_time), flush=True)

        # run analsysis
        print("Running analysis...", end='', flush=True)
        start_time = time.time()
        if args.AnalysisInputFile is not None:
            print("The inputfile '" + args.AnalysisInputFile + "' is read! \n")
            input_parameters = infile.Inputfile(args.AnalysisInputFile)
        else:
            input_parameters['system']['folder'] = name_folder

        # Perform actual analysis only if commands are present
        if len(input_parameters.commands) > 0:
            analysis.do_analysis(input_parameters, systems)
            print("...analysis done in {: .2f} s.".format(time.time() - start_time), flush=True)
        else:
            print("...no analysis requested!")

    elif job == "sample":
        print("Running Sampling...", end='', flush=True)
        sampling.sample(system, input_parameters)
        print("...Samping done in {: .2f} s.".format(time.time() - start_time), flush=True)

    elif job == "propagate":
        print("Running Real time propagation...", end='', flush=True)
        rt.propagate(system, input_parameters)
        print("...real time propagation done in {: .2f} s.".format(time.time() - start_time), flush=True)

    else:
        raise NotImplementedError("\nXPACDT: Requested job type not"
                                  " implemented: " + job)

    return


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
