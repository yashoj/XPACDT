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

import os
import pickle
import numpy as np

import XPACDT.Tools.Bootstrap as bs
import XPACDT.Tools.Operations as op

"""Module to perform analysis on a set of XPACDT.Systems
TODO: A lot of basic docu here.
# this is all horrible!! Just testing some basic C_xx for simple systems
"""


def do_analysis(parameters, systems=None):
    """Run a full analysis. TODO: details here

    Parameters
    ----------
    parameters : XPACDT.Infputfile
        Input file defining all parameters for the analysis.
    systems : list of XPACDT.Systems, optional, default: None
        A list of systems to perform the analysis on. If not given, then the
        systems are read from file.
    """

    folder = parameters.get('system').get('folder')
    if systems is None:
        file_name = parameters.get('system').get('picklefile', 'pickle.dat')
        dirs = get_directory_list(folder, file_name)
    else:
        dirs = None
        file_name = None

    t_old = None
    for system in get_systems(dirs, file_name, systems):

        # do different stuff for each command
        for key, command in parameters.commands.items():
            times = apply_command(command, system)

            # time consistency check
            if t_old is not None and not times == t_old:
                # TODO more info on which traj - how to do that?
                raise RuntimeError("The time in the trajectories is not "
                                   "aligned for command: " + key)
            t_old = times.copy()

    # todo: structure better for results
    for key, command in parameters.commands.items():
        # Set up function to combine data...
        func = np.mean
        bins = None
        if 'value' in command:
            if command['value'] == 'mean':
                func = np.mean
            elif command['value'] == 'std':
                func = np.std
            elif 'percentile' in command['value']:
                pe = float(command['value'].split()[1])
                func = (lambda x: np.nanpercentile(x, pe))
            elif 'histogram' in command['value']:
                # TODO other setup possibilities
                values = command['value'].split()
                edges = np.linspace(float(values[1]), float(values[2]),
                                    int(values[3])+1)
                bins = edges[:-1] + 0.5*np.diff(edges)

                func = (lambda x: np.histogram(x, bins=edges, density=True)[0])

        # bootstrap
        final_data = [bs.bootstrap(data, func)
                      for data in np.array(command['results']).T]

        # TODO: Output a lot of info in comments!!
        # Output in different formats:
        # time: One line per timestep, all values and errors in that line
        # value: One line per value (with error), all times in that line
        file_output = os.path.join(folder, key + '.dat')
        if command['format'] == 'time':
            number_times = len(times)
            np.savetxt(file_output, np.c_[times, np.array(final_data).
                                          reshape((number_times, -1))])
        elif command['format'] == 'value':
            # TODO: add values in front! (instead of time)
            number_values = len(final_data[0][0])
            if bins is None:
                bins = np.zeros(number_values)
            np.savetxt(file_output, np.c_[bins, np.array(final_data).
                                          reshape((-1, number_values)).T])


def apply_command(command, system):
    """ Apply a given command to a given system.

    Parameters
    ----------
    command : dict
        The definition of the command to evaluated as given in the input file.
    system : XPACDT.System
        The read in system containing its own log.

    Returns
    -------
    list of floats
        The times for which the command was evaluated, i.e., the times in the
        log of the system.
    """

    # Time zero operation for correlation functions, etc.
    value_0 = 1.0
    if 'op0' in command:
        value_0 = apply_operation(command['op0'], system._log[0])

    # Iterate over all times and calculate the full command.
    # TODO: Implement to only do a subpart of the times. I.e. first, last, ...
    command['results'].append([value_0 * apply_operation(command['op'], log)
                               for log in system._log])

    return [log['time'] for log in system._log]


def apply_operation(operation, system):
    """ Apply a given sequence of operation to a given system log given.

    Parameters
    ----------
    operation : string
        The string defining the sequence of operations. Each operation starts
        with a '+' and an identifyer (e.g., +position, +velocity, ...).
        Arguments specifying the operation are given after that.
    system : dict containing the system information (i.e., from the log).

    Returns
    -------
    value :
        The value resulting from the operations.
    """

    value = 1.0
    # The split has '' as a first results on something like '+pos'.split('+'),
    # and we need to ignore that one.
    for op_string in operation.split('+')[1:]:
        ops = op_string.split()

        # match the different operations here.
        if ops[0] == 'id' or ops[0] == 'idendity':
            value *= 1.0
        elif ops[0] == 'pos' or ops[0] == 'position':
            value *= op.position(ops[1:], system)
        else:
            raise RuntimeError("XPACDT: The given operation is not"
                               "implemented. " + ops)

    return value


def get_directory_list(folder='./', file_name=None):
    """ Get trj_ subfolders in a given folder. If a file name is given, only
    trj_ subfolders are returned that contain a file with that name. The
    returned list is sorted.

    Parameters
    ----------
    folder : string, optional, default: './'
        Folder to search for trj_ subfolders.
    file_name : string, optional, default: None
        If given, only trj_ subfolders are returned that contain a file with
        that name.

    Returns
    -------
    dirs : list of string
        Sorted list of trj_ subfolders.
    """
    allEntries = os.listdir(folder)
    dirs = []
    for entry in allEntries:
        path = os.path.join(folder, entry)
        if entry[0:4] == 'trj_' and os.path.isdir(path):
            if file_name is None or os.path.isfile(os.path.join(path,
                                                                file_name)):
                dirs.append(path)
    dirs.sort()
    return dirs


def get_systems(dirs, file_name, systems):
    """Obtain an iterator over all systems to sweep through them in the
    analysis.
    The systems are either given as a list of systems or read from pickle
    files in the given list of folders.

    Parameters
    ----------
    dirs : list of strings
        Directories to read the pickle files from.
    file_name : String
        Name of the pickle files to be read.
    systems: list of XPACDT.System
        A list of systems to perform the analysis on. If not given, then the
        systems are read from file.

    Returns
    -------
    Iterator over all sytems.
    """

    if dirs is not None:
        return (pickle.load(open(os.path.join(folder_name, file_name), 'rb'))
                for folder_name in dirs)
    elif systems is not None:
        return (system for system in systems)
    else:
        raise RuntimeError("Neither dirs nor systems given!")
