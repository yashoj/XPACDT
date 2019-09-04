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

"""Module to perform analysis on a set of XPACDT.Systems. The most general
cases that can be calculated are:
    Expectation values <A(t)>
    Correlation functions <B(0)A(t)>
    One- and Two-dimensional histograms

In the input file one defines:
    A(t), B(0):  Quantities of interest, e.g., the position of a certain atom,
         a bond length, the charge, etc.
    f(x): A function to be calculated over the quantities obtained from all
         trajectories, i.e., the mean or standard devitaion, a histogram.

The analysis then iterates over all XPACDT.Systems and calculates A(t), B(0)
for each system. Then the function f(x) is evaluated, i.e., the mean of the
quantity is obtained or a histogram of the quantity is obtained. The standard
error of the obtain results is evaluated employing bootstrapping.

Results are printed to file for easy plotting with gnuplot.

Please note that for each quantity one wishes to obtain, an individual
'command'-block has to be defined in the input file. If n operation, i.e. A(t),
B(0), returns more than one value, they all together enter the function f(x)
and are treated as independet in the bootstrapping.
This might be desired behavior for obtaining mean positions of the beads or
obtaining a density plot of the ring polymer, but for most scenarios, this is
not desired. Thus, whenever a command returns more than one value, a
RuntimeWarning is printed for the first system and timestep.

Some basic plotting commands for 2D plots in gnuplot: (TODO: where to actually
put this, etc. Maybe generate a basic gnuplot script along the way?)

#2d plotting:
unset ztics
unset key
unset title
set contour base
set view map
unset surface

do for [a=0:100] {
splot 'command3.dat' index a using 1:2:3 w l lw 3.2
pause 1
}
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

    # Obtain an iterator over all systems
    folder = parameters.get('system').get('folder')
    if systems is None:
        file_name = parameters.get('system').get('picklefile', 'pickle.dat')
        dirs = get_directory_list(folder, file_name)
    else:
        dirs = None
        file_name = None

    n_systems = 0
    # Calculate 'observables' for each system
    for system in get_systems(dirs, file_name, systems):
        # do different stuff for each command
        for key, command in parameters.commands.items():
            if n_systems == 0:
                # Consistency check for operations and print warning if more
                # than one value is returned
                check_command(command, system)

            apply_command(command, system)

        n_systems += 1

    # Apply function for each 'observable'
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

            elif '2dhistogram' in command['value']:
                # TODO other setup possibilities
                values = command['value'].split()

                bins = []
                edges1 = np.linspace(float(values[1]), float(values[2]),
                                     int(values[3])+1)
                bins.append(edges1[:-1] + 0.5*np.diff(edges1))

                edges2 = np.linspace(float(values[4]), float(values[5]),
                                     int(values[6])+1)
                bins.append(edges2[:-1] + 0.5*np.diff(edges2))

                func = (lambda x: np.histogram2d(x[:len(x)//2], x[len(x)//2:],
                                                 bins=(edges1, edges2),
                                                 density=True)[0])

            elif 'histogram' in command['value']:
                # TODO other setup possibilities
                values = command['value'].split()

                bins = []
                edges = np.linspace(float(values[1]), float(values[2]),
                                    int(values[3])+1)
                bins.append(edges[:-1] + 0.5*np.diff(edges))

                func = (lambda x: np.histogram(x, bins=edges, density=True)[0])

            else:
                raise RuntimeError("XPACDT: No function for 'value'"
                                   " defined in the analysis part")

        # The resuts array is reshaped to look like (n_times, n_values).
        #
        # After performing all commands the results array looks is of the
        # follwoing order. It is a list of np.arrays.
        #
        # The np.arrays have the length of the number of returned values from
        # the command.
        #
        # The list has length n_times*(op)*n_systems. (op) is 2 for
        # 2d histograms, 1 otherwise. The list is ordered as follows. For each
        # system, the operation results for all times are given. In case of a
        # 2d histogram the first operation is stored for all times, then the
        # second operation is stored for all times. Then results for the next
        # system are given.
        n_times = len(command['times'])
        reshaped_results = np.swapaxes(np.array(command['results']).
                                       reshape(n_systems, (2 if '2d' in command else 1), n_times, -1),
                                       0, 2).reshape(n_times, -1)
        # bootstrap
        final_data = [bs.bootstrap(data, func, is_2D=('2d' in command))
                      for data in reshaped_results]

        # Generate header
        header = "## Generated for the following command: \n"
        for k, v in command.items():
            if k != 'results' and k != 'times':
                header += "# " + str(k) + " = " + str(v) + " \n"

        # Output data:
        file_output = os.path.join(folder, command.get('filename', key + '.dat'))
        output_data(header, file_output, command['format'], command['times'],
                    bins, final_data, two_d=('2d' in command))


def output_data(header, file_output, form, times, bins, results, two_d=False):
    """ Output the analyzed data in different formats.

        - time: One line per timestep, all values and errors in that line. Or
            for 2D data, one block per timestep (separated by two blank lines)
            and the 2D data gridded (i.e. separated by one blank line for one
                                     dimension)

        - value: One line per value (with error), all times in that line.

        - 2d: Blocked data for 2d plotting. All values for a given time
            separated by a blank line.

        Parameters
        ----------
        header : string
            Header to be printed in the file stating the parameters, etc.
        file_output : string
            Name of the file to write to.
        form : string
            Format in which the data should be written.
            Allowed: time, value, 2d
        times : ndarray of floats
            The times for which the data was created
        bins : list of ndarrays
            If a histogram of the data was generated, bins contains the
            mid-points of each bin. For a 1d histogram, the list only has
            1 element, for a 2d histogram it has two.
        results : ndarray
            The results to be written to file.
        two_d : bool, optional, default False
            Whether a 2d histogram was produced.
    """

    # Output format: one line or block per timestep
    if form == 'time':

        # If 2d histogram, we output to a blocked format used in gnuplot
        # Each timestep is separated by two blank lines
        # Within each timestep the data is blocked such that an increase
        # in the first coordinate is separated by one blank line
        # standard errors are ignored here!
        if two_d:
            outfile = open(file_output, 'w')
            outfile.write(header)
            dd = np.c_[times, np.array(results)[:, 0, :]]
            for data in dd:
                outfile.write("# time = " + str(data[0]) + " \n")
                for i, b1 in enumerate(bins[0]):
                    for j, b2 in enumerate(bins[1]):
                        outfile.write(str(b1) + " " + str(b2) + " " +
                                      str(data[1+i*len(bins[1])+j]) + " \n")
                    outfile.write("\n")
                outfile.write("\n \n")

        # Regular output, just one line per timestep
        else:
            number_times = len(times)
            np.savetxt(file_output,
                       np.c_[times, np.array(results).reshape((number_times, -1))],
                       header=header)

    # Output format: One line per value (e.g. per bin in histogram)
    # each column represents one timestep
    elif form == 'value':
        # add time values in header for later reference
        for t in times:
            header += str(t) + "\t" + str(t) + "\t"
        header += " \n"

        number_values = len(results[0][0])
        if bins is None:
            bins = np.zeros(number_values)
        np.savetxt(file_output,
                   np.c_[bins[0], np.array(results).reshape((-1, number_values)).T],
                   header=header)

    # Output format: For 2D plots of histograms vs. time
    elif form == '2d':
        outfile = open(file_output, 'w')
        outfile.write(header)

        number_times = len(times)
        dd = np.c_[times, np.array(results).reshape((number_times, -1))]
        for data in dd:
            for i, b in enumerate(bins[0]):
                outfile.write(str(data[0]) + " " + str(b) + " " +
                              str(data[1+i]) + " \n")
            outfile.write("\n")

    else:
        raise RuntimeError("XPACDT: No or incorrect output format given: " + form)


def check_command(command, system):
    """ Check for return size of a command and print warning if longer than 1,
    as this might often be unwanted, unless all returned values are of the same
    type and independent. Then the warning can be ignored.

    Parameters
    ----------
    command : dict
        The definition of the command to be evaluated as given in the input
        file.
    system : XPACDT.System
        The read in system containing its own log.
    """
    # Time zero operation for correlation functions, etc.
    value_0 = 1.0
    if 'op0' in command:
        value_0 = apply_operation(command['op0'], system.log[0])

    try:
        if len(value_0 * apply_operation(command['op'], system.log[0])) > 1:
            raise RuntimeWarning("XPACDT: The operation in analysis returns"
                                 "more than one value. Please check if this"
                                 "is the intended behavior. Please note that"
                                 "all returned values will be used in the"
                                 "subsequent operations (e.g., mean or"
                                 "historgram) together and that they are"
                                 "treated as independent in the"
                                 "bootstrapping. op0:" + command['op0'] +
                                 "; op:" + command['op'])
    except ValueError as e:
        raise type(e)(str(e) + "\nXPACDT: If 'operands could not be broadcast"
                               " together it probably is due to incompatible"
                               " array sizes returned by the 'op0' and 'op'"
                               " operationsgiven. Please check!")


def apply_command(command, system):
    """ Apply a given command to a given system. The results are stored in
    command['results'] and the times for which the system is evaluated
    are stored in command['times'].

    Parameters
    ----------
    command : dict
        The definition of the command to be evaluated as given in the input
        file.
    system : XPACDT.System
        The read in system containing its own log.
    """

    steps_used = [int(x) for x in command.get('step', '').split()]

    # Time zero operation for correlation functions, etc.
    value_0 = 1.0
    if 'op0' in command:
        value_0 = apply_operation(command['op0'], system.log[0])

    try:
        # Iterate over all times and calculate the full command.
        command['results'].extend([value_0 * apply_operation(command['op'], log_nuclei)
                                   for i, log_nuclei in enumerate(system.log)
                                   if _use_time(i, steps_used)])
    except ValueError as e:
        raise type(e)(str(e) + "\nXPACDT: If 'operands could not be broadcast"
                               " together it probably is due to incompatible"
                               " array sizes returned by the 'op0' and 'op'"
                               " operationsgiven. Please check!")

    # For a 2d histogram another 'obeservable' needs to be computed
    if '2d' in command:
        value_0 = 1.0
        if '2op0' in command:
            value_0 = apply_operation(command['2op0'], system.log[0])

        try:
            # Iterate over all times and calculate the full command.
            command['results'].extend([value_0 * apply_operation(command['2op'], log_nuclei)
                                       for i, log_nuclei in enumerate(system.log)
                                       if _use_time(i, steps_used)])
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: If 'operands could not be"
                                   " broadcast together it probably is due to"
                                   " incompatible array sizes returned by the"
                                   " '2op0' and '2op' operationsgiven. Please"
                                   " check!")

    command['times'] = [log.time for i, log in enumerate(system.log)
                        if _use_time(i, steps_used)]

    return


def _use_time(i, steps_used):
    """ Wrapper to check if a certain command is supposed to be evaluated for a
    given timestep.

    Parameters
    ----------
    i : integer
        Index of the timestep in question.
    steps_used : list of integer
        Empty list if all timesteps should be used. Else a list of integers
        identifying the timesteps to be used.

    Returns
    -------
    bool :
        True is steps_used is empty or if i is present in steps_used. Else
        False is returned.
    """

    if steps_used == []:
        return True
    else:
        return (i in steps_used)


def apply_operation(operation, log_nuclei):
    """ Apply a given sequence of operation to a given log_nuclei log given.

    Parameters
    ----------
    operation : string
        The string defining the sequence of operations. Each operation starts
        with a '+' and an identifyer (e.g., +position, +velocity, ...).
        Arguments specifying the operation are given after that.
    log_nuclei : XPACDT.System.Nuclei object from the log to perform
                 operations on.

    Returns
    -------
    value :
        The value resulting from the operations.
    """

    if '+' not in operation:
            raise RuntimeError("XPACDT: No operation given, instead: " + operation)

    value = 1.0
    # The split has '' as a first results on something like '+pos'.split('+'),
    # and we need to ignore that one.
    for op_string in operation.split('+')[1:]:
        ops = op_string.split()

        # match the different operations here.
        # TODO: more automatic
        if ops[0] == 'id' or ops[0] == 'identity':
            pass
        elif ops[0] == 'pos' or ops[0] == 'position':
            value *= op.position(ops[1:], log_nuclei)
        elif ops[0] == 'mom' or ops[0] == 'momentum':
            value *= op.momentum(ops[1:], log_nuclei)
        elif ops[0] == 'vel' or ops[0] == 'velocity':
            value *= op.momentum(ops[1:] + ['-v'], log_nuclei)
        else:
            raise RuntimeError("XPACDT: The given operation is not"
                               "implemented. " + " ".join(ops))

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
    """Obtain a generator over all systems to sweep through them in the
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
    Generator over all sytems.
    """

    if dirs is not None:
        return (pickle.load(open(os.path.join(folder_name, file_name), 'rb'))
                for folder_name in dirs)
    elif systems is not None:
        return (system for system in systems)
    else:
        raise RuntimeError("Neither dirs nor systems given!")
