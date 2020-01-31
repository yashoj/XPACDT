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

""" Module to perform operations on a XPACDT.System.Nuclei object, e.g.,
obtained from the system log, to obtain some observables and perform
analysis. """

import numpy as np
import argparse


def position(arguments, log_nuclei):
    """Performs operations related to positions. If no options given it
    will raise an error.

    Valid options are as follows:

    -1 <a> given: Position value of a given degree of freedom, e.g., -1 0,
                   gives the first position, or -1 0,3,7 gives the first,
                   fourth and seventh position. Alternatively, also the
                   center of mass position can be obtained by giving m and a
                   comma separated list of degrees of freedom.
    -2 <b> given:  Like 1. If both given, then the distance between them is
                   used.
    -p <a> given: if a single value is calculated (i.e. a distance or single
                  position) this option projects it onto a certain range.
                  Valid are fully bound ranges (A,<,B), or below a value (<,A)
                  or above a value (>,A). If within the given value the
                  function returns 1.0, else the given pValue.
    -r given: Use ring-polymer bead positions.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the position command. See above.
    log_nuclei: XPACDT.System.Nuclei object from the log to perform
                operations on.

    Returns
    -------
    (n_values) ndarray of floats
        Values obtained from the position operation. The length depends on
        the operation to be performed. If, e.g., all bead positions of a
        single degree of freedom is requested, n_values will be n_beads of
        that degree of freedom.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(usage="Options for +pos", add_help=False)

    parser.add_argument('-h', '--help',
                        dest='help',
                        action='store_true',
                        default=False,
                        help='Prints this help page.')

    parser.add_argument('-1', '--x1',
                        dest='x1',
                        type=str,
                        default=None,
                        help='Obtain the position of a list of degrees of freedom (given as comma-separated list) or a center of mass position; given as m followed by the list of degrees of freedom included. Please note that the numbering starts a 0.')

    parser.add_argument('-2', '--x2',
                        dest='x2',
                        type=str,
                        default=None,
                        help='If given, the distance between the two given (x1 and x2) sites should be calculated.')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=str,
                        default=None,
                        help='Take the distance or coordinate value given and project onto a certain range. Valid are fully bound ranges (A,<,B), or below a value (<,A) or above a value (>,A).')

    parser.add_argument('-r', '--rpmd',
                        dest='rpmd',
                        action='store_true',
                        default=False,
                        help='Use beads instead of centroids.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to position operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    # get coordinate values under consideration here!
    current_value = log_nuclei.get_selected_quantities(opts.x1, 'x', opts.rpmd)
    if opts.x2 is not None:
        coordinate_2 = log_nuclei.get_selected_quantities(opts.x2, 'x', opts.rpmd)
        # Also calculated per beads
        try:
            current_value = np.linalg.norm(current_value - coordinate_2, axis=0)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Cannot calculate distance between given sites. Maybe different number of dofs per site.")

    # if we want to project the distance/position onto a certain interval
    if opts.proj is not None:
        current_value = _projection(opts.proj, current_value)

    return np.array(current_value).flatten()


def momentum(arguments, log_nuclei):
    """Performs operations related to momenta. If no options given it
    will raise an error.

    Valid options are as follows:

    -v given: Use velocities instead of momenta.
    -1 <a> given: momentum value of a given degree of freedom, e.g., -1 0,
                   gives the first momentum, or -1 0,3,7 gives the first,
                   fourth and seventh momentum. Alternatively, also the
                   center of mass momentum can be obtained by giving m and a
                   comma separated list of degrees of freedom.
    -2 <b> given: Like 1. If both given, then the relative momentum between
                   them is used.
    -p <a> given: if a single value is calculated (i.e. a relative or single
                  momentum) this option projects it onto a certain range.
                  Valid are fully bound ranges (A,<,B), or below a value (<,A)
                  or above a value (>,A). If within the given value the
                  function returns 1.0, else the given pValue.
    -r given: Use ring-polymer bead momenta.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the position command. See above.
    log_nuclei: XPACDT.System.Nuclei object from the log to perform
                operations on.

    Returns
    -------
    (n_values) ndarray of floats
        Values obtained from the momentum operation. The length depends on
        the operation to be performed. If, e.g., all bead momenta of a
        single degree of freedom is requested, n_values will be n_beads of
        that degree of freedom.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(usage="Options for +mom", add_help=False)

    parser.add_argument('-h', '--help',
                        dest='help',
                        action='store_true',
                        default=False,
                        help='Prints this help page.')

    parser.add_argument('-v', '--velocities',
                        dest='vel',
                        action='store_true',
                        default=False,
                        help='Use velocities instead of momenta.')

    parser.add_argument('-1', '--x1',
                        dest='x1',
                        type=str,
                        default=None,
                        help='Obtain the momentum of a list of degrees of freedom (given as comma-separated list) or a center of mass; given as m followed by the list of degrees of freedom included. Please note that the numbering starts a 0.')

    parser.add_argument('-2', '--x2',
                        dest='x2',
                        type=str,
                        default=None,
                        help='If given, the relative momentum between the two given (x1 and x2) sites should be calculated.')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=str,
                        default=None,
                        help='Take the momentum value given and project onto a certain range. Valid are fully bound ranges (A,<,B), or below a value (<,A) or above a value (>,A).')

    parser.add_argument('-r', '--rpmd',
                        dest='rpmd',
                        action='store_true',
                        default=False,
                        help='Use beads instead of centroids.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to momentum operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    quantity = 'v' if opts.vel else 'p'

    # get coordinate values under consideration here!
    current_value = log_nuclei.get_selected_quantities(opts.x1, quantity, opts.rpmd)
    if opts.x2 is not None:
        raise NotImplementedError("Implement relative momentum calculations, etc.")
#        coordinate_2 = log_nuclei.parse_coordinate(opts.x2, quantity, opts.rpmd)
#        try:
#            current_value = np.linalg.norm(current_value - coordinate_2, axis=0)
#        except ValueError as e:
#            raise type(e)(str(e) + "\nXPACDT: Cannot calculate relative momentum between given sites. Maybe different number of dofs per site.")

    # if we want to project the distance/position onto a certain interval
    if opts.proj is not None:
        current_value = _projection(opts.proj, current_value)

    return np.array(current_value).flatten()


def _projection(options, values):
    """Check if the given values are below, above or within some limits.

    Parameters
    ----------
    options : string
        Defines the test to be performed for each value. Valid are:
            - fully bound ranges: A,<,B
            - below a value: <,A
            - above a value: >,A
    values : float or ndarray of floats
        The values to be checked.

    Returns
    -------
    values.shape ndarray of floats
        1.0 if given value is within the range, 0.0 otherwise.
    """

    limits = options.split(',')
    if len(limits) < 2 or len(limits) > 3:
        raise RuntimeError("Error parsing projection: " + options)

    try:
        # case: above a value; > A
        if limits[0] == '>':
            values = values > float(limits[1])

        # case: below a value; < A
        elif limits[0] == '<':
            values = values < float(limits[1])

        # case: between values < A <
        elif limits[1] == '<':
            values = np.logical_and(values > float(limits[0]), values < float(limits[2]))

        else:
            raise RuntimeError("Error parsing projection: " + options)

    except ValueError as e:
        raise type(e)(str(e) + "\nXPACDT: Cannot convert limits in projection: " + options)

    if type(values) == bool:
        values = float(values)
    elif type(values) == np.ndarray:
        values = values.astype(float)

    return values


def electronic_state(arguments, log_nuclei):
    """Performs operations related to electronic state. If no options are
    given, then it will raise an error.

    Valid options are as follows:

    -b <basis> given: Electronic basis to be used. Can be "adiabatic" or
                      "diabatic". Default: "adiabatic".
    -p <a> given: State to be projected onto in the basis given by 'basis'.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the state command. See above.
    log_nuclei: XPACDT.System.Nuclei object from the log to perform
                operations on.

    Returns
    -------
    (1) ndarray of float
        Value obtained from state operation.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(usage="Options for +state", add_help=False)

    parser.add_argument('-h', '--help',
                        dest='help',
                        action='store_true',
                        default=False,
                        help='Prints this help page.')

    parser.add_argument('-b', '--basis',
                        dest='basis',
                        type=str,
                        default='adiabatic',
                        choices=['adiabatic', 'diabatic'],
                        required=False,
                        help='Basis to be used. Possible "adiabatic" or "diabatic". Default: "adiabatic".')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=int,
                        default=None,
                        help='State to be projected onto.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to position operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    if (opts.proj >= log_nuclei.electrons.pes.n_states):
        raise ValueError("\nXPACDT: State to be projected onto is greater than"
                         " the number of states. Note: State count starts from"
                         " 0. Given state to project is: " + str(opts.proj))

    current_value = log_nuclei.electrons.get_population(opts.proj, opts.basis)

    return np.array(current_value)
