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
analysis. This is done through the class 'Operations' defined here, which
stores the operations to be performed and calls appropriate operation
functions, also defined in this module, to perform it on the
XPACDT.System.Nuclei object.

Help on the options for these operation functions can be obtained through the
general help of xpacdt.py with additional key word 'analysis'.
All operation functions in this module should return a one dimensional
numpy array with independent values, the length of which depends upon the
operation performed."""

import numpy as np
import argparse
import sys


class Operations(object):
    """
    This class represents an operation to be applied on XPACDT.System.Nuclei
    object. This operation can be combination of different possible
    sub-operations, which gives the final result as a product of these
    sub-operations.

    Parameters:
    -----------
    operation_string : string
        The string defining the sequence of operations. Each operation starts
        with a '+' and an identifyer (e.g., +position, +velocity, ...).
        Arguments specifying the operation are given after that.
    print_help : bool, default False
        Whether to just print help for each possible operation.

    Attributes:
    -----------
    operations_list
    """

    def __init__(self, operation_string, print_help=False):

        # Printing help for each operation. Any new operation added to this
        # class should be added here too.
        if print_help:
            all_operations = ["position", "momentum", "electronic_state",
                              "energy", "electronic_dm"]

            for op in all_operations:
                getattr(sys.modules[__name__], "_arguments_" + op)(["-h"])
                print("\n")
            return

        # List of operations to be performed.
        self.__operations_list = []

        if ('+' not in operation_string) and (not print_help):
            raise RuntimeError("XPACDT: No operation given, instead: "
                               + operation_string)

        # The split has '' as a first results on something like
        # '+pos'.split('+'), and we need to ignore that one.
        for i, op_string in enumerate(operation_string.split('+')[1:]):
            ops = op_string.split()

            # Match the different operations here and store the argument parsed
            # object in 'operations_list'.
            if ops[0] == 'pos' or ops[0] == 'position':
                self.__operations_list.append(('_position',
                                               _arguments_position(ops[1:])))
            elif ops[0] == 'mom' or ops[0] == 'momentum':
                self.__operations_list.append(('_momentum',
                                               _arguments_momentum(ops[1:])))
            elif ops[0] == 'vel' or ops[0] == 'velocity':
                self.__operations_list.append(('_momentum',
                                               _arguments_momentum(ops[1:] + ['-v'])))
            elif ops[0] == 'state':
                self.__operations_list.append(('_electronic_state',
                                               _arguments_electronic_state(ops[1:])))
            elif ops[0] == 'energy':
                self.__operations_list.append(('_energy',
                                               _arguments_energy(ops[1:])))
            elif ops[0] == 'dm' or ops[0] == 'electronic_dm':
                self.__operations_list.append(('_electronic_dm',
                                               _arguments_electronic_dm(ops[1:])))
            else:
                raise RuntimeError("\nXPACDT: The given operation is not"
                                   "implemented. " + " ".join(ops))

    @property
    def operations_list(self):
        """ list: Contains all operations to be performed. Each element is a
        tuple with operation function name and their respective
        argparse.Namespace objects.
        """
        return self.__operations_list

    def apply_operation(self, log_nuclei):
        """ Apply requested operations in sequence and get the final value as
        a product of these individual operation results.

        Parameters
        ----------
        log_nuclei : XPACDT.System.Nuclei object
            Nuclei object from the log to perform operations on.

        Returns
        -------
        value : (n_values) ndarray of floats
            The value resulting from the operations. The length 'n_values'
            depends on the operation performed.
        """
        value = 1.0

        for op in self.operations_list:
            value *= getattr(sys.modules[__name__], op[0])(op[1], log_nuclei)

        return np.array(value).flatten()


def _arguments_position(arguments):
    """ Parses the options for position command. If no options given it will
    raise an error.

    Valid options are as follows:

    -1 <a> given: Position value of a given degree of freedom, e.g., -1 0,
                  gives the first position, or -1 0,3,7 gives the first,
                  fourth and seventh position. Alternatively, also the
                  center of mass position can be obtained by giving m and a
                  comma separated list of degrees of freedom.
    -2 <b> given: Like 1. If both given, then the distance between them is
                  used.
    -p <a> given: if a single value is calculated (i.e. a distance or single
                  position) this option projects it onto a certain range.
                  Valid are fully bound ranges (A,<,B), or below a value (<,A)
                  or above a value (>,A). If within the given value the
                  function returns 1.0, else the given pValue.
    -r given: Use ring-polymer bead positions.
    -g given: Get radius of gyration of ring polymer.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the position command. See above.

    Returns
    -------
    opts: argparse.Namespace object
        Options for position operation.
    """
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
                        help='Obtain the position of a list of degrees of'
                             ' freedom (given as comma-separated list) or a'
                             ' center of mass position; given as m followed by'
                             ' the list of degrees of freedom included. Please'
                             ' note that the numbering starts a 0.')

    parser.add_argument('-2', '--x2',
                        dest='x2',
                        type=str,
                        default=None,
                        help='If given, the distance between the two given'
                             ' (x1 and x2) sites should be calculated.')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=str,
                        default=None,
                        help='Take the distance or coordinate value given and'
                             ' project onto a certain range. Valid are fully'
                             ' bound ranges (A,<,B), or below a value (<,A) or'
                             ' above a value (>,A).')

    parser.add_argument('-r', '--rpmd',
                        dest='rpmd',
                        action='store_true',
                        default=False,
                        help='Use beads instead of centroids.')

    parser.add_argument('-g', '--gyration',
                        dest='gyration',
                        action='store_true',
                        default=False,
                        help='Get radius of gyration of ring polymer.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to position operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    return opts


def _position(opts, log_nuclei):
    """ Performs operations related to positions on `log_nuclei` using input
    options from `opts`.

    Parameters
    ----------
    opts: argparse.Namespace object
        Options for position operation.
    log_nuclei: XPACDT.System.Nuclei object
        Nuclei object from the log to perform operations on.

    Returns
    -------
    (n_values) ndarray of floats
        Values obtained from the position operation. The length depends on
        the operation to be performed. If, e.g., all bead positions of a
        single degree of freedom is requested, n_values will be n_beads of
        that degree of freedom.
    """
    if opts.gyration:
        current_value = log_nuclei.radius_of_gyration

    else:
        # get coordinate values under consideration here!
        current_value = log_nuclei.get_selected_quantities(opts.x1, 'x',
                                                           opts.rpmd)
        if opts.x2 is not None:
            coordinate_2 = log_nuclei.get_selected_quantities(opts.x2, 'x',
                                                              opts.rpmd)
            # Also calculated per beads
            try:
                current_value = np.linalg.norm(current_value - coordinate_2,
                                               axis=0)
            except ValueError as e:
                raise type(e)(str(e) + "\nXPACDT: Cannot calculate distance between"
                              " given sites. Maybe different number of dofs per site.")

        # if we want to project the distance/position onto a certain interval
        if opts.proj is not None:
            current_value = _projection(opts.proj, current_value)

    return np.array(current_value).flatten()


def _arguments_momentum(arguments):
    """ Parses the options for momentum command. If no options given it will
    raise an error.

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
        Command line type options given to the momentum command. See above.

    Returns
    -------
    opts: argparse.Namespace object
        Options for momentum operation.
    """
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
                        help='Obtain the momentum of a list of degrees of'
                             ' freedom (given as comma-separated list) or a'
                             ' center of mass; given as m followed by the list'
                             ' of degrees of freedom included. Please note'
                             ' that the numbering starts a 0.')

    parser.add_argument('-2', '--x2',
                        dest='x2',
                        type=str,
                        default=None,
                        help='If given, the relative momentum between the two'
                             ' given (x1 and x2) sites should be calculated.')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=str,
                        default=None,
                        help='Take the momentum value given and project onto a'
                             ' certain range. Valid are fully bound'
                             ' ranges (A,<,B), or below a value (<,A) or'
                             ' above a value (>,A).')

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

    return opts


def _momentum(opts, log_nuclei):
    """Performs operations related to momenta on `log_nuclei` using input
    options from `opts`.

    Parameters
    ----------
    opts: argparse.Namespace object
        Options for momentum operation.
    log_nuclei: XPACDT.System.Nuclei object
        Nuclei object from the log to perform operations on.

    Returns
    -------
    (n_values) ndarray of floats
        Values obtained from the momentum operation. The length depends on
        the operation to be performed. If, e.g., all bead momenta of a
        single degree of freedom is requested, n_values will be n_beads of
        that degree of freedom.
    """

    quantity = 'v' if opts.vel else 'p'

    # get coordinate values under consideration here!
    current_value = log_nuclei.get_selected_quantities(opts.x1, quantity, opts.rpmd)
    if opts.x2 is not None:
        raise NotImplementedError("Implement relative momentum calculations, etc.")
        # coordinate_2 = log_nuclei.parse_coordinate(opts.x2, quantity, opts.rpmd)
        # try:
        #     current_value = np.linalg.norm(current_value - coordinate_2, axis=0)
        # except ValueError as e:
        #     raise type(e)(str(e) + "\nXPACDT: Cannot calculate relative momentum"
        #                   " between given sites. Maybe different number of dofs per site.")

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


def _arguments_electronic_state(arguments):
    """ Parses the options for electronic state command. If no options given
    it will raise an error.

    Valid options are as follows:

    -b <basis> given: Electronic basis to be used. Can be "adiabatic" or
                      "diabatic". Default: "adiabatic".
    -p <a> given: State to be projected onto in the basis given by 'basis'.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the state command. See above.

    Returns
    -------
    opts: argparse.Namespace object
        Options for state operation.
    """
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
                        help='Basis to be used. Possible "adiabatic" or'
                             ' "diabatic". Default: "adiabatic".')

    parser.add_argument('-p', '--project',
                        dest='proj',
                        type=int,
                        default=None,
                        help='State to be projected onto.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to state operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    return opts


def _electronic_state(opts, log_nuclei):
    """Performs operations related to electronic state on `log_nuclei` using
    input options from `opts`.

    Parameters
    ----------
    opts: argparse.Namespace object
        Options for state operation.
    log_nuclei: XPACDT.System.Nuclei object
        Nuclei object from the log to perform operations on.

    Returns
    -------
    (1) ndarray of float
        Value obtained from state operation.
    """

    # TODO: add PES for analysis which was removed for storing purposes.
    if log_nuclei.electrons.pes is not None:
        if (opts.proj >= log_nuclei.electrons.pes.n_states):
            raise ValueError("\nXPACDT: State to be projected onto is greater than"
                             " the number of states. Note: State count starts from"
                             " 0. Given state to project is: " + str(opts.proj))

    current_value = log_nuclei.electrons.get_population(opts.proj, opts.basis)

    return np.array(current_value).flatten()


def _arguments_energy(arguments):
    """ Parses the options for energy command. If no options given it will
    return parser object to get total centroid energy.

     Valid options are as follows:

    -t <type> given: Type of energy to be used. This can be "total", "kinetic",
                     "potential" or "spring". Default: "total". Note: energy
                     of spring term is not valid for centroid.
    -r given: Get ring-polymer bead energy instead of centroid energy.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the energy command. See above.

    Returns
    -------
    opts: argparse.Namespace object
        Options for energy operation.
    """
    parser = argparse.ArgumentParser(usage="Options for +energy", add_help=False)

    parser.add_argument('-h', '--help',
                        dest='help',
                        action='store_true',
                        default=False,
                        help='Prints this help page.')

    parser.add_argument('-t', '--type',
                        dest='type',
                        type=str,
                        default='total',
                        choices=['total', 'kinetic', 'potential', 'spring'],
                        required=False,
                        help='Type of energy to be used. Possible "total",'
                             ' "kinetic", "potential" or "spring".'
                             ' Default: "total".')

    parser.add_argument('-r', '--rpmd',
                        dest='rpmd',
                        action='store_true',
                        default=False,
                        help='Use bead energy instead of centroid energy.')

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    return opts


def _energy(opts, log_nuclei):
    """ Performs operations related to energy on `log_nuclei` using input
    options from `opts`.

    Parameters
    ----------
    opts: argparse.Namespace object
        Options for energy operation.
    log_nuclei: XPACDT.System.Nuclei object
        Nuclei object from the log to perform operations on.

    Returns
    -------
    (1) ndarray of float
        Value obtained from energy operation.
    """

    if ((not opts.rpmd) and (opts.type == "spring")):
        raise RuntimeError("\nXPACDT: Energy of spring terms for centroid"
                           " does not make sense.")

    energy_attribute = "energy"

    if not opts.rpmd:
        energy_attribute += "_centroid"

    if not (opts.type == "total"):
        energy_attribute = opts.type + "_" + energy_attribute

    current_value = getattr(log_nuclei, energy_attribute)

    return np.array(current_value).flatten()


def _arguments_electronic_dm(arguments):
    """ Parses the options for electronic density matrix command. If no
    options given it will raise an error.

    Valid options are as follows:

    -b <basis> given: Electronic basis to be used. Can be "adiabatic" or
                      "diabatic". Default: "adiabatic".
    -1 <a> given: First electronic state index of the density matrix.
    -2 <b> given: Second electronic state index of the density matrix.

    Parameters
    ----------
    arguments: list of strings
        Command line type options given to the electronic density matrix
        command. See above.

    Returns
    -------
    opts: argparse.Namespace object
        Options for electronic density matrix operation.
    """
    parser = argparse.ArgumentParser(usage="Options for +dm", add_help=False)

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
                        required=True,
                        help='Basis to be used. Possible "adiabatic" or'
                             ' "diabatic". Default: "adiabatic".')

    parser.add_argument('-1', '--s1',
                        dest='s1',
                        type=int,
                        default=None,
                        help='First electronic state index of the density'
                             ' matrix. Please note that the numbering starts at 0.')

    parser.add_argument('-2', '--s2',
                        dest='s2',
                        type=int,
                        default=None,
                        help='Second electronic state index of the density'
                             ' matrix. Please note that the numbering starts at 0.')

    if len(arguments) == 0:
        raise RuntimeError("XPACDT: No arguments given to electronic density"
                            " matrix operation.")

    opts = parser.parse_args(arguments)

    if opts.help is True:
        parser.print_help()
        return None

    return opts


def _electronic_dm(opts, log_nuclei):
    """Performs operations related to electronic density matrix on `log_nuclei`
    using input options from `opts`.

    Parameters
    ----------
    opts: argparse.Namespace object
        Options for density matrix operation.
    log_nuclei: XPACDT.System.Nuclei object
        Nuclei object from the log to perform operations on.

    Returns
    -------
    (1) ndarray of float /or/ complex
        Value obtained from electronic density matrix operation.
    """
    rho = log_nuclei.electrons.get_rho(opts.basis, log_nuclei.positions)

    if opts.s1 == opts.s2:
        # Get real value for populations
        current_value = rho[opts.s1, opts.s2].real
    else:
        # TODO: Should this return complex coherence rho_12?
        current_value = np.abs(rho[opts.s1, opts.s2])

    return np.array(current_value).flatten()

