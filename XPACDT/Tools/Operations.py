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

""" Module to perform operations on a system log to obtain some observales
and perform analysis. """

import numpy as np
from optparse import OptionParser


def position(arguments, log):
    """Does perform operations related to positions. If no options given it
    will return None.

    Valid options are as follows:

    -x1 <a> given: Position value of a given degree of freedom, e.g., -x 0,
                   gives the first position, or -x 0,3,7 gives the first,
                   fourth and seventh position. Alternatively, also the
                   center of mass position can be obtained by giving m and a
                   comma separated list of degrees of freedom.
    -x2 <b> given: Like x1. If both given, then the distance between them is
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
    log: dict containing the system information (i.e., from the log).

    Output:
        values obtained from the position operation.
    """

    obsparser = OptionParser(usage="Options for +pos", add_help_option=False)

    obsparser.add_option('-h', '--help',
                         dest='help',
                         action='store_true',
                         default=False,
                         help='Prints this help page.')

    obsparser.add_option('-1', '--x1',
                         dest='x1',
                         type='str',
                         default=None,
                         help='Obtain the position of a list of degrees of freedom (given as comma-separated list) or a center of mass position; given as m followed by the list of degrees of freedom included.')

    obsparser.add_option('-2', '--x2',
                         dest='x2',
                         type='str',
                         default=None,
                         help='If given, the distance between the two given (x1 and x2) sites should be calculated.')

    obsparser.add_option('-p', '--project',
                         dest='proj',
                         type='str',
                         default=None,
                         help='Take the distance or coordinate value given and project onto a certain range. Valid are fully bound ranges (A,<,B), or below a value (<,A) or above a value (>,A).')

    obsparser.add_option('-r', '--rpmd',
                         dest='rpmd',
                         action='store_true',
                         default=False,
                         help='Use beads instead of centroids.')

    opts, args = obsparser.parse_args(arguments)

    if opts.help is True:
        obsparser.print_help()
        return None

    # get coordinate values under consideration here!
    current_value = log['nuclei'].parse_dof(opts.x1, 'x', opts.rpmd)
    if opts.x2 is not None:
        coordinate_2 = log['nuclei'].parse_dof(opts.x2, 'x', opts.rpmd)
        # TODO: howto with beads...
        current_value = np.linalg.norm(current_value - coordinate_2)

    # if we want to project the distance/coordinate onto a certain interval
    if opts.proj is not None:
        vals = opts.proj.split(',')

        # case: above a value; > A
        if vals[0] == '>':
            if current_value > float(vals[1]):
                current_value = 1.0
            else:
                current_value = 0.0

        # case: below a value; < A
        if vals[0] == '<':
            if current_value < float(vals[1]):
                current_value = 1.0
            else:
                current_value = 0.0

        # case: between values < A <
        if vals[1] == '<':
            if current_value > float(vals[0]) and current_value < float(vals[2]):
                current_value = 1.0
            else:
                current_value = 0.0

    return current_value


def momentum(arguments, log):
    """Does perform operations related to momenta. If no options given it
    will return None.

    Valid options are as follows:

    -v given: Use velocities instead of momenta.
    -x1 <a> given: momentum value of a given degree of freedom, e.g., -x 0,
                   gives the first momentum, or -x 0,3,7 gives the first,
                   fourth and seventh momentum. Alternatively, also the
                   center of mass momentum can be obtained by giving m and a
                   comma separated list of degrees of freedom.
    -x2 <b> given: Like x1. If both given, then the relative momentum between
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
    log: dict containing the system information (i.e., from the log).

    Output:
        values obtained from the momentum operation.
    """

    obsparser = OptionParser(usage="Options for +mom", add_help_option=False)

    obsparser.add_option('-h', '--help',
                         dest='help',
                         action='store_true',
                         default=False,
                         help='Prints this help page.')

    obsparser.add_option('-v', '--velocities',
                         dest='vel',
                         action='store_true',
                         default=False,
                         help='Use velocities instead of momenta.')

    obsparser.add_option('-1', '--x1',
                         dest='x1',
                         type='str',
                         default=None,
                         help='Obtain the momentum of a list of degrees of freedom (given as comma-separated list) or a center of mass; given as m followed by the list of degrees of freedom included.')

    obsparser.add_option('-2', '--x2',
                         dest='x2',
                         type='str',
                         default=None,
                         help='If given, the relative momentum between the two given (x1 and x2) sites should be calculated.')

    obsparser.add_option('-p', '--project',
                         dest='proj',
                         type='str',
                         default=None,
                         help='Take the momentum value given and project onto a certain range. Valid are fully bound ranges (A,<,B), or below a value (<,A) or above a value (>,A).')

    obsparser.add_option('-r', '--rpmd',
                         dest='rpmd',
                         action='store_true',
                         default=False,
                         help='Use beads instead of centroids.')

    opts, args = obsparser.parse_args(arguments)

    if opts.help is True:
        obsparser.print_help()
        return None

    quantity = 'v' if opts.vel else 'p'

    # get coordinate values under consideration here!
    current_value = log['nuclei'].parse_dof(opts.x1, quantity, opts.rpmd)
    if opts.x2 is not None:
        coordinate_2 = log['nuclei'].parse_coordinate(opts.x2, quantity, opts.rpmd)
        # TODO: howto with beads...
        current_value = np.linalg.norm(current_value - coordinate_2)

    # if we want to project the distance/coordinate onto a certain interval
    if opts.proj is not None:
        vals = opts.proj.split(',')

        # case: above a value; > A
        if vals[0] == '>':
            if current_value > float(vals[1]):
                current_value = 1.0
            else:
                current_value = 0.0

        # case: below a value; < A
        if vals[0] == '<':
            if current_value < float(vals[1]):
                current_value = 1.0
            else:
                current_value = 0.0

        # case: between values < A <
        if vals[1] == '<':
            if current_value > float(vals[0]) and current_value < float(vals[2]):
                current_value = 1.0
            else:
                current_value = 0.0

    return current_value
