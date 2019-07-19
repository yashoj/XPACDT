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

""" Module to perform operations on a system log. SUPER RAW!"""

import numpy as np
from optparse import OptionParser


def position(arguments, log):
    """Does perform operations related to positions. If no options given it
    will return None.

    Valid options are as follows:

    -x1 <a> given: coordinate value of a given coordinates, e.g., -x 0, gives
                   the x coordinate of the first atom, or -x 0,3,7 gives the x
                   coordinate of the first and second atom and the y coordinate
                   of the third atom. Alternatively, also the 3D coordinate of
                   a center  of mass can be obtained by giving m and a comma
                   separated list of atoms. Single atoms can be obtained as
                   center of mass of that atom, i.e., -x m,1.
    -x2 <b> given: Like x1. If both given, then the distance between them is
                   used.
    -p <a> given: if a single value is calculated (i.e. a distance or single
                  coordinate) this option projects it onto a certain range.
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
                         help='Obtain the coordinate of a site. A site can be defined as an atom by its number or as a center of mass by giving m and a set of atoms by their numbers with comma sepatation.')

    obsparser.add_option('-2', '--x2',
                         dest='x2',
                         type='str',
                         default=None,
                         help='If given, the distance between the two given (x1 and x2) sites should be calculated. A site can be defined as an atom by its number or as a center of mass by giving m and a set of atoms by their numbers with comma sepatation.')

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
    current_value = log['nuclei'].parse_coordinate(opts.x1, opts.rpmd)
    if opts.x2 is not None:
        coordinate_2 = log['nuclei'].parse_coordinate(opts.x2)
        # TODO: howto with beads...
        current_value = np.linalg.norm(current_value - coordinate_2, opts.rpmd)

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
