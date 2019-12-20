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

""" This module combines all sorts of general mathematical functions related
to geometries, vectors, etc. """

import numpy as np


def angle(v1, v2):
    """ Returns the angle in radians between vectors `v1` and `v2`. The values
    of the angle are from :math:`0` to :math:`\pi`. This version is
    numerically stable.
    See: www.cs.berkeley.edu/~wkahan/MathH110/Cross.pdf (page 15)

    Parameters
    ----------
    v1: (3) ndarray of floats
        First vector.
    v2: (3) ndarray of floats
        Second vector.

    Returns
    -------
    float
        Angle in radians between `v1` and `v2`.
        The values are [:math:`0` : :math:`\pi`]
    """

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    vec_difference = np.linalg.norm(v1/norm_v1 - v2/norm_v2)
    vec_sum = np.linalg.norm(v1/norm_v1 + v2/norm_v2)

    return 2.0*np.arctan2(vec_difference, vec_sum)
