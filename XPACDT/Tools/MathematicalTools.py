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

"""Module containing some mathematical functions customized to the need of the
program."""

import math
import numpy as np


def linear_interpolation_1d(x_fraction, y_initial, y_final):
    """ Performs linear interpolation in 1 dimension to obtain y value at
    `x_fraction` between the intial and final x values.
    TODO: write properly: x_fraction = (x_required - x_initial) / (x_final - x_initial)

    Parameters
    ----------
    x_fraction : float
        Fraction between initial and final x values. Needs to be less than or equal to 1.
    y_initial : ndarray
        Initial y value. Can be any dimensional ndarray.
    y_final : ndarray
        Final y value. Same shape as y_initial.

    Returns
    -------
    ndarray of floats (same as y_initial)
        Interpolated value at `x_fraction` between the intial and final values.
    """
    # Asserting x_fraction <= 1 is not done since sometimes scipy.integrate.ode
    # has it more than 1 for adaptive time step so it does linear extrapolation
    # instead, which is still fine with this function.
    return ((1. - x_fraction) * y_initial + x_fraction * y_final)
