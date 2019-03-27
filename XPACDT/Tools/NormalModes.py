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

import numpy as np


def get_normal_modes(Hessian, mass):
    """
    Obtain the normal mode frequencies and mass weighted normal modes of a
    given Hessian. c.f.

    TODO: Implement option to project out rotation, translation.

    Parameters
    ----------
    Hessian : two-dimensional ndarray of floats
        The Hessian matrix of the system in au. Not mass weighted!
    mass : ndarray of floats
        The mass for each degree of freedom in au.

    Returns
    -------
    omega : ndarray of floats
        List of all normal mode frequencies in au. "Complex" frequencies
        (i.e. associated with a negative eigenvalue of the Hessian) are given
        back as the negative of the squareroot of the absolute eigenvalue.
    mode_masses : ndarray of floats
        The masses associated with each normal mode in au.
    vec : two-dimensional ndarray of floats
        Mass-weighted normal modes in au as columns.
    cartesian : two-dimensional ndarray of floats
        Cartesian displacements for each normal mode in au as column.
    """

    mass_matrix = np.diag(1.0 / np.sqrt(mass))
    mass_weighted_hessian = np.dot(mass_matrix, np.dot(Hessian, mass_matrix))

    val, vec = np.linalg.eigh(mass_weighted_hessian)

    mask = np.piecewise(val, [val < 0, val >= 0], [-1, 1])
    omega = np.sqrt(abs(val)) * mask

    cartesian = np.dot(mass_matrix, vec)
    mode_masses = 1.0 / np.diag(np.dot(cartesian.T, cartesian))
    cartesian = np.dot(cartesian, np.diag(np.sqrt(mode_masses)))

    return omega, mode_masses, vec, cartesian


def transform_to_cartesian(x, p, x0, normal_modes):
    """
    Transform from normal mode coordinates and momenta to cartesian
    coordinates and momenta.

    TODO: check broadcasting
    Parameters
    ----------
    x : ndarray of floats
        Normal mode coordinate values.
    p : ndarray of floats
        Normal mode momenta values.
    x0 : ndarray of floats
        Reference position used in normal mode calculation/Hessian calculation
        in au.
    normal_modes : two-dimensional ndarray of floats
        Cartesian displacments for each normal mode in au as columns.

    Returns
    -------
    ndarray of floats
        Cartisian coordinates
    ndarray of floats
        Cartisian momenta
    """

    return x0 + np.dot(normal_modes, x).T, np.dot(normal_modes, p).T
