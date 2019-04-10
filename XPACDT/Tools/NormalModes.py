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


"""This module handles normal modes of a given system (e.g. a molecule). """

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


def get_sampling_modes(system, parameters):
    """
    Obtain all information required for the normal modes to be sampled. The
    required Hessian is calculated directly from the given potential and then
    normal modes are obtained. Only information on the normal modes that should
    be sampled as specified in the input file are returned.

    TODO: Option to read Hessian from file!
    TODO: Option to optimize geometry before Hessian calculation.

    The input file can give the modes to be sampled by specifying the 'modes'
    keyword in the 'sampling' section. If
        'linear' is given, the first 5 modes are not sampled.
        'nonlinear' is given, the first 6 modes are not sampled.
        a list of integers is given, these modes are sampled.
        nothing is given, all modes are sampled.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        System that defines the initial geometry and the potential.
    parameters : XPACDT.Input.Inputfile
        XPACDT representation of the given input file.
    Returns
    -------
    omega : ndarray of floats
        Normal mode frequencies in au.
    nm_masses : ndarray of floats
        Normal mode masses in au.
    nm_cartesian : two-dimensional ndarray of floats, shape(#dof, #modes)
        Transformation matrix from the normal modes that should be sampled to
        the full cartesian coordinates.
    """

    hessian = system.pes.get_Hessian(system.nuclei.positions[:, 0])

    omega, nm_masses, normal_modes, nm_cartesian = \
        get_normal_modes(hessian, system.masses)

    # get modes to be sampled from input. This can be
    # - nothing given - sample all modes
    # - linear - linear system, so remove first 5 modes
    # - nonlinear - not a linear system, so remove first 6
    # - a list of numbers - only sample the given ones
    modes = parameters.get("sampling").get("modes")
    if modes is None or modes == '':
        modelist = range(system.n_dof)
    elif modes == 'linear':
        modelist = range(5, system.n_dof)
    elif modes == 'nonlinear':
        modelist = range(6, system.n_dof)
    else:
        modelist = [int(i) for i in modes.split()]

    assert(len(modelist) > 0), "No modes selected for sampling!"

    return omega[modelist], nm_masses[modelist], nm_cartesian[:, modelist]
