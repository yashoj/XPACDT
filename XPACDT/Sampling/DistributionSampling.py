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

""" Module that implements sampling from a distribution."""

import sys
import copy
import numpy as np


def do_Distribution_sampling(system, parameters, n_sample):
    """
    Perform sampling from a chosen distribution.

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        A representation of the basic system with potential interface set up
        and a valid starting geometry.
    parameters : XPACDT input file
        Dictonary-like presentation of the input file.
    n_sample : int
        Actual number of samples required.

    Returns
    -------
    systems : (n_sample) list of XPACDT.Dynamics.System
        A list of n_sample copies of the given system.
    """

    sampling_parameters = parameters.get('sampling')

    if 'x_dist' not in sampling_parameters:
        raise RuntimeError("\nXPACDT: Position distribution not specified for"
                            " to sample from.")

    if 'p_dist' not in sampling_parameters:
        raise RuntimeError("\nXPACDT: Momentum distribution not specified for"
                            " to sample from.")

    x_dist = sampling_parameters.get('x_dist').split()
    p_dist = sampling_parameters.get('p_dist').split()

    if (len(x_dist[1:]) != system.nuclei.n_dof):
        raise RuntimeError("\nXPACDT: Additional parameters given for position"
                           " distribution is not equal to the degree of freedom.")

    if (len(p_dist[1:]) != system.nuclei.n_dof):
        raise RuntimeError("\nXPACDT: Additional parameters given for momentum"
                           " distribution is not equal to the degree of freedom.")

    systems = []
    for _ in range(n_sample):
        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.positions = getattr(
            sys.modules[__name__], "sample_" + x_dist[0] + "_distribution")(
                np.array(x_dist[1:], dtype=float), max(system.nuclei.n_beads))
        systems[-1].nuclei.momenta = getattr(
            sys.modules[__name__], "sample_" + p_dist[0] + "_distribution")(
                np.array(p_dist[1:], dtype=float), max(system.nuclei.n_beads))
        systems[-1].do_log(init=True)

    return systems


def sample_fixed_distribution(values, n_beads=1):
    """
    Get fixed sample.
    Parameters
    ----------
    values : (n_dof)
    n_beads : int, default 1
        Number of beads

    Returns
    -------
    systems : (n_sample) list of XPACDT.Dynamics.System
        A lis
    """
    x = np.repeat(values[:, np.newaxis], n_beads, axis=1)
    return x


def sample_gaussian_distribution(values, n_beads=1):
    """
    Get random samples from gaussian distribution centred at 0.
    """
    x = np.array([(np.random.normal(0.0, float(v), n_beads)) for v in values])
    return x