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

import copy
import numpy as np

import XPACDT.Tools.NormalModes as nm


def do_Quasiclassical_sampling(system, parameters, n_sample):
    """
    Perform quasiclassical sampling, i.e., sample the normal modes with a
    random phase and a fixed energy.
    The basic idea is presented in: Chem. Phys. Lett. 74, 284 (1980)
    TODO: Are there other, better references?

    The following things are assumed to be set in parameters:
    TODO

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
        A list of systems located at the sampled phase-space points.
    """

    x0 = system.nuclei.positions[:, 0]
    omega, nm_masses, nm_cartesian = nm.get_sampling_modes(system, parameters)

    assert((omega > 0.0).all()), "Negative frequency given for sampling. " \
                                 + "omega = " + str(omega)

    # Get the quantum numbers or set to 0
    if 'quantum_numbers' in parameters.get("sampling"):
        qn_string = parameters.get("sampling").get("quantum_numbers")
        quantum_numbers = np.array([float(i) for i in qn_string.split()])
    else:
        quantum_numbers = np.zeros_like(omega)

    factor = np.sqrt(1.0 + 2.0*quantum_numbers)
    factor_x = factor * np.sqrt(1.0 / (omega * nm_masses))
    factor_p = factor * np.sqrt((omega * nm_masses))

    # Draw from random angle distribution
    angles = 2.0 * np.pi * np.random.random(n_sample)
    x_normal_modes = np.outer(factor_x, np.sin(angles))
    p_normal_modes = np.outer(factor_p, np.cos(angles))

    # Transform to cartesian coordinates
    xs, ps = nm.transform_to_cartesian(x_normal_modes, p_normal_modes,
                                       x0, nm_cartesian)

    # From all coordinates and momenta, generate a list of systems at these
    # particular phase space points. That is the standard format handled later.
    systems = []
    for x, p in zip(xs, ps):
        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.positions = x.reshape(-1, 1)
        systems[-1].nuclei.momenta = p.reshape(-1, 1)
        systems[-1].log(init=True)

    return systems
