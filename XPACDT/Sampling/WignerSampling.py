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

""" Modul that implements Wigner sampling of a system. Details on Wigner
sampling can be found in TODO: add paper. Please note that Wigner sampling
only makes sense in a classical calculation and not for RPMD. """

import copy
import molmod.constants as const
import numpy as np
import XPACDT.Tools.NormalModes as nm


def do_Wigner_sampling(system, parameters, n_sample, hessian=None):
    """
    Perform Wigner sampling of normal modes. Either the ground state or
    a thermal distribution is sampled. A list of systems located at the
    sampled phase-space points is given back.

    # The following things are assumed to be set in parameters:
    # TODO move all optional stuff to parameters...

    Parameters
    ----------
    system : XPACDT.Dynamics.System
        A representation of the basic system with potential interface set up
        and a valid starting geometry.
    parameters : XPACDT input file
        Dictonary-like presentation of the input file.
    n_sample : int
        Actual number of samples required.
    hessian : (n_dof, n_dof) ndarray of floats, optional
        A Hessian for the system the defines the normal modes to be sampled.

    Returns
    -------
    systems : (n_sample) list of XPACDT.Dynamics.System
        A list of systems located at the sampled phase-space points.
    """

    assert('samples' in parameters.get('sampling')), "Number of " \
        "samples required not given."

    x0 = system.nuclei.positions[:, 0]
    omega, nm_masses, nm_cartesian = nm.get_sampling_modes(system, parameters)

    assert((omega > 0.0).all()), "Negative frequency given for sampling. " \
                                 + "omega = " + str(omega)

    # Get the width of the ground state distribution
    sigma_x = np.sqrt(1.0 / (2.0 * omega * nm_masses))
    sigma_p = np.sqrt((omega * nm_masses) / (2.0))

    # Add a thermal scaling if a thermal distribution is required
    if "temperature" in parameters.get("sampling"):
        temperature = float(parameters.get("sampling").
                            get('temperature').split()[0])
        beta = 1.0 / (temperature * const.boltzmann)
        thermal_factor = np.sqrt(1.0 / np.tanh(beta * omega / 2.0))

        sigma_x *= thermal_factor
        sigma_p *= thermal_factor

    # Draw from normal distribution
    x_normal_modes = np.random.normal(np.zeros_like(sigma_x), sigma_x,
                                      (len(sigma_x), n_sample))
    p_normal_modes = np.random.normal(np.zeros_like(sigma_p), sigma_p,
                                      (len(sigma_p), n_sample))

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
        systems[-1].do_log(init=True)

    return systems
