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

""" Module that implements free (without any external potential) or harmonic
potential ring polymer sampling needed for RPMD initialization."""

import copy
import numpy as np

import XPACDT.Tools.NormalModes as nm
import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo


def do_RingPolymer_sampling(system, parameters, n_sample):
    """
    Perform ring polymer sampling by picking from either free or harmonic
    potential ring polymer distribution.

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
    # !!! Here centroid x and p seem to be fixed! How to solve this?

    nuclei = system.nuclei

    # TODO: are centroid momenta and RPMD needed?
    if nuclei.momenta is None:
        raise RuntimeError("\nXPACDT: Momenta not provided in system or input"
                           " file, but required in fixed sampling.")

    if 'rpmd' not in parameters:
        raise RuntimeError("\nXPACDT: RPMD information needed for"
                            " ring polymer sampling.")

    sampling_parameters = parameters.get('sampling')

    # Should w_o be a list for each dof??? Then in that case modes should not be selected.
    if 'add_harmonic' in sampling_parameters:
        w_o = float(sampling_parameters.get('w_o', None))

        # TODO : Check w_o is of length n_dof
        # if w_o is not None:
        #    w_o = [float(i) for i in w_o.split()]

        # else:
        #    w_o = nm.get_sampling_modes(system, parameters)[0]

        # Need to assert that omega are not -ve!!! Should this be done only at minima.
        #if (w_o <= 0).any():
        #    raise RuntimeError("\nXPACDT: Negative frequency given for sampling. "
        #                       + "omega = " + str(w_o)
        #                       + " Please make sure that the input geometry"
        #                       " is optimized.")
    else:
        w_o = None

    NMtransform_type = parameters.get('rpmd').get("nm_transform", "matrix")
    RPtransform = RPtrafo.RingPolymerTransformations(nuclei.n_beads,
                                                     NMtransform_type)

    systems = []
    for _ in range(n_sample):
        rp_coord = np.zeros((nuclei.n_dof, max(nuclei.n_beads)))
        rp_momenta = np.zeros((nuclei.n_dof, max(nuclei.n_beads)))

        for i in range(nuclei.n_dof):
            rp_coord[i] = RPtransform.sample_free_rp_coord(
                nuclei.n_beads[i], nuclei.masses[i], nuclei.beta,
                nuclei.x_centroid[i], w_o)

            rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                nuclei.n_beads[i], nuclei.masses[i], nuclei.beta,
                nuclei.p_centroid[i])

        systems.append(copy.deepcopy(system))
        systems[-1].nuclei.positions = rp_coord.copy()
        systems[-1].nuclei.momenta = rp_momenta.copy()
        systems[-1].do_log(init=True)

    return systems
