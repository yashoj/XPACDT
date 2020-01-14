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

""" This module converts two level system from diabatic representation to
adiabatic representation."""

import numpy as np


def get_adiabatic_energy(V):
    """
    Get adiabatic energies from a given two level diabatic matrix.

    Parameters:
    ----------
    V : (2, 2) ndarrays of floats /or/ (2, 2, n_beads) ndarrays of floats
        Two state diabatic potential matrix.

    Returns:
    ----------
    V_ad : (2) ndarrays of floats /or/ (2, n_beads) ndarrays of floats
        Adiabatic potential energies for the two states.
    """
    # TODO: Do we really need this checks? Isn't V matrix built from n_states?
    # Also check for hermiticity?
    assert ((V.shape[0] == 2) and (V.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")

    sum_diag = V[1, 1] + V[0, 0]
    diff_diag = V[1, 1] - V[0, 0]
    root = np.sqrt(diff_diag * diff_diag + 4 * V[0, 1] * V[0, 1])

    V_ad = np.array([0.5 * (sum_diag - root), 0.5 * (sum_diag + root)])

    return V_ad


def get_adiabatic_gradient(V, dV):
    """
    Get adiabatic gradients from a given two level diabatic matrix.

    Parameters:
    ----------
    V : (2, 2) ndarrays of floats /or/ (2, 2, n_beads) ndarrays of floats
        Two state diabatic potential energy matrix.
    dV : (2, 2, n_dof) ndarrays of floats
         /or/ (2, 2, n_dof, n_beads) ndarrays of floats
        Two state diabatic potential gradient matrix.

    Returns:
    ----------
    dV_ad : (2, n_dof) ndarrays of floats
            /or/ (2, n_dof, n_beads) ndarrays of floats
        Adiabatic potential gradients for the two states.
    """

    assert ((V.shape[0] == 2) and (V.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")
    assert ((dV.shape[0] == 2) and (dV.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")

    diff_diag = V[1, 1] - V[0, 0]
    root = np.sqrt(diff_diag * diff_diag + 4 * V[0, 1] * V[0, 1])

    grad_sum_diag = dV[1, 1] + dV[0, 0]
    grad_diff_diag = dV[1, 1] - dV[0, 0]

    pm_term = (diff_diag * grad_diff_diag + 4 * dV[0, 1] * V[0, 1]) / root

    dV_ad = np.array([0.5 * (grad_sum_diag - pm_term),
                      0.5 * (grad_sum_diag + pm_term)])

    return dV_ad


def get_NAC(V, dV):
    """
    Get non-adiabatic coupling (NAC) vector from a given two level diabatic
    matrix. For the choice of transformation matrix here, this is basically
    the negative gradient of the rotation angle :math:'\\theta'.

    Parameters:
    ----------
    V : (2, 2) ndarrays of floats /or/ (2, 2, n_beads) ndarrays of floats
        Two state diabatic potential energy matrix.
    dV : (2, 2, n_dof) ndarrays of floats
         /or/ (2, 2, n_dof, n_beads) ndarrays of floats
        Two state diabatic potential gradient matrix.

    Returns:
    ----------
    nac : (2, 2, n_dof) ndarrays of floats
          /or/ (2, 2, n_dof, n_beads) ndarrays of floats
        NAC for the two states given in matrix form.
    """

    assert ((V.shape[0] == 2) and (V.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")
    assert ((dV.shape[0] == 2) and (dV.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")

    nac = np.zeros_like(dV)

    diff_diag = V[1, 1] - V[0, 0]
    square_of_root = diff_diag * diff_diag + 4 * V[0, 1] * V[0, 1]
    grad_diff_diag = dV[1, 1] - dV[0, 0]

    nac[0, 1] = (dV[0, 1] * diff_diag - V[0, 1] * grad_diff_diag) / square_of_root
    nac[1, 0] = -1. * nac[0, 1]

    return nac


def get_transformation_matrix(V):
    """
    Get the unitary transformation matrix U to change the basis from diabatic
    to adiabatic. The columns of the matrix are the adiabatic eigenstates.
    The matrix is chosen to be a 2D rotation matrix of the form:
    .. math::

        U = \\begin{pmatrix}
            \\cos\\theta & -\\sin\\theta \\
            \\sin\\theta &  \\cos\\theta
            \\end{pmatrix}

    where the rotation angle
    :math:'\\theta = \\frac{1}{2} \\arctan(\\frac{2V_{12}}{V_{11} - V_{22}})'

    Parameters:
    ----------
    V : (2, 2) ndarrays of floats /or/ (2, 2, n_beads) ndarrays of floats
        Two state diabatic potential energy matrix.

    Returns:
    ----------
    U : (2, 2) ndarrays of floats /or/ (2, 2, n_beads) ndarrays of floats
        Unitary transformation matrix.
    """

    assert ((V.shape[0] == 2) and (V.shape[1] == 2)),\
           ("Diabatic energy matrix needs to have exactly 2 states")

    # Note: np.arctan2 is used as it gives rotation angle between -pi and pi
    #       whereas np.arctan only gives angle between -pi/2 and pi/2
    theta = 0.5 * np.arctan2(2.0 * V[0, 1], (V[0, 0] - V[1, 1]))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    U = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    return U