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

""" This module converts N level system from diabatic representation to
adiabatic representation."""

import numpy as np


def get_adiabatic_energy(V):
    """
    Obtain adiabatic energies from a given N level diabatic matrix.

    Parameters:
    ----------
    V : (n_states, n_states) ndarrays of floats
        /or/ (n_states, n_states, n_beads) ndarrays of floats
        Diabatic potential matrix.

    Returns:
    ----------
    V_ad : (n_states) ndarrays of floats
           /or/ (n_states, n_beads) ndarrays of floats
        Adiabatic potential energies for each state.
    """
    if len(V.shape) == 3:
        # Get shape (n_beads, n_states, n_states) for vectorized diagonalization
        V = V.transpose(2, 0, 1)

    V_ad = np.linalg.eigvalsh(V)

    return V_ad.T


def get_adiabatic_gradient(R, func_diabatic_energy, step):
    """
    Obtain adiabatic gradients from a given N level diabatic matrix.

    Parameters:
    ----------
    R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
        The positions of all centroids or beads in the system.
    func_diabatic_energy : function
        Function to get diabatic energies of shape (n_states, n_states)
        or (n_states, n_states, n_beads) ndarrays of floats to be compatible
        with 'get_adiabatic_energy'. Should take 'R' as parameter.
    step : float
        Step size for numerical derivative.

    Returns:
    ----------
    dV_ad : (n_states, n_dof) ndarrays of floats
            /or/ (n_states, n_dof, n_beads) ndarrays of floats
        Adiabatic potential gradients for each state.
    """

    n_dof = R.shape[0]
    dV_ad = []

    # Get gradients for each dof using symmetric finite difference
    # TODO: maybe have a different module to do numerics efficiently
    # TODO: maybe can be made more efficient by computing only for upper
    #       triangle part of the matrix and use hermiticity property
    for i in range(n_dof):
        R_step = R.copy()
        R_step[i] += step
        V_ad_plus = get_adiabatic_energy(func_diabatic_energy(R_step))

        R_step[i] -= (2. * step)
        V_ad_minus = get_adiabatic_energy(func_diabatic_energy(R_step))

        dV_ad.append((V_ad_plus - V_ad_minus) / (2. * step))

    # Initially dV_ad is n_dof list of (n_states) /or/ (n_states, n_beads) ndarrays of floats 
    dV_ad = np.swapaxes(np.array(dV_ad), 0, 1)

    return dV_ad


def get_NAC(V, dV):
    """
    Obtain non-adiabatic coupling (NAC) vector from a given N level diabatic
    matrix. The NAC between the k-th and j-th state, labelled
    :math:'d_{kj}', is given by
    .. math::

        d_{kj} = \\bra{\\phi^{adiab}_k} \\overrightarrow{\\nabla} \\ket{\\phi^{adiab}_j}
               = \\frac{\\bra{\\phi^{adiab}_k} \\overrightarrow{\\nabla} \\hat{V} \\ket{\\phi^{adiab}_j}}
                       {\\epsilon_j - \\epsilon_k}

    where :math:'\\epsilon_k' and :math:'\ket{\phi^{adiab}_k}' are the k-th
    state adiabatic energy and eigenstate respectively, and
    :math:'\\overrightarrow{\\nabla} \\hat{V}' is the diabatic gradient.
    
    For reference, please see J. Chem. Phys. 101, 6 (1994).

    Parameters:
    ----------
    V : (n_states, n_states) ndarrays of floats
        /or/ (n_states, n_states, n_beads) ndarrays of floats
        Diabatic potential energy matrix
    dV : (n_states, n_states, n_dof) ndarrays of floats
         /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats
        Diabatic potential gradient matrix

    Returns:
    ----------
    nac : (n_states, n_states, n_dof) ndarrays of floats
          /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats
        NAC for each state given in matrix form.
    """
    # TODO: Comparing with 2 state dia2ad, the sign is negative, could be 
    # possibly due to phase factor in adiabatic states. Does that matter??
    n_states = V.shape[0]
    nac = np.zeros_like(dV)

    # Maybe not the most efficient, make this more efficient by having only
    # one function that returns all adiabatic properties?
    V_ad = get_adiabatic_energy(V)
    U = get_transformation_matrix(V)

    # Gradients after applying transformation matrices which contain the
    # adiabatic states
    if len(V.shape) == 2:
        transformed_grad = np.matmul((U.conjugate()).T, np.matmul(dV.transpose(2, 0, 1), U))
        # Changing from (n_dof, n_states, n_states) to (n_states, n_states, n_dof)
        transformed_grad = transformed_grad.transpose(1, 2, 0)
    else:
        transformed_grad = np.matmul((U.conjugate()).transpose(2, 1, 0),
                                     np.matmul(dV.transpose(2, 3, 0, 1), U.transpose(2, 0, 1)))
        # Changing from (n_dof, n_beads, n_states, n_states) to (n_states, n_states, n_dof, n_beads)
        transformed_grad = transformed_grad.transpose(2, 3, 0, 1)

    for i in range(n_states):
        for j in range(i+1, n_states):
            nac[i, j] = transformed_grad[i, j] / (V_ad[j] - V_ad[i])
        # This should be fine as all the corresponding upper triangular
        # off diagonal elements should have already been calculated.
        for k in range(i):
            nac[i, k] = - nac[k, i].copy()

    return nac


def get_transformation_matrix(V):
    """
    Obtain the unitary transformation matrix U to change the basis from diabatic
    to adiabatic. The columns of the matrix are the adiabatic eigenstates.

    Parameters:
    ----------
    V : (n_states, n_states) ndarrays of floats
        /or/ (n_states, n_states, n_beads) ndarrays of floats
        Diabatic potential matrix.

    Returns:
    ----------
    U : (n_states, n_states) ndarrays of floats
        /or/ (n_states, n_states, n_beads) ndarrays of floats
        Unitary transformation matrix.
    """

    if len(V.shape) == 3:
        # Get shape (n_beads, n_states, n_states) for vectorized diagonalization
        V = V.transpose(2, 0, 1)

    V_ad, U = np.linalg.eigh(V)

    if len(V.shape) == 3:
        return U.transpose(1, 2, 0)
    else:
        return U


if __name__ == '__main__':
    import XPACDT.Interfaces.MorseDiabatic as morse
    pot = morse.MorseDiabatic(4, 'adiabatic', **{'n_states': '3', 'model_type': 'model_1'})

    R = np.array([[3.3, 3.4,  3.5, 3.6]])
    #R = np.array([[2., 3.5, 4., 5.]])

    pot._calculate_all(R)

    #print(pot._energy)
    #print(pot._gradient)
    print(pot._nac, '\n\n')

    print(get_adiabatic_energy(pot._diabatic_energy))
    print(get_adiabatic_gradient(R, pot._get_diabatic_energy_3states, pot.DERIVATIVE_STEPSIZE))
    print(get_NAC(pot._diabatic_energy, pot._diabatic_gradient), '\n\n')

    print(np.allclose(get_adiabatic_energy(pot._diabatic_energy), pot._energy))
    print(np.allclose(get_adiabatic_gradient(R, pot._get_diabatic_energy_3states, pot.DERIVATIVE_STEPSIZE), pot._gradient))
    print(np.allclose(get_NAC(pot._diabatic_energy, pot._diabatic_gradient), pot._nac, atol=1e-5))