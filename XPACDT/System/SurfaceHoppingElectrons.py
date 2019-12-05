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

"""This is an implementation of fewest switches surface hopping (FSSH) for
electronic dynamics, extended to ring polymer molecular dynamics to give
ring polymer surface hopping (RPSH)."""

import math
import numpy as np
import random
from scipy.linalg import expm
from scipy.integrate import ode

# TODO : where to place these scipy import? Only needed for certain inputs.

import XPACDT.System.Electrons as electrons
import XPACDT.Tools.Units as units


class SurfaceHoppingElectrons(electrons.Electrons):
    """
    Tully's fewest switches surface hopping (FSSH) from his original paper [1]_
    is implemented for electronic dynamics with modifications to incorporate
    nuclear quantum effects using ring polymer molecular dynamics. This
    extension, termed ring polymer surface hopping (RPSH), follows from
    another paper from Tully [2]_ and some own modifications to it.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    masses_nuclei : (n_dof) ndarray of floats
        The masses of each nuclear degree of freedom in au.
    R, P : (n_dof, n_beads) ndarray of floats
        The (ring-polymer) positions `R` and momenta `P` representing the
        system nuclei in au.

    References
    ----------
    .. [1] J. Chem. Phys. 93, 1061 (1990)
    .. [2] J. Chem. Phys. 137, 22S549 (2012)

    """

    def __init__(self, parameters, n_beads, masses_nuclei, R, P):

        electronic_parameters = parameters.get("SurfaceHoppingElectrons")
        basis = electronic_parameters.get("basis", "adiabatic")

        electrons.Electrons.__init__(self, "SurfaceHoppingElectrons",
                                     parameters, n_beads, basis)
        # print("Initiating surface hopping in ", self.basis, " basis")

        self.__masses_nuclei = masses_nuclei
        try:
            self.__timestep_scaling_factor = float(electronic_parameters.get("timestep_scaling_factor", 1.0))
            assert (self.timestep_scaling_factor <= 1.0), \
                ("Time step scaling factor for electrons is more than 1. Given"
                 " in input file: " + self.timestep_scaling_factor)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameter 'timestep_scaling_factor'"
                          "for surface hopping not convertable to float.")

        self.__rpsh_type = electronic_parameters.get("rpsh_type", "centroid")
        self.__rpsh_rescaling = electronic_parameters.get("rpsh_rescaling", "bead")
        self.__rescaling_type = electronic_parameters.get("rescaling_type", "nac")
        self.__evolution_picture = electronic_parameters.get("evolution_picture", "schroedinger")
        self.__ode_solver = electronic_parameters.get("ode_solver", "runga_kutta")

        assert (self.rpsh_type in ['bead', 'centroid', 'density_matrix']),\
               ("Ring polymer surface hopping (RPSH) type not available.")
        assert (self.rpsh_rescaling in ['bead', 'centroid']),\
               ("RPSH rescaling type not available.")
        assert (self.rescaling_type in ['nac', 'gradient']),\
               ("Momentum rescaling type not available.")
        assert (self.evolution_picture in ['schroedinger', 'interaction']),\
               ("Evolution picture not available.")
        assert (self.ode_solver in ['runga_kutta', 'unitary', 'scipy']),\
               ("ODE solver not available.")
        if (self.ode_solver == "unitary"):
            assert (self.evolution_picture == "schroedinger"), \
                ("Evolution picture needs to be Schroedinger for unitary propagation.")

        n_states = self.pes.n_states
        max_n_beads = self.pes.max_n_beads

        try:
            initial_state = int(parameters.get("SurfaceHoppingElectrons").get("initial_state"))
            assert ((initial_state >= 0) and (initial_state < n_states)), \
                ("Initial state is either less 0 or exceeds total number of "
                 "states. Given initial state is " + initial_state)
        except (TypeError, ValueError) as e:
            raise type(e)(str(e) + "\nXPACDT: Initial state for surface hopping"
                          "not given or not convertible to int. Given initial"
                          "state is " + parameters.get("SurfaceHoppingElectrons").get("initial_state"))

        self.__current_state = initial_state

        # The order chosen is transposed compared to that in interfaces with
        # n_beads as first axis. This is done for more efficient memory access
        # and matrix multiplications in this module.

        # The wavefunction coefficients are in interaction or Schroedinger
        # picture depending upon 'evolution_picture' selected.
        if (self.rpsh_type == 'density_matrix'):
            self._c_coeff = np.zeros((max_n_beads, n_states), dtype=complex)
            self._D = np.zeros((max_n_beads, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((max_n_beads, n_states, n_states), dtype=complex)
            # Needed for getting proper phase factors integrated over time in interaction picture.
            if (self.evolution_picture == 'interaction'):
                # Phase is integral over time for diff_diag_V term, doesn't include -i/hbar.
                self._phase = np.zeros((max_n_beads, n_states, n_states), dtype=float)
                self._diff_diag_V = np.zeros((max_n_beads, n_states, n_states), dtype=float)
        else:
            self._c_coeff = np.zeros((1, n_states), dtype=complex)
            self._D = np.zeros((1, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((1, n_states, n_states), dtype=complex)
            if (self.evolution_picture == 'interaction'):
                self._phase = np.zeros((1, n_states, n_states), dtype=float)
                self._diff_diag_V = np.zeros((1, n_states, n_states), dtype=float)

        self._c_coeff[:, self.current_state] = 1.0 + 0.0j

        self._old_D = self._get_kinetic_coupling_matrix(R, P)
        self._old_H_e = self._get_H_matrix(R, self._old_D)
        if (self.evolution_picture == 'interaction'):
            self._old_diff_diag_V = self._get_diff_diag_V_matrix(R)

    @property
    def current_state(self):
        """Int : Current electronic state of the system. All beads are in same
        state for now."""
        # TODO: change to list for each bead if rpsh_type = 'individual' added.
        return self.__current_state

    @property
    def masses_nuclei(self):
        """(n_dof) ndarray of floats : The masses of each nuclear degree of
           freedom in au."""
        return self.__masses_nuclei

    @property
    def timestep_scaling_factor(self):
        """float : The timestep for electronic propagation in a.u. """
        # Changed from timestep to factor; should this be allowed to change with a setter?
        return self.__timestep_scaling_factor

    @property
    def rpsh_type(self):
        """{'bead', 'centroid', 'density_matrix'} : String defining type of
        ring polymer surface hopping (RPSH) to be used."""
        # TODO: Possibly add 'individual'; does having individual bead hops make sense?
        return self.__rpsh_type

    @property
    def rpsh_rescaling(self):
        """{'bead', 'centroid'} : String defining type of RPSH rescaling to be
        used; this can be either to conserve bead or centroid energy."""
        return self.__rpsh_rescaling

    @property
    def rescaling_type(self):
        """{'nac', 'gradient'} : String defining type of momentum rescaling to
        be used."""
        # TODO: possibly add 'nac_with_momenta_reversal' if needed
        return self.__rescaling_type

    @property
    def evolution_picture(self):
        """{'schroedinger', 'interaction'} : String defining
        representation/picture for quantum evolution."""
        return self.__evolution_picture

    @property
    def ode_solver(self):
        """{'runga_kutta', 'unitary', 'scipy'} : String defining type of
        velocity rescaling to be used."""
        return self.__ode_solver

    def energy(self, R, centroid=False):
        """Calculate the electronic energy at the current geometry and active
        state as defined by the systems PES. This is a diagonal term in the
        energy matrix.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_beads) ndarray of float /or/ float
            The energy of the systems PES at each bead position or at the
        centroid in hartree.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.adiabatic_energy(R, self.current_state, centroid=centroid)
        else:
            return self.pes.diabatic_energy(R, self.current_state, self.current_state, centroid=centroid)

    def gradient(self, R, centroid=False):
        """Calculate the gradient of the electronic energy at the current
        geometry and active state as defined by the systems PES. This is a
        diagonal term in the gradient matrix.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
            The gradient of the systems PES at each bead position or at the
        centroid in hartree/au.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.adiabatic_gradient(R, self.current_state, centroid=centroid)
        else:
            return self.pes.diabatic_gradient(R, self.current_state, self.current_state, centroid=centroid)

    def _get_velocity(self, P):
        """Obtain nuclear velocities of the system.

        Parameters
        ----------
        P : (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
            The ring-polymer beads or centroid momenta in au.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
            The velocities of the system for each bead position or centroid in au.
        """
        # Is it better to have checks for len(P.shape) == 1 or 2?
        # This works also for centroid but has unnecessary .T, however this is not usually done.
        return (P.T / self.masses_nuclei).T

    def _get_modified_V(self, R):
        """Obtain proper potential energy matrix fit for surface hopping. This
        mostly involves performing a transpose since 'n_beads' is the first
        axis here compared to the interfaces where it is the last. And also
        converting 1D array of adiabatic energy for each bead from interface to
        2D diagonal matrix.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.

        Returns
        -------
        V : (n_beads, n_states, n_states) ndarray of floats if 'rpsh_type' == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if 'rpsh_type' == 'bead' or 'centroid'
            Energy matrix.
        """
        if (self.basis == 'adiabatic'):
            # Create diagonal matrices with 1D array of adiabatic energies.
            if (self.rpsh_type == 'centroid'):
                V = np.array([np.diag(self.pes.adiabatic_energy(R, centroid=True, return_matrix=True))])
            else:
                # (n_states, n_beads).T done for pes.adiabatic_energy!
                V = np.array([np.diag(i) for i in
                              (self.pes.adiabatic_energy(R, centroid=False, return_matrix=True).T)])
        else:
            if (self.rpsh_type == 'centroid'):
                V = np.array([self.pes.diabatic_energy(R, centroid=True, return_matrix=True)])
            else:
                # (n_states, n_states, n_beads) -> (n_beads, n_states, n_states) 
                V = self.pes.diabatic_energy(R, centroid=False, return_matrix=True).transpose(2, 0, 1)

        if (self.rpsh_type == 'bead'):
            V = np.array([np.mean(V, axis=0)])
        return V

    def _get_kinetic_coupling_matrix(self, R, P):
        """Calculate kinetic coupling matrix.
        D_kj = v \\dot d_kj
        TODO: Add equation for each case.
              Also rename it as derivative coupling if needed?

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system in au.

        Returns
        -------
        D : (n_beads, n_states, n_states) ndarray of floats if 'rpsh_type' == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if 'rpsh_type' == 'bead' or 'centroid'
            /or/ None if basis == 'diabatic'
            Kinetic coupling matrix.
        """
        if (self.basis == 'adiabatic'):
            # Here NAC is assumed to be real.
            # Add assert for R and P shapes?
            # nac is (n_states, n_states, n_dof(, n_beads)) ndarrays
            if (self.rpsh_type == 'centroid'):
                v_centroid = self._get_velocity(np.mean(P, axis=1))
                nac = self.pes.nac(R, centroid=True, return_matrix=True)
                D = np.array([np.dot(nac, v_centroid)])
            else:
                # Transposes are done for faster memory accessing by making 'nbeads' first axis.
                vel = self._get_velocity(P).T
                nac = (self.pes.nac(R, centroid=False, return_matrix=True)).transpose(3, 0, 1, 2)

                # D_list here is (n_beads) list of (n_states, n_states)
                D_list = []
                for i, v in enumerate(vel):
                    D_list.append(np.dot(nac[i], v))

                if (self.rpsh_type == 'bead'):
                    D = np.array([np.mean(np.array(D_list), axis=0)])
                else:
                    D = np.array(D_list)
            return D

        else:
            return None

    def _get_H_matrix(self, R, D=None):
        """Calculate total non-adiabatic electronic Hamiltonian.
        The diagonal part is set to 0 in interaction picture.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        D : (n_beads, n_states, n_states) ndarray of floats if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if rpsh_type == 'bead' or 'centroid'
            /or/ None if basis == 'adiabatic'
            Kinetic coupling matrix.

        Returns
        -------
        H : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Total non-adiabatic electronic Hamiltonian.
        """
        V = self._get_modified_V(R)

        if (self.basis == 'adiabatic'):
            H = V - 1j * D
        else:
            # .astype' creates a copy so '.copy()' isn't needed
            H = V.astype(complex)

        if (self.evolution_picture == "interaction"):
            # Should this be done here, or later in the interpolation?
            # This depends upon if we need the Hamiltonian stored.
            for i in H:
                np.fill_diagonal(i, 0.0 + 0.0j)

        return H

    def _get_diff_diag_V_matrix(self, R):
        """Calculate the matrix representing the differences between the
        diagonal energy terms needed to make in interaction picture.
        delta_V_kj = V_jj (columns) - V_kk (rows)

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.

        Returns
        -------
        diff : (n_beads, n_states, n_states) ndarray of floats if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if rpsh_type == 'bead' or 'centroid'
            Diagonal energy difference matrix.
        """
        # Maybe put the assert in the initialization?
        assert (self.evolution_picture == "interaction"), \
               ("Phase in electronic coefficients only makes sense in interaction picture.")

        diff = []
        n_states = self.pes.n_states
        V = self._get_modified_V(R)

        # Is looping over all state indices and using anti-Hermitian property better?
        for i in V:
            # Taking only the diagonal part
            i_diag = np.diag(i)
            diff.append(np.tile(i_diag.reshape(1, -1), (n_states, 1))
                        - np.tile(i_diag.reshape(-1, 1), (1, n_states)))
        return np.array(diff)

    def step(self, R, P, time_propagate, **kwargs):
        """ Advance the electronic subsytem by a given time. This is done here
        by propagating electronic wavefunction coefficients. After that, there
        is a check whether to stay on the current state or change it based on
        these coefficients. This determines the electronic state to propagate
        the nuclei in.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        time_propagate : float
            The time to advance the electrons in au.

        Other Parameters
        ----------------
        step_index : {'before_nuclei', 'after_nuclei'}
            String to determine when electronic step is performed. 'before_nuclei'
            refers to before nuclear step and 'after_nuclei' refers to after.
        """
        # TODO: again too many asserts? Is this needed?
        assert ('step_index' in kwargs), ("Step index not provided for surface"
                                          " hopping propagtion.")

        # Only advance by full time step after the nuclear step.
        if (kwargs.get('step_index') != 'after_nuclei'):
            return

        # TODO: Add option to subtract large diagonal term.
        self._D = self._get_kinetic_coupling_matrix(R, P)
        self._H_e_total = self._get_H_matrix(R, self._D)
        if (self.evolution_picture == 'interaction'):
            self._diff_diag_V = self._get_diff_diag_V_matrix(R)

        # This permits steps not multiple of nuclear step since
        timestep = self.timestep_scaling_factor * time_propagate
        
        # TODO: This seems a bit complicated? Simplify by letting n_steps (= t_prop/t_step = 1/t_scaling) only be int?

        # t and b lists are (n_steps + 1) size
        # Only relative time matters for integration so setting initial t = 0.
        time_array = np.arange(0, time_propagate + 1e-8, timestep)
        # Needed since need to go till 'time_propagate' and since time_rem != time_step
        time_remaining = time_propagate - time_array[-1]

        # Checking if there is any remaining time
        print("Remaining time: ", time_remaining)
        
        # Add remaining time
        if (np.isclose(time_remaining, 0.)):
            pass
        else:
            time_array = np.append(time_array, time_propagate)
        print("Time array: ", time_array)



        # How to get rid of the if statements? Maybe make phase = None if in schroedinger picture?
        if (self.evolution_picture == 'interaction'):
            b_list = [self._get_b_jk(self._c_coeff, self._old_H_e, self._phase)]
        else:
            b_list = [self._get_b_jk(self._c_coeff, self._old_H_e)]

        # Get initial population of current state
        if (self.rpsh_type == 'density_matrix'):
            # Mean is taken instead of sum to get normalizing factor of 1/n_beads
            a_kk_initial = np.mean(np.absolute(self._c_coeff[:, self.current_state]))
        else:
            a_kk_initial = abs(self._c_coeff[0, self.current_state])

        prop_func = getattr(self, "_propagation_equation_" + self.evolution_picture + "_picture")
        ode_solver_function = getattr(self, "_integrator_" + self.ode_solver)
        # Until end-1
        for i, t in enumerate(time_array[:-1]):
            print("time now: ", t)
            t_step = time_array[i+1] - t
            self._c_coeff = ode_solver_function(t, self._c_coeff, t_step, time_propagate, prop_func)

            # How to avoid interpolating again and store at once? For now, this should work.
            t_frac = time_array[i+1] / time_propagate
            H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)

            if (self.evolution_picture == 'interaction'):
                phase = self._phase + 0.5 * time_array[i+1] * (self._old_diff_diag_V
                                                 + self._linear_interpolation_1d(
                                                   t_frac, self._old_diff_diag_V, self._diff_diag_V))
                b_list.append(self._get_b_jk(self._c_coeff, H, phase))
            else:
                b_list.append(self._get_b_jk(self._c_coeff, H))

        # Perform surface hopping if needed.
        self._surface_hopping(R, P, time_array, a_kk_initial, b_list)

        if (self.evolution_picture == 'interaction'):
            # At the end, add to phase (by trapezoidal integration) and update values
            self._phase += (0.5 * time_propagate * (self._old_diff_diag_V + self._diff_diag_V))
            self._old_diff_diag_V = self._diff_diag_V.copy()

        self._old_H_e = self._H_e_total.copy()
        if (self.basis == 'adiabatic'):
            self._old_D = self._D.copy()
        else:
            self._old_D = None

        return

    def _get_a_kj(self, c_coeff, phase=None):
        """ Calculate density matrix for all electronic coefficients. This for
        now is mostly used for testing purposes only and not used in the code.

        TODO: add equation for it. For density_matrix, each bead has its own e- coeff.

        Parameters
        ----------
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        phase : (n_beads, n_states, n_states) ndarray of floats if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if rpsh_type == 'bead' or 'centroid'
            /or/ None if evolution_picture == 'schroedinger'
            Phase of the wavefunction in interaction picture.

        Returns
        -------
        a_jk : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic density matrix. This should be independent of 'evolution_picture' selected.
        """
        if (self.evolution_picture == 'interaction'):
            a_kj = np.array([np.outer(c, np.conj(c)) * np.exp(1j * phase[i])
                             for i, c in enumerate(c_coeff)])
        else:
            a_kj = np.array([np.outer(c, np.conj(c)) for c in c_coeff])

        return a_kj

    def _get_b_jk(self, c, H, phase=None):
        """ Calculate b_jk  where k is the current state and j is any other
        state except k, as defined in J. Chem. Phys. 93, 1061 (1990).

        TODO : add an equation for it. For density_matrix, it is averaged.
        b_jk = 2/ hbar Im(a_kj V_jk) - 2 Re(a_kj \\dot{R} d_kj)
        where k is the current state and a_kj is the density matrix.

        Parameters
        ----------
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        H : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Total non-adiabatic electronic Hamiltonian.
        phase : (n_beads, n_states, n_states) ndarray of floats if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of floats if rpsh_type == 'bead' or 'centroid'
            /or/ None if evolution_picture == 'schroedinger'
            Phase of the wavefunction in interaction picture.

        Returns
        -------
        b_jk : (n_states) ndarray of floats
            b_jk array
        """
        # Transposes need to be done for proper broadcasting to give a_kj of
        # shape (1 or max_n_beads, n_states).
        if (self.evolution_picture == 'interaction'):
            a_kj = (c[:, self.current_state] * (np.conj(c) * np.exp(1j * phase[:, self.current_state, :])).T).T
        else:
            a_kj = (c[:, self.current_state] * np.conj(c).T).T

        # b_kk = 0 and D_jk or V_jk should be (1 or n_beads, n_states)
        if (self.basis == 'adiabatic'):
            # Multiply by -1 to get rid of -ve sign from H.imag = -D
            D_jk = -1. * H[:, :, self.current_state].imag
            b_jk = -2. * (a_kj * D_jk).real
        else:
            # For interaction picture, V_kk = 0; but that shouldn't matter
            # since b_kk is also 0.
            V_jk = H[:, :, self.current_state]
            b_jk = 2. * (a_kj * V_jk).imag

        if (self.rpsh_type == 'density_matrix'):
            b_jk = np.mean(b_jk, axis=0)
        else:
            b_jk = b_jk[0].copy()

        return b_jk

    # TODO : Should these electronic coefficint integrators be put in its own
    # module? This would be needed for Ehrenfest as well if we implement that.
    def _integrator_runga_kutta(self, time, c_coeff, t_step, time_propagate, prop_func):
        """ Propagate the electronic coefficients by one electronic time step
        using the classic 4th order Runga-Kutta method.
        Reference: http://mathworld.wolfram.com/Runge-KuttaMethod.html

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients. For
            now it refers to propagation in interaction or Schroedinger picture.

        Returns
        -------
        c_new : n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            New electronic coefficients obtained after propagation.
        """
        # Should rk4 calculate and return all of the interpolated values???
        #t_step = self.timestep_scaling_factor * time_propagate
        c_1 = t_step * prop_func(time, c_coeff, time_propagate)
        c_2 = t_step * prop_func((time + 0.5 * t_step), c_coeff + 0.5 * c_1, time_propagate)
        c_3 = t_step * prop_func((time + 0.5 * t_step), c_coeff + 0.5 * c_2, time_propagate)
        c_4 = t_step * prop_func((time + t_step), c_coeff + c_3, time_propagate)

        c_new = c_coeff + 1.0 / 6.0 * (c_1 + c_4) + 1.0 / 3.0 * (c_2 + c_3)

        # Normalizing by hand to conserve norm.
        for i in c_new:
            normalization = np.linalg.norm(i)
            i /= normalization

        return c_new

    def _integrator_scipy(self, time, c_coeff, time_propagate, prop_func):
        """ Propagate the electronic coefficients by electronic time step using
        ode integrator from scipy.

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients. For
            now it refers to propagation in interaction or Schroedinger picture.

        Returns
        -------
        c_new : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            New electronic coefficients obtained after propagation.
        """
        c_new = []
        timestep = self.timestep_scaling_factor * time_propagate

        # TODO: How to choose the method?
        solver = ode(prop_func).set_integrator('zvode', first_step=timestep, method='bdf')

        for i, c in enumerate(c_coeff):
            solver.set_f_params(time_propagate, i)
            solver.set_initial_value(c, time)
            while solver.successful() and solver.t < (time + timestep):
                solver.integrate(solver.t + timestep)

            # Normalizing by hand to conserve norm.
            normalization = np.linalg.norm(solver.y)
            c_new.append(solver.y / normalization)

        return np.array(c_new)

    def _integrator_unitary(self, time, c_coeff, time_propagate, prop_func=None):
        """ Propagate the electronic coefficients by electronic time step using
        unitary evolution matrix at the midpoint. This is a 3rd order method.
        TODO : Add reference to it?

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients.
            This is not required for this method so default is None.

        Returns
        -------
        c_new : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            New electronic coefficients obtained after propagation.
        """
        t_step = self.timestep_scaling_factor * time_propagate
        t_frac_mid = (time + 0.5 * t_step) / time_propagate

        H_mid = self._linear_interpolation_1d(t_frac_mid, self._old_H_e, self._H_e_total)
        propagator = np.array([expm(-1j * h * t_step) for h in H_mid])

        c_new = np.matmul(propagator, np.expand_dims(c_coeff, axis=-1)).reshape(-1, self.pes.n_states)

        return c_new
    
    def _propagation_equation_schroedinger_picture(self, t, c, t_prop, i=None):
        """ Propagation equation for the electronic coefficients in
        Schroedinger picture.
        TODO: add equation
              Also in matrix form: -i/hbar np.matmul(H, c)

        Parameters
        ----------
        t : float
            Current time in this propagation in au.
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
            /or/ (n_states) ndarray of complex if ode_solver == 'scipy'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        t_prop : float
            Total time to advance the electrons in au.
        i : int
            Index for electronic coefficients; needed if ode_solver == 'scipy'.

        Returns
        -------
        (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
        /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
        /or/ (n_states) ndarray of complex if ode_solver == 'scipy'
            Time derivative of electronic coefficients at time `t`.
        """
        # Time fraction in the propagation step, needed for interpolation.
        t_frac = t / t_prop
        if i is None:
            H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)
            return (-1.0j * np.matmul(H, np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            H = self._linear_interpolation_1d(t_frac, self._old_H_e[i], self._H_e_total[i])
            return (-1.0j * np.matmul(H, c))

    def _propagation_equation_interaction_picture(self, t, c, t_prop, i=None):
        """Propagation equation for the electronic coefficients in
        interaction picture.
        TODO: add equation
              Also in matrix form: -i/hbar np.matmul(H * phase, c)

        Parameters
        ----------
        t : float
            Current time in this propagation in au.
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
            /or/ (n_states) ndarray of complex if ode_solver == 'scipy'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        t_prop : float
            Total time to advance the electrons in au.
        i : int
            Index for electronic coefficients; needed if ode_solver == 'scipy'.

        Returns
        -------
        (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
        /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
        /or/ (n_states) ndarray of complex if ode_solver == 'scipy'
            Time derivative of electronic coefficients at time `t`.
        """
        # Time fraction in the propagation step, needed for interpolation.
        t_frac = t / t_prop

        if i is None:
            H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)
            phase = self._phase + 0.5 * t * (self._old_diff_diag_V + 
                                             self._linear_interpolation_1d(t_frac, self._old_diff_diag_V, self._diff_diag_V))
            # 'np.expand_dims' used to add a dimension for proper matrix multiplication.
            return (-1.0j * np.matmul((H * np.exp(-1.0j * phase)), np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            H = self._linear_interpolation_1d(t_frac, self._old_H_e[i], self._H_e_total[i])
            phase = self._phase[i] + 0.5 * t * (self._old_diff_diag_V[i] + 
                                             self._linear_interpolation_1d(t_frac, self._old_diff_diag_V[i], self._diff_diag_V[i]))
            return (-1.0j * np.matmul((H * np.exp(-1.0j * phase)), c))

    # TODO : Maybe put this also in math module
    def _linear_interpolation_1d(self, x_fraction, y_initial, y_final):
        """ Performs linear interpolation in 1 dimension to obtain y value at
        `x_fraction` between the intial and final x values.
        TODO: write properly: x_fraction = (x_required - x_initial) / (x_final - x_initial)

        Parameters
        ----------
        x_fraction : float
            x fraction.
        y_initial : ndarray
            Initial y value. Can be any dimensional ndarray.
        y_final : ndarray
            Final y value. Same shape as y_initial.

        Returns
        -------
        ndarray of floats (same as y_initial)
            Interpolated value at `x_fraction` between the intial and final values.
        """
        print(x_fraction)
        # Again too many asserts?
        assert (x_fraction <= 1.), \
            ("x fraction is larger than 1 in linear interpolation.")
        return ((1. - x_fraction) * y_initial + x_fraction * y_final)

    def _surface_hopping(self, R, P, time_array, a_kk_initial, b_list):
        """ Perform surface hopping if required by comparing the probability
        to hop with a random number from a uniform distribution between 0 and 1.
        If there is a chance to hop, then check for energy conservation and
        rescale momenta accordingly if needed.

        TODO: add equation for probability with integration.
        Reference: J. Chem. Phys. 101 (6), 4657 (1994)

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        time_array : (n_steps + 1) ndarray of floats
             Times in au to which the electrons were propagated.
        a_kk_initial : float
            Probability density of current state initially before the propagation.
        b_list : (n_steps + 1) list of (n_states) ndarray of floats
            List of b_jk at times given in `t_list`.
        """
        # Perform trapezoidal integration to get hopping probability for each
        # state, so 'prob' is (n_state) ndarray
        prob = np.trapz(np.array(b_list), np.array(time_array), axis=0) / a_kk_initial

        # Setting negative probability and hopping to current state to 0.
        # TODO : have cutoff for small probabilities! (determine value!), also check if total prob exceeds 1.
        prob[prob < 0] = 0.
        prob[self.current_state] = 0.
        
#        for i in range(len(prob)):
#            if (prob[i] < 0. or i == self.current_state):
#                prob[i] = 0.

        # Switch state if needed by comparing to random number.
        new_state = None
        sum_prob = 0.
        rand_num = random.random()

        for i, p in enumerate(prob):
            if (rand_num >= sum_prob and rand_num < (sum_prob + p)):
                new_state = i
                break
            sum_prob += p

        # If there is a new state to switch to, check if there is enough energy
        # for the hop.
        if (new_state is not None):
            should_change = self._momentum_rescaling(R, P, new_state)
            if should_change:
                self.__current_state = new_state

        return

    def _momentum_rescaling(self, R, P, new_state):
        """ Check if there is enough energy to change state; if yes, then
        calculate the change in momentum to conserve total energy of centroid
        or ring polymer.

        TODO: add equation for different momentum rescaling.

        Reference: J. Chem. Phys. 101 (6), 4657 (1994)

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        new_state : int
            New state to switch to.

        Returns
        -------
        bool
            True if there is enough energy to change state (momentum rescaling
            is done in this case), False if not.
        """
        if (self.rpsh_rescaling == 'centroid'):
            centroid = True
        else:
            centroid = False

        # Obtain change in potential for each bead or centroid to give
        # (nbeads) ndarray of floats or a float value
        if (self.basis == 'adiabatic'):
            diff_V = self.pes.adiabatic_energy(R, self.current_state, centroid=centroid) \
                    - self.pes.adiabatic_energy(R, new_state, centroid=centroid)
        else:
            diff_V = self.pes.diabatic_energy(R, self.current_state, self.current_state, centroid=centroid) \
                    - self.pes.diabatic_energy(R, new_state, new_state, centroid=centroid)

        # Obtain projection vector for each dof and beads or centroid, which
        # gives (ndof, nbeads) or (ndof) array
        if (self.rescaling_type == 'nac'):
            proj_vec = self.pes.nac(R, self.current_state, new_state, centroid=centroid)
        elif (self.rescaling_type == 'gradient'):
            # Get gradient differences
            if (self.basis == 'adiabatic'):
                proj_vec = self.pes.adiabatic_gradient(R, self.current_state, centroid=centroid) \
                          - self.pes.adiabatic_gradient(R, new_state, centroid=centroid)
            else:
                proj_vec = self.pes.diabatic_gradient(R, self.current_state, self.current_state, centroid=centroid) \
                          - self.pes.diabatic_gradient(R, new_state, new_state, centroid=centroid)

        # Need inv_mass as a property? Since mult is faster than division.
        if (self.rpsh_rescaling == 'centroid'):
            # Conserve H_centroid
            A_kj = 0.5 * np.sum(proj_vec * proj_vec / self.masses_nuclei)
            v_centroid = self._get_velocity(np.mean(P, axis=1))
            B_kj = np.dot(v_centroid, proj_vec)
        else:
            # Conserve H_n of ring polymer.
            # First sum over beads and then divide each dof by its mass and
            # finally sum over dof.
            A_kj = 0.5 * np.sum(np.sum(proj_vec * proj_vec, axis=1) / self.masses_nuclei)
            B_kj = np.sum(np.sum(P * proj_vec, axis=1) / self.masses_nuclei)
            diff_V = np.sum(diff_V)

        # Check if there is enough energy to hop
        inside_root = B_kj * B_kj + 4 * A_kj * diff_V

        if (inside_root < 0.):
            # Not enough energy
            return False
        else:
            # Take the least possible change
            if (B_kj < 0.):
                factor = (B_kj + math.sqrt(inside_root)) / (2. * A_kj)
            else:
                factor = (B_kj - math.sqrt(inside_root)) / (2. * A_kj)

            # Changing P here should also change in nuclei.
            for i, p_A in enumerate(P):
                p_A -= factor * proj_vec[i]

            return True
