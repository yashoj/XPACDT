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

        # TODO: decide what should be the default, for nb=1 does that shouldn't matter though
        self.rpsh_type = electronic_parameters.get("rpsh_type", "bead")
        self.rpsh_rescaling = electronic_parameters.get("rpsh_rescaling", "bead")
        self.rescaling_type = electronic_parameters.get("rescaling_type", "nac")

        self.__masses_nuclei = masses_nuclei
        # Add try and exception here for the factor?
        timestep_scaling_factor = float(electronic_parameters.get("timestep_scaling_factor"))
        nuclear_timestep = units.parse_time(parameters.get("nuclei_propagator").get("timestep"))
        self.__timestep = timestep_scaling_factor * nuclear_timestep

        self.evolution_picture = electronic_parameters.get("evolution_picture", "schroedinger")
        self.ode_solver = electronic_parameters.get("ode_solver", "rk4")
        if (self.ode_solver == "unitary"):
            assert (self.evolution_picture == "schroedinger"), \
                ("Evolution picture needs to be Schroedinger for unitary propagation.")

        # print("Representation is ", self.evolution_picture)
        # print("ODE solver is ", self.ode_solver)

        n_states = self.pes.n_states
        max_n_beads = self.pes.max_n_beads

        try:
            initial_state = int(parameters.get("SurfaceHoppingElectrons").get("initial_state"))
            assert ((initial_state >= 0) and (initial_state < n_states)), \
                ("Initial state is either less 0 or exceeds total number of "
                 "states")
        except (TypeError, ValueError):
            print("Initial state for surface hopping not given or not "
                  "convertible to int. Setting to default value: 0")
            initial_state = 0

        self.current_state = initial_state

        # Again are these initializations needed? Also add density matrix?
        # The order chosen is opposite of what is obtained from the interfaces
        # for more efficient memory access and matrix multiplications.

        # The wavefunction coefficients are in interaction or Schroedinger
        # picture depending upon 'evolution_picture' selected.
        if (self.rpsh_type == 'density_matrix'):
            self._c_coeff = np.zeros((max_n_beads, n_states), dtype=complex)
            self._D = np.zeros((max_n_beads, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((max_n_beads, n_states, n_states), dtype=complex)
            # Needed for getting proper phase in interaction picture.
            if (self.evolution_picture == 'interaction'):
                # Integral over diff_diag_V term, doesn't include -i/hbar
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

    # Is this setter needed?
    @current_state.setter
    def current_state(self, s):
        self.__current_state = s

    @property
    def masses_nuclei(self):
        """(n_dof) ndarray of floats : The masses of each nuclear degree of
           freedom in au."""
        return self.__masses_nuclei

    @property
    def timestep(self):
        """float : The timestep for electronic propagation in a.u. """
        return self.__timestep

    @property
    def rpsh_type(self):
        """{'bead', 'centroid', 'density_matrix'} : String defining type of
        ring polymer surface hopping (RPSH) to be used."""
        # TODO: Possibly add 'individual'; does having individual bead hops make sense?
        return self.__rpsh_type

    @rpsh_type.setter
    def rpsh_type(self, r):
        assert (r in ['bead', 'centroid', 'density_matrix']),\
               ("Ring polymer surface hopping (RPSH) type not available.")
        self.__rpsh_type = r

    @property
    def rpsh_rescaling(self):
        """{'bead', 'centroid'} : String defining type of RPSH rescaling to be
        used; this can be either to conserve bead or centroid energy."""
        return self.__rpsh_rescaling

    @rpsh_rescaling.setter
    def rpsh_rescaling(self, r):
        assert (r in ['bead', 'centroid']),\
               ("RPSH rescaling type not available.")
        self.__rpsh_rescaling = r

    @property
    def rescaling_type(self):
        """{'nac', 'gradient'} : String defining type of momentum rescaling to
        be used."""
        # TODO: possibly add 'nac_with_momenta_reversal' if needed
        return self.__rescaling_type

    @rescaling_type.setter
    def rescaling_type(self, r):
        assert (r in ['nac', 'gradient']),\
               ("Momentum rescaling type not available.")
        self.__rescaling_type = r

    @property
    def evolution_picture(self):
        """{'schroedinger', 'interaction'} : String defining
        representation/picture for quantum evolution."""
        return self.__evolution_picture

    @evolution_picture.setter
    def evolution_picture(self, p):
        assert (p in ['schroedinger', 'interaction']),\
               ("Evolution picture not available.")
        self.__evolution_picture = p

    @property
    def ode_solver(self):
        """{'rk4', 'unitary', 'scipy_solver'} : String defining type of
        velocity rescaling to be used."""
        return self.__ode_solver

    @ode_solver.setter
    def ode_solver(self, s):
        assert (s in ['runga_kutta', 'unitary', 'scipy']),\
               ("ODE solver not available.")
        self.__ode_solver = s

    def energy(self, R, centroid=False):
        """Return the electronic energy at the current geometry and active
        state as defined by the systems PES. This is a diagonal term in the
        energy matrix.shape

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
        """Obtain proper potential energies fit for surface hopping.

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
                # (n_states, n_beads).T done for pes.energy!
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
              Also rename it as derivative coupling if needed.

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
            Kinetic coupling matrix. (Optional: only needed when 'basis' is adiabatic.)

        Returns
        -------
        H : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Total non-adiabatic electronic Hamiltonian.
        """
        if (self.basis == 'adiabatic'):
            # Is this assert needed as this function is only accessed internally?
            assert (D is not None), ("Kinetic coupling matrix not provided in adiabatic basis.")

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
        step_index : {'first', last'}
            String to determine when electronic step is performed. 'first'
            refers to before nuclear step and 'last' refers to after.
        """
        # TODO: again too many asserts? Is this needed?
        assert ('step_index' in kwargs), ("Step index not provided for surface"
                                          " hopping propagtion.")

        # Only advance by full time step after the nuclear step.
        if (kwargs.get('step_index') != 'last'):
            return

        # TODO: Add option to subtract large diagonal term
        # Need phase matrix, and getting the density matrix back
        # Solve coefficient eq using rk4 or scipy solve_ivp or scipy ode?
        # Get phase matrix and propagate that
        self._D = self._get_kinetic_coupling_matrix(R, P)
        self._H_e_total = self._get_H_matrix(R, self._D)
        if (self.evolution_picture == 'interaction'):
            self._diff_diag_V = self._get_diff_diag_V_matrix(R)

        # For now consider only multiple of nuclear steps, hence also imposing
        # that output times be multiple of nucelar step
        # TODO : add not multiple steps if possible.
        n_steps = int((time_propagate + 1e-8) // self.timestep)

        # t and b lists are (n_steps + 1) size
        # Only relative time matters for integration so setting initial t = 0.
        t = 0
        t_list = [t]

        # 'phase' is not modified yet, how to get rid of the if statements?
        # maybe make phase = None if in schroedinger picture?
        if (self.evolution_picture == 'interaction'):
            b_list = [self._get_b_jk(self._c_coeff, self._old_H_e, self._phase)]
            #self._print_a_kj(self._c_coeff, self._phase)
        else:
            b_list = [self._get_b_jk(self._c_coeff, self._old_H_e)]
            #self._print_a_kj(self._c_coeff)
        
        # c_coeffs are (max_n_beads or 1, n_states), phases cancel out for populations
        if (self.rpsh_type == 'density_matrix'):
            # Mean is taken instead of sum to get normalizing factor of 1/n_beads
            a_kk = np.mean(np.absolute(self._c_coeff[:, self.current_state]))
        else:
            a_kk = abs(self._c_coeff[0, self.current_state])

        # Using for loop has better accuracy than while loop with adding step, right?
        prop_func = getattr(self, "_propagation_equation_" + self.evolution_picture + "_picture")
        ode_solver_function = getattr(self, "_integrator_" + self.ode_solver)
        
        for i in range(n_steps):
#            if (self.ode_solver == 'rk4'):
#                if (self.evolution_picture == 'interaction'):
#                    prop_func = self._propagation_equation_interaction_picture
#                elif (self.evolution_picture == 'schroedinger'):
#                    prop_func = self._propagation_equation_schroedinger_picture
#
#                self._c_coeff = self._integrator_runga_kutta(t, self._c_coeff,
#                                                             time_propagate, prop_func)
#
#            elif (self.ode_solver == 'unitary'):
#                self._c_coeff = self._integrator_unitary(t, time_propagate, self._c_coeff)
#
#            elif (self.ode_solver == 'scipy_solver'):
#                self._c_coeff = self._integrator_scipy()
                
            # Need to change some of the names of ode solvers!!!
                
            
            self._c_coeff = ode_solver_function(t, self._c_coeff, time_propagate, prop_func)

#            print('c_coeff=', self._c_coeff)
#            if (self.evolution_picture == 'interaction'):
#                self._print_a_kj(self._c_coeff, self._phase)
#            else:
#                self._print_a_kj(self._c_coeff)
            
            t = (i + 1) * self.timestep

            t_list.append(t)
            # How to avoid interpolating again and store at once? For now, this should work.
            t_frac = t / time_propagate
            H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)

            if (self.evolution_picture == 'interaction'):
                phase = self._phase + 0.5 * t * (self._old_diff_diag_V
                                                 + self._linear_interpolation_1d(
                                                         t_frac, self._old_diff_diag_V, self._diff_diag_V))
                b_list.append(self._get_b_jk(self._c_coeff, H, phase))
            else:
                b_list.append(self._get_b_jk(self._c_coeff, H))
                
        # Perform Surface hops, for this need to get integrated b_jk for which we need list of c_coeff and H at those steps
        # However only need to calculate prob from current state to all other states.
        # !!! maybe better to give b_jk and a_jk??
        self._surface_hopping(R, P, t_list, a_kk, b_list)
        
        if (self.evolution_picture == 'interaction'):
            # At the end, add to phase (by trapezoidal integration) and update values
            self._phase += (0.5 * time_propagate * (self._old_diff_diag_V + self._diff_diag_V))
            self._old_diff_diag_V = self._diff_diag_V.copy()
            
            #self._print_a_kj(self._c_coeff, self._phase)
        else:
            #self._print_a_kj(self._c_coeff)
            pass
        
        self._old_H_e = self._H_e_total.copy()
        if (self.basis == 'adiabatic'):
            self._old_D = self._D.copy()
        else:
            self._old_D = None
        

        return

    def _get_b_jk(self, c, H, phase=None):
        """ Calculate b_jk

        Parameters
        ----------
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        H : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Total non-adiabatic electronic Hamiltonian.
        phase : (n_beads, n_states, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            /or/ None if evolution_picture == 'schroedinger'
            Phase of the wavefunction in interaction picture.

        Returns
        -------
        b_jk : (n_states) ndarray of floats
            b_jk array...
        """

        # Calculating b_jk where k is the current state and j is any other state except k.
        # Phases are (1 or max_n_beads, n_states, n_states)
        # a_kj should be (1 or max_n_beads, n_states)
        if (self.evolution_picture == 'interaction'):
            # Is this assert needed as this function is only accessed internally?
            assert (phase is not None), ("Phase matrix not provided for interaction picture.")
            
        # Transposes need to be done for proper broadcasting.
        if (self.evolution_picture == 'interaction'):
            a_kj = (c[:, self.current_state] * (np.conj(c) * np.exp(1j * phase[:, self.current_state, :])).T).T
        else:
            a_kj = (c[:, self.current_state] * np.conj(c).T).T
        
        #print(a_kj)

        # H is (1 or n_beads, n_states, n_states)
        # and D or V_jk should be (1 or n_beads, n_states)
        # b_kk will be 0.
        if (self.basis == 'adiabatic'):
            # Multiply by -1 to get rid of -ve sign from H.imag = -D
            D_jk = -1. * H[:, :, self.current_state].imag
            b_jk = -2. * (a_kj * D_jk).real
        else:
            # For interaction picture, V_kk = 0; but that shouldn't matter
            # since b_kk is also 0.
            V_jk = H[:, :, self.current_state]
            b_jk = 2. * (a_kj * V_jk).imag

        # b_jk should be (n_states) array
        if (self.rpsh_type == 'density_matrix'):
            b_jk = np.mean(b_jk, axis=0)
        else:
            b_jk = b_jk[0].copy()
        return b_jk

    # TODO : Should this be put in its own module? This would be needed by Ehrenfest as well if we implement that.
    def _integrator_runga_kutta(self, t, c, time_propagate, prop_func):
        # Classical Runga-Kutta RK4 algorithm for electronic quantum coefficients
        # Should rk4 calculate and return all of the interpolated values???
        # c_coeffs are (max_n_beads or 1, n_states)

        # TODO : Normalize the wavefunction in the end (not done in CDTK though),
        # or do unitary dynamics using midpoint H.
        t_step = self.timestep
        c_1 = t_step * prop_func(t, c, time_propagate)
        c_2 = t_step * prop_func((t + 0.5 * t_step), c + 0.5 * c_1, time_propagate)
        c_3 = t_step * prop_func((t + 0.5 * t_step), c + 0.5 * c_2, time_propagate)
        c_4 = t_step * prop_func((t + t_step), c + c_3, time_propagate)

        c_new = c + 1.0 / 6.0 * (c_1 + c_4) + 1.0 / 3.0 * (c_2 + c_3)

        # Normalizing by hand to conserve norm.
        for i in c_new:
            normalization = np.linalg.norm(i)
            i /= normalization

        return c_new

    # Get wrapper function to give interpolated H and use with scipy
    def _integrator_scipy(self, t, c_coeff, time_propagate, prop_func):
        r = ode(prop_func).set_integrator('zvode', first_step=self.timestep, method='bdf')
        c_new = []

        for i, c in enumerate(c_coeff):
            #print(c_i.shape, c_i)
            r.set_f_params(time_propagate, i)
            r.set_initial_value(c, t)
            while r.successful() and r.t < (t + self.timestep):
                r.integrate(r.t + self.timestep)
            c_new_i = r.y
            # Normalizing by hand to conserve norm.
            normalization = np.linalg.norm(c_new_i)
            c_new_i /= normalization
            c_new.append(c_new_i)

        return np.array(c_new)

    def _integrator_unitary(self, t, c, time_propagate, prop_func=None):
        
        # unitary dynamics using midpoint H. Error in the order of dt^3
        t_step = self.timestep
        t_frac_mid = (t + 0.5 * t_step) / time_propagate

        H_mid = self._linear_interpolation_1d(t_frac_mid, self._old_H_e, self._H_e_total)
        propagator = np.array([expm(-1j * h * t_step) for h in H_mid])
        
        c_new = np.matmul(propagator, np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states)

        return c_new

    def _propagation_equation_interaction_picture(self, t, c, t_prop, i=None):
        """Propagation equation in interaction picture.

        Parameters
        ----------
        c : (n_beads, n_states) ndarray of complex if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.

        Returns
        -------
        (n_beads, n_states) ndarray of floats
        /or/ (1, n_states) ndarray of floats
            New electronic wavefuntion coefficients after propagation.
        """
        # Matrix product of -i/hbar np.matmul(H, np.matmul(phase, c))

        # Perform linear interoplation using old_H and H_e
        # Is this better or to use scipy interp1d function?
        t_frac = t / t_prop
        H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)

        # Trapeziodal integration scheme: add 1/2 * dt * (y_ini + y_req)
        # This is better due to linear interpolation
        # changing self.phase itself might be dangerous due to intermediate steps in integration
        phase = self._phase + 0.5 * t * (self._old_diff_diag_V + 
                                         self._linear_interpolation_1d(t_frac, self._old_diff_diag_V, self._diff_diag_V))
        #print('\n Phase =', phase)
        if i is None:
            # 'np.expand_dims' used to add a dimension for proper matrix multiplication
            #return (-1.0j * np.matmul(H, np.matmul(np.exp(-1.0j * phase),
            #                                   np.expand_dims(c, axis=-1))).reshape(-1, self.pes.n_states))
            return (-1.0j * np.matmul((H * np.exp(-1.0j * phase)), np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            # return (-1.0j * np.matmul(H[i], np.matmul(np.exp(-1.0j * phase[i]), c)))
            return (-1.0j * np.matmul((H[i] * np.exp(-1.0j * phase[i])), c))

    def _propagation_equation_schroedinger_picture(self, t, c, t_prop, i=None):
        # Matrix product of -i/hbar np.matmul(H, c)
        t_frac = t / t_prop
        H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)
        if i is None:
            return (-1.0j * np.matmul(H, np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            return (-1.0j * np.matmul(H[i], c))

    # TODO : Maybe put this also in math module
    def _linear_interpolation_1d(self, x_fraction, y_initial, y_final):
        # x_fraction = (x_required - x_initial) / (x_final - x_initial)
        # y can be any dimension array, x_fraction is float
        # Need assert for shapes of y_initial and y_final??
        # Assert for x_frac being less than 1?
        return ((1. - x_fraction) * y_initial + x_fraction * y_final)

    def _surface_hopping(self, R, P, t_list, a_kk, b_list):
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
        time_list : float
            The time to advance the electrons in au.
        a_kk : float
            Probability density at current state initially.
        """
        # Perform surface hopping if required
        # Here t_list is (n_steps + 1) list, b_list is (n_steps + 1) list of (n_states) array
        # a_kk is a float

        # Integrate to get b_jk
        # Get prob to hop, set to 0 if less than zero
        
        # g_kj (prob) with g_kk = 0 so prob is a (Nstate) array
        prob = np.trapz(np.array(b_list), np.array(t_list), axis=0) / a_kk

        # Setting negative probability and probability to hop to current state to 0.
        # TODO : have cutoff for small probabilities! (determine value!)
        for i in range(len(prob)):
            if (prob[i] < 0. or i == self.current_state):
                prob[i] = 0.

        # State switch if needed, have another function for momentum rescaling
        new_state = None
        rand_num = random.random()

        print("Prob = ", prob)

        sum_prob = 0.
        for i, p in enumerate(prob):
            if (rand_num >= sum_prob and rand_num < (sum_prob + p)):
                new_state = i
                break
            sum_prob += p

        if (new_state is not None):
            should_change = self._momentum_rescaling(R, P, new_state)
            if should_change:
                self.current_state = new_state

        return

    def _momentum_rescaling(self, R, P, new_state):
        """ Check if there is enough energy to change state; if yes, then
        calculate the change in momentum due to energy conservation.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        new_state : int
            New state

        Returns
        -------
        bool
            True if there is enough energy to change state, False if not.
        """
        # Perform momentum rescaling if needed.
        # Need to check energy conservaton and return whether to change v or not
        # Need to return true or false for changing state? or do that already here
        # How to give back momenta? Does changing it here also change in nuclei??
        
        if (self.rpsh_rescaling == 'centroid'):
            centroid = True
        else:
            centroid = False

        # Obtain change in PE for each bead or centroid, i.e. (nbeads) array or float
        if (self.basis == 'adiabatic'):
            diff_V = self.pes.adiabatic_energy(R, self.current_state, centroid=centroid) \
                    - self.pes.adiabatic_energy(R, new_state, centroid=centroid)
        else:
            diff_V = self.pes.diabatic_energy(R, self.current_state, self.current_state, centroid=centroid) \
                    - self.pes.diabatic_energy(R, new_state, new_state, centroid=centroid)

        # Obtain projection vector for each dof and beads or centroid, i.e.
        # (ndof, nbeads) or (ndof) array
        if (self.rescaling_type == 'nac'):
            proj_vec = self.pes.nac(R, self.current_state, new_state, centroid=centroid)
        else:
            # Get gradient differences
            if (self.basis == 'adiabatic'):
                proj_vec = self.pes.adiabatic_gradient(R, self.current_state, centroid=centroid) \
                          - self.pes.adiabatic_gradient(R, new_state, centroid=centroid)
            else:
                proj_vec = self.pes.diabatic_gradient(R, self.current_state, self.current_state, centroid=centroid) \
                          - self.pes.diabatic_gradient(R, new_state, new_state, centroid=centroid)

        # Need inv_mass as a property? Since mult is faster than division.
        if (self.rpsh_rescaling == 'centroid'):
            # Use centroid V to conserve H_centroid
            A_kj = np.dot(proj_vec, proj_vec) / self.masses_nuclei
            v_centroid = self._get_velocity(np.mean(P, axis=1))
            B_kj = np.dot(v_centroid, proj_vec)
        else:
            # Use sum of V to conserve Hn
            # Loop over ndof
            A_kj = 0.
            B_kj = 0.
            for i, vec in enumerate(proj_vec):
                A_kj += (np.dot(vec, vec) / (2. * self.masses_nuclei[i]))
                B_kj += (np.dot(P[i], vec) / self.masses_nuclei[i])
            diff_V = np.sum(diff_V)

        # Check if enough energy for hop
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

            # Does changing P here, also change for nuclei????
            # Should work for both centroid and bead rescaling?
            for i, p_A in enumerate(P):
                p_A -= factor * proj_vec[i]

            return True
    
    def _print_a_kj(self, c_coeff, phase=None):
        # Print the density matrix of first bead in schroedinger representation
        # c is (max_n_beads or 1, n_states) and phase is (1 or max_n_beads, n_states, n_states)
        # a_kj should be (max_n_beads or 1, n_states, n_states)
        if (self.evolution_picture == 'interaction'):
            for i, c in enumerate(c_coeff):
                a_kj = np.array([np.outer(c, np.conj(c)) * np.exp(1j * phase[i]) for i, c in enumerate(c_coeff)])
        else:
            a_kj = np.array([np.outer(c, np.conj(c)) for c in c_coeff])
            
        print('Printing density matrix')
        
        print(a_kj, '\n')
        
        # Print the trace instead
        #for a in a_kj:
        #    print(np.trace(np.absolute((a))))
        return


if __name__ == '__main__':
    import sys
    import XPACDT.Input.Inputfile as infile

    parameters = infile.Inputfile("SH_test_input.in")

    sh_e = SurfaceHoppingElectrons(parameters, parameters.n_beads)
    time_propagate = sh_e.timestep * 1000.
    #sh_e.evolution_picture = "interaction"
    #sh_e.evolution_picture = "schroedinger"
    
    R = np.array([[3., 3.5]])
    P = np.array([[0.002, 0.0025]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print("Current state is: ", sh_e.current_state)
    #sys.exit()
    
    
    R = np.array([[3.1, 3.6]])
    P = np.array([[0.0021, 0.0026]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print("Current state is: ", sh_e.current_state)
    
    R = np.array([[3.2, 3.7]])
    P = np.array([[0.0022, 0.0027]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print("Current state is: ", sh_e.current_state)
    
    R = np.array([[3.3, 3.8]])
    P = np.array([[0.0023, 0.0028]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print("Current state is: ", sh_e.current_state)
    
    sys.exit()
    
    R = np.array([[3.4]])
    P = np.array([[0.0024]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print(sh_e.current_state)
    
    R = np.array([[3.5]])
    P = np.array([[0.0025]])
    sh_e.step(R, P, time_propagate, **{'step_index': 'last'})
    print(sh_e.current_state)

