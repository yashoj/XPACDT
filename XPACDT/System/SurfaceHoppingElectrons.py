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

"""This is an implementation of fewest switches surface hopping (FSSH) for
electronic dynamics, extended to ring polymer molecular dynamics to give
ring polymer surface hopping (RPSH)."""

import math
import numpy as np
import random
from scipy.linalg import expm
from scipy.integrate import ode

import XPACDT.System.Electrons as electrons
import XPACDT.Tools.MathematicalTools as mtools


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
    masses_nuclei : (n_dof) ndarray of floats
        The masses of each nuclear degree of freedom in au.
    R, P : (n_dof, n_beads) ndarray of floats
        The (ring-polymer) positions `R` and momenta `P` representing the
        system nuclei in au.

    Attributes
    ----------
    masses_nuclei
    n_steps
    rpsh_type
    rpsh_rescaling
    rescaling_type
    evolution_picture
    ode_solver
    hop_status

    References
    ----------
    .. [1] J. Chem. Phys. 93, 1061 (1990)
    .. [2] J. Chem. Phys. 137, 22S549 (2012)

    """

    def __init__(self, parameters, masses_nuclei, R, P):

        electronic_parameters = parameters.get("SurfaceHoppingElectrons")
        basis = electronic_parameters.get("basis", "adiabatic")

        electrons.Electrons.__init__(self, "SurfaceHoppingElectrons",
                                     parameters, basis)

        self.__masses_nuclei = masses_nuclei
        try:
            self.__n_steps = int(electronic_parameters.get("n_steps", 1))
            if self.n_steps < 1:
                raise ValueError("\nXPACDT: Number of electronic steps per"
                                 " nuclear step needs to be more"
                                 " than or equal to 1. Given in input file: "
                                 + self.n_steps)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Parameter 'n_steps'"
                          "for surface hopping not convertable to int.")

        self.__rpsh_type = electronic_parameters.get("rpsh_type", "centroid")
        self.__rpsh_rescaling = electronic_parameters.get("rpsh_rescaling", "bead")
        self.__rescaling_type = electronic_parameters.get("rescaling_type", "nac")
        self.__evolution_picture = electronic_parameters.get("evolution_picture", "schroedinger")
        self.__ode_solver = electronic_parameters.get("ode_solver", "scipy")

        if self.rpsh_type not in ['bead', 'centroid', 'density_matrix', 'dm_cb']:
            raise ValueError("\nXPACDT: Ring polymer surface hopping (RPSH)"
                             " type not available.")
        if self.rpsh_rescaling not in ['bead', 'centroid']:
            raise ValueError("\nXPACDT: RPSH rescaling type not available.")
        if self.rescaling_type not in ['nac', 'gradient']:
            raise ValueError("\nXPACDT: Momentum rescaling type not available.")
        if self.evolution_picture not in ['schroedinger', 'interaction']:
            raise ValueError("\nXPACDT: Evolution picture not available.")
        if self.ode_solver not in ['runga_kutta', 'unitary', 'scipy']:
            raise ValueError("\nXPACDT: ODE solver not available.")
        if (self.ode_solver == "unitary"):
            if (self.evolution_picture != "schroedinger"):
                raise ValueError("\nXPACDT: Evolution picture needs to be"
                                 " Schroedinger for unitary propagation.")

        max_n_beads = self.pes.max_n_beads
        n_states = self.pes.n_states
        if (n_states < 2):
            raise RuntimeError("\nXPACDT: Number of states should be more than"
                               " or equal to 2 for surface hoping. Else it"
                               " doesn't make sense to use it.")

        # Note: For now, RPSH-DM-Common basis(CB) only works for two states,
        # has to be in adiabatic basis, Schroedinger picture and
        # with same number of electronic and nuclear steps, i.e. n_steps==1.
        if (self.rpsh_type == 'dm_cb'):
            if (n_states != 2):
                raise RuntimeError("\nXPACDT: Number of states should be 2 for"
                                   " RPSH-DM-CB.")
            if (basis != 'adiabatic'):
                raise RuntimeError("\nXPACDT: Adiabatic basis should be used"
                                   " for RPSH-DM-CB. If you want to use diabatic"
                                   " basis, please use simply RPSH-DM as it is"
                                   " already in a common basis.")
            if (self.evolution_picture != "schroedinger"):
                raise ValueError("\nXPACDT: Evolution picture needs to be"
                                 " Schroedinger for RPSH-DM-CB.")
            if (self.n_steps != 1):
                raise RuntimeError("\nXPACDT: Number of electronic steps for each"
                                   " nuclear step should be 1 for RPSH-DM-CB.")

        try:
            initial_state = int(parameters.get("SurfaceHoppingElectrons").get("initial_state"))
            if (initial_state < 0) or (initial_state >= n_states):
                raise ValueError("\nXPACDT: Initial state is either less 0 or"
                                 " exceeds total number of "
                                 "states. Given initial state is "
                                 + initial_state)
        except (TypeError, ValueError) as e:
            raise type(e)(str(e) + "\nXPACDT: Initial state for surface hopping"
                          "not given or not convertible to int. Given initial"
                          "state is "
                          + parameters.get("SurfaceHoppingElectrons").get("initial_state"))

        self.__current_state = initial_state
        self.__hop_status = 'No hop'

        # The order chosen is transposed compared to that in interfaces with
        # n_beads as first axis. This is done for more efficient memory access
        # and matrix multiplications in this module.

        # The wavefunction coefficients are in interaction or Schroedinger
        # picture depending upon 'evolution_picture' selected.
        if (self.rpsh_type == 'density_matrix' or self.rpsh_type == 'dm_cb'):
            self._c_coeff = np.zeros((max_n_beads, n_states), dtype=complex)
            self._D = np.zeros((max_n_beads, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((max_n_beads, n_states, n_states), dtype=complex)
            # Needed for getting proper phase factors integrated over time in
            # interaction picture.
            if (self.evolution_picture == 'interaction'):
                # Phase is integral over time for diff_diag_V term,
                # doesn't include -i/hbar.
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
        return self.__current_state

    @property
    def masses_nuclei(self):
        """(n_dof) ndarray of floats : The masses of each nuclear degree of
           freedom in au."""
        return self.__masses_nuclei

    @property
    def n_steps(self):
        """int : Number of electronic steps to be taken for each nuclear step."""
        return self.__n_steps

    @property
    def rpsh_type(self):
        """{'bead', 'centroid', 'density_matrix', 'dm_c'} : String defining
        type of ring polymer surface hopping (RPSH) to be used."""
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

    @property
    def hop_status(self):
        """string : Whether surface hopping took place or not during a
        particular time step. This is set to 'No hop' before every step and
        depending upon successful or unsuccessful hop, it is changed to
        'Successful hop' or 'Attempted hop' respectively, followed by the
        states involved. """
        return self.__hop_status

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
            The energy of the current electronic state at each bead position
            or at the centroid in au.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.adiabatic_energy(R, self.current_state,
                                             centroid=centroid,
                                             return_matrix=False)
        else:
            return self.pes.diabatic_energy(R, self.current_state,
                                            self.current_state,
                                            centroid=centroid,
                                            return_matrix=False)

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
            The gradient of the current electronic state at each bead position
            or at the centroid in hartree/au.
        """
        if (self.basis == 'adiabatic'):
            return self.pes.adiabatic_gradient(R, self.current_state,
                                               centroid=centroid,
                                               return_matrix=False)
        else:
            return self.pes.diabatic_gradient(R, self.current_state,
                                              self.current_state,
                                              centroid=centroid,
                                              return_matrix=False)

    def get_population(self, proj, basis_requested):
        """ Get electronic population for a certain adiabatic or diabatic
        state. Adiabatic populations can always be obtained. Diabatic
        populations can only be obtained for potentials that are based on a
        diabatic model.

        TODO: Write down equations for populations for different rpsh_type for RPSH.
        TODO: Get proper population for RPSH-DM-CB.

        Parameters
        ----------
        proj : int
            State to be projected onto in the basis given by `basis_requested`.
        basis_requested : str
            Electronic basis to be used. Can be "adiabatic" or "diabatic".

        Returns
        -------
        population : float
            Electronic population value.
        """
        # If requested basis is the same as electronic basis, simply check if
        # it is in the requested state or not.
        if (self.basis == basis_requested):
            if proj == self.current_state:
                population = 1.0
            else:
                population = 0.0
        # If not, then get value by performing change of basis.
        else:
            if (self.pes.n_states == 2):
                import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad
            elif (self.pes.n_states > 2):
                import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad
            else:
                raise ValueError("Number of states should be 2 or more to use "
                                 "diabatic to adiabatic transformation. Here "
                                 "number of states is: " + str(self.pes.n_states))

            # Get diabatic to adiabatic transformation matrix U for centroid
            # or all beads based on rpsh type.
            if (self.rpsh_type == 'centroid'):
                U = dia2ad.get_transformation_matrix(self.pes._diabatic_energy_centroid)
                if (self.basis == 'diabatic' and basis_requested == 'adiabatic'):
                    population = (np.abs(U[proj, self.current_state]))**2
                else:
                    # For reverse case, need U_daggar so complex conjugate
                    # transposed element is needed, however conjugate not done
                    # due to absolute value.
                    population = (np.abs(U[self.current_state, proj]))**2
            else:
                # Getting shape (n_beads, n_states, n_states)
                U = dia2ad.get_transformation_matrix(self.pes._diabatic_energy).transpose(2, 0, 1)
                if (self.rpsh_type == 'bead'):
                    if (self.basis == 'diabatic' and basis_requested == 'adiabatic'):
                        population = (np.abs(np.mean([u_a[proj, self.current_state]
                                                     for u_a in U])))**2
                    else:
                        population = (np.abs(np.mean([np.conj(u_a[self.current_state, proj])
                                                     for u_a in U])))**2

                elif (self.rpsh_type == 'density_matrix'):
                    if (self.basis == 'diabatic' and basis_requested == 'adiabatic'):
                        population = np.mean([(np.abs(u_a[proj, self.current_state]))**2
                                             for u_a in U])
                    else:
                        # Complex conjugate is not done due to absolute value.
                        population = np.mean([(np.abs(u_a[self.current_state, proj]))**2
                                             for u_a in U])
        return population

    def _get_velocity(self, P):
        """Obtain velocities of the nuclei.

        Parameters
        ----------
        P : (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
            The ring-polymer beads or centroid momenta in au.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
            The velocities of the system for each bead position or centroid
            in au.
        """

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
        V : (n_beads, n_states, n_states) ndarray of floats
                if 'rpsh_type' == 'density_matrix' or 'dm_cb'
            /or/ (1, n_states, n_states) ndarray of floats
                if 'rpsh_type' == 'bead' or 'centroid'

            Energy matrix.
        """
        if (self.basis == 'adiabatic'):
            # Create diagonal matrices with 1D array of adiabatic energies.
            if (self.rpsh_type == 'centroid'):
                V = np.array([np.diag(self.pes.adiabatic_energy(R, centroid=True,
                                                                return_matrix=True))])
            else:
                # (n_states, n_beads).T done for pes.adiabatic_energy!
                V = np.array([np.diag(i) for i in
                              (self.pes.adiabatic_energy(R, centroid=False,
                                                         return_matrix=True).T)])
        else:
            if (self.rpsh_type == 'centroid'):
                V = np.array([self.pes.diabatic_energy(R, centroid=True,
                                                       return_matrix=True)])
            else:
                # (n_states, n_states, n_beads) changed to
                # (n_beads, n_states, n_states)
                V = self.pes.diabatic_energy(R, centroid=False,
                                             return_matrix=True).transpose(2, 0, 1)

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
        D : (n_beads, n_states, n_states) ndarray of floats
                if 'rpsh_type' == 'density_matrix' or 'dm_cb'
            /or/ (1, n_states, n_states) ndarray of floats
                if 'rpsh_type' == 'bead' or 'centroid'
            /or/ None
                if basis == 'diabatic'

            Kinetic coupling matrix.
        """
        if (self.basis == 'adiabatic'):
            # Here NAC is assumed to be real.
            # nac is (n_states, n_states, n_dof(, n_beads)) ndarrays
            if (self.rpsh_type == 'centroid'):
                v_centroid = self._get_velocity(np.mean(P, axis=1))
                nac = self.pes.nac(R, centroid=True, return_matrix=True)
                D = np.array([np.dot(nac, v_centroid)])
            else:
                # Transposes are done for faster memory accessing by making
                # 'nbeads' first axis.
                vel = self._get_velocity(P).T
                nac = (self.pes.nac(R, centroid=False,
                                    return_matrix=True)).transpose(3, 0, 1, 2)

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
        D : (n_beads, n_states, n_states) ndarray of floats
                if rpsh_type == 'density_matrix' or 'dm_cb'
            /or/ (1, n_states, n_states) ndarray of floats
                if rpsh_type == 'bead' or 'centroid'
            /or/ None
                if basis == 'adiabatic'

            Kinetic coupling matrix.

        Returns
        -------
        H : (n_beads, n_states, n_states) ndarray of complex
                if rpsh_type == 'density_matrix'
            /or/ (1, n_states, n_states) ndarray of complex
                if rpsh_type == 'bead' or 'centroid'

            Total non-adiabatic electronic Hamiltonian.
        """
        V = self._get_modified_V(R)

        if (self.basis == 'adiabatic'):
            H = V - 1j * D
        else:
            # .astype' creates a copy so '.copy()' isn't needed
            H = V.astype(complex)

        if (self.evolution_picture == "interaction"):
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
        diff : (n_beads, n_states, n_states) ndarray of floats
                  if rpsh_type == 'density_matrix' or 'dm_cb'
               /or/ (1, n_states, n_states) ndarray of floats
                  if rpsh_type == 'bead' or 'centroid'

            Diagonal energy difference matrix.
        """
        assert (self.evolution_picture == "interaction"), \
               ("Phase in electronic coefficients only makes sense in"
                " interaction picture.")

        diff = []
        n_states = self.pes.n_states
        V = self._get_modified_V(R)

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
            String to determine when electronic step is performed.
            'before_nuclei' refers to before nuclear step and
            'after_nuclei' refers to after.
        """

        assert ('step_index' in kwargs), ("Step index not provided for surface"
                                          " hopping propagtion.")
        assert ('step_count' in kwargs), ("Step count not provided for surface"
                                          " hopping propagtion.")

        # Only advance by full time step after the nuclear step.
        if (kwargs.get('step_index') != 'after_nuclei'):
            return

        # Set status to no hop before step only for the first nuclear step.
        # TODO: is there a better way than this (i.e. having this step_count)
        #       so that hop status still contains hop info even when output_step > nuclear_step?
        if (kwargs.get('step_count') == 0):
            self.__hop_status = 'No hop'

        self._D = self._get_kinetic_coupling_matrix(R, P)
        self._H_e_total = self._get_H_matrix(R, self._D)
        if (self.evolution_picture == 'interaction'):
            self._diff_diag_V = self._get_diff_diag_V_matrix(R)

        # Only relative time matters for integration so initial t = 0.
        # 'time_array' is (n_steps + 1) size containing end time as well.
        time_array = np.linspace(0., time_propagate, self.n_steps + 1,
                                 endpoint=True)
        # This permits steps not multiple of nuclear step since it only depends
        # upon 'time_propagate'. Although for now this is not done in nuclear step.
        timestep = time_propagate / float(self.n_steps)
        # 'b_list' will be list of size (n_steps + 1) or just 1 element (if
        # rpsh_type == 'dm_cb' since here only need final timestep value).
        b_list = []

        # Get initial population of current state
        if (self.rpsh_type == 'density_matrix'):
            # Mean is taken instead of sum to get normalizing factor of 1/n_beads
            a_kk_initial = np.mean(np.abs(self._c_coeff[:, self.current_state])**2)
        elif (self.rpsh_type == 'dm_cb'):
            # Here following orginal FSSH procedure, need a_kk after the step.
            # This is done since R here is after the nuclear step.
            a_kk_initial = None
        else:
            a_kk_initial = abs(self._c_coeff[0, self.current_state])**2

        prop_func = getattr(self, "_propagation_equation_"
                            + self.evolution_picture + "_picture")
        ode_solver_function = getattr(self, "_integrator_" + self.ode_solver)

        # Loop over until second last time step, i.e. time_propagate - timestep
        for t in time_array[:-1]:
            # Don't need intermediate steps for RPSH-DM-CB.
            if (self.rpsh_type != 'dm_cb'):
                b_list.append(self._get_b_jk(t, time_propagate, self._c_coeff))

            # Propagate by an electronic timestep
            self._c_coeff = ode_solver_function(t, self._c_coeff, timestep,
                                                time_propagate, prop_func)

        # Get b_jk for last time step.
        if (self.rpsh_type != 'dm_cb'):
            b_list.append(self._get_b_jk(time_array[-1], time_propagate,
                                         self._c_coeff))
        else:
            # Get a_kk for RPSH-DM-CB
            a_kk = self._get_rho_dm_cb(self._c_coeff, R)[self.current_state,
                                                         self.current_state]
            # This assumes that population doesn't change much in a single step
            a_kk_initial = a_kk.real

            # Get b_jk by time derivatives of populations.
            drho_cb_dt = self._get_drho_dt_dm_cb(self._c_coeff, R, P)
            b_list.append(np.diag(drho_cb_dt).real)

        # Perform surface hopping if needed.
        self._surface_hopping(R, P, time_array, a_kk_initial, b_list)

        # Update values
        self._old_H_e = self._H_e_total.copy()
        if (self.basis == 'adiabatic'):
            self._old_D = self._D.copy()
        else:
            self._old_D = None

        if (self.evolution_picture == 'interaction'):
            # Add to phase (by trapezoidal integration).
            self._phase += (0.5 * time_propagate * (self._old_diff_diag_V
                                                    + self._diff_diag_V))
            self._old_diff_diag_V = self._diff_diag_V.copy()

        return

    def _get_a_kj(self, c_coeff, phase=None):
        """ Calculate density matrix for all electronic coefficients. This for
        now is mostly used for testing purposes only and not used in the code.

        TODO: add equation for it. For rpsh=density_matrix, each bead has its own e- coeff.

        Parameters
        ----------
        c_coeff : (n_beads, n_states) ndarray of complex
                      if rpsh_type == 'density_matrix' or 'dm_cb'
                  /or/ (1, n_states) ndarray of complex
                      if rpsh_type == 'bead' or 'centroid'
            (Note: This shape here comes from rpsh_type within the module, but
            doesn't necessary have to be that.)
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        phase : None if evolution_picture == 'schroedinger'
            /or/ (*, n_states, n_states) ndarray of floats, where * is the
                 first dimension of the input array `c_coeff`
            Phase of the wavefunction in interaction picture.

        Returns
        -------
        a_jk : (*, n_states, n_states) ndarray of complex, where * is the
               first dimension of the input array `c_coeff`
            Electronic density matrix. This should be independent of
            'evolution_picture' selected.
        """
        if (self.evolution_picture == 'interaction'):
            a_kj = np.array([np.outer(c, np.conj(c)) * np.exp(1j * phase[i])
                             for i, c in enumerate(c_coeff)])
        else:
            a_kj = np.array([np.outer(c, np.conj(c)) for c in c_coeff])

        return a_kj

    def get_rho(self, basis_requested, R=None):
        """ Get electronic density matrix in adiabatic or diabatic basis.

        NOTE: For now, it assumes self.basis=='adiabatic'
        TODO: Extend to self.basis=='diabatic'

        Parameters
        ----------
        basis_requested : str
            Electronic basis to be used. Can be "adiabatic" or "diabatic".

        Returns
        -------
        rho : (n_states, n_states) ndarray of complex
            Electronic density matrix.
        """
        # If requested basis is the same as electronic basis, simply check if
        # it is in the requested state or not.
        if (basis_requested == "adiabatic"):
            if (self.rpsh_type == 'dm_cb'):
                rho = self._get_rho_dm_cb(self._c_coeff, R)
            else:
                rho = self._get_a_kj(self._c_coeff)
                if (self.rpsh_type == 'density_matrix'):
                    rho = np.mean(rho, axis=0)
                else:
                    rho = rho[0].copy()

        # If not, then perform change of basis.
        else:
            a_kj = self._get_a_kj(self._c_coeff)

            if (self.pes.n_states == 2):
                import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad
            elif (self.pes.n_states > 2):
                import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad
            else:
                raise ValueError("Number of states should be 2 or more to use "
                                 "diabatic to adiabatic transformation. Here "
                                 "number of states is: " + str(self.pes.n_states))
            if (self.rpsh_type == 'density_matrix' or self.rpsh_type == 'dm_cb'):
                V_d = self.pes.diabatic_energy(R, centroid=False, return_matrix=True)
                # Get U for each bead in shape (n_beads, n_states, n_states)
                U = dia2ad.get_transformation_matrix(V_d).transpose(2, 0, 1)
                U_dag = U.transpose(0, 2, 1)

                # Transform first to diabatic basis and take average.
                rho = np.mean([np.matmul(U_dag[i], np.matmul(a_i, U[i]))
                                    for i, a_i in enumerate(a_kj)], axis=0)

            else:
                # Get diabatic to adiabatic transformation matrix U for centroid
                # or all beads based on rpsh type.
                if (self.rpsh_type == 'centroid'):
                    U = dia2ad.get_transformation_matrix(self.pes._diabatic_energy_centroid)
                    rho = np.matmul(U.T, np.matmul(a_kj[0], U))

                else:
                    # TODO: How to do this for rpsh_type == 'bead'
                    rho = None

        return rho

    def _get_rho_dm_cb(self, c_coeff, R):
        """ Get density matrix for RPSH-DM-Common basis (CB). For this each
        bead coefficient is transformed to a common basis, here the centroid
        adiabatic basis and then averaged.

        Parameters
        ----------
        c_coeff : (n_beads, n_states) ndarray of complex
            Electronic wavefuntion coefficients in Schroedinger picture.
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` representing the system nuclei in
            au.

        Returns
        -------
        rho_cb : (n_states, n_states) ndarray of complex
            Density matrix for RPSH-DM-CB.
        """
        import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad

        assert (self.rpsh_type == 'dm_cb'),\
               ("This function only works for RPSH-DM-CB.")

        a_kj = self._get_a_kj(c_coeff)

        V_d = self.pes.diabatic_energy(R, centroid=False, return_matrix=True)
        # Get U for each bead in shape (n_beads, n_states, n_states)
        U = dia2ad.get_transformation_matrix(V_d).transpose(2, 0, 1)
        U_dag = U.transpose(0, 2, 1)

        # Transform first to diabatic basis and take average.
        rho_diab = np.mean([np.matmul(U_dag[i], np.matmul(a_i, U[i]))
                            for i, a_i in enumerate(a_kj)], axis=0)

        # Then transform to common adiabtic basis of centroid.
        V_d_centroid = self.pes.diabatic_energy(R, centroid=True,
                                                return_matrix=True)
        U_centroid = dia2ad.get_transformation_matrix(V_d_centroid)

        rho_cb = np.matmul(U_centroid, np.matmul(rho_diab, U_centroid.T))

        return rho_cb

    def _get_drho_dt_dm_cb(self, c_coeff, R, P):
        """ Get time derivative of density matrix in common basis (CB) for
        RPSH-DM-CB.

        Parameters
        ----------
        c_coeff : (n_beads, n_states) ndarray of complex
            Electronic wavefuntion coefficients in Schroedinger picture.
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.

        Returns
        -------
        drho_cb_dt : (n_states, n_states) ndarray of complex
            Time derivative of density matrix in common basis.
        """
        import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad

        assert (self.rpsh_type == 'dm_cb'),\
               ("This function only works for RPSH-DM-CB.")
        # This is nb*ns*ns
        rho = self._get_a_kj(c_coeff)
        vel = self._get_velocity(P)

        # Get drho_cb_dt in three step process.
        # Step 1: Here, get drho_dt for each bead at the end of the time step.
        # This is nb*ns
        dc_dt = self._propagation_equation_schroedinger_picture(1, c_coeff, 1)

        # This is nb*ns*ns
        drho_dt = np.array([np.outer(c, np.conj(dc_dt[i]))
                            for i, c in enumerate(c_coeff)])
        # Get the second part by adding complex conjugate transpose
        drho_dt += np.conj(drho_dt.transpose(0, 2, 1))


        # Step 2: Now get d_rho_diab_dt
        V_d = self.pes.diabatic_energy(R, centroid=False, return_matrix=True)
        dV_d = self.pes.diabatic_gradient(R, centroid=False, return_matrix=True)
        # Get U for each bead in shape (n_beads, n_states, n_states) ndarray of floats
        U = dia2ad.get_transformation_matrix(V_d).transpose(2, 0, 1)
        U_dag = U.transpose(0, 2, 1)
        dU_dt = dia2ad.get_time_derivative_of_U(V_d, dV_d,
                                                vel).transpose(2, 0, 1)

        # Transform first to diabatic basis and take average.
        rho_diab = np.mean([np.matmul(U_dag[i], np.matmul(rho_i, U[i]))
                            for i, rho_i in enumerate(rho)], axis=0)

        # This is also ns*ns
        drho_diab_dt = np.mean([np.matmul(U_dag[i], np.matmul(rho_i, dU_dt[i]))
                                for i, rho_i in enumerate(rho)], axis=0)
        drho_diab_dt += np.conj(drho_diab_dt.T)
        drho_diab_dt += np.mean([np.matmul(U_dag[i], np.matmul(drho_i, U[i]))
                                for i, drho_i in enumerate(drho_dt)], axis=0)


        # Step 3: now transform to common adiabatic basis of centroid.
        V_d_cen = self.pes.diabatic_energy(R, centroid=True, return_matrix=True)
        dV_d_cen = self.pes.diabatic_gradient(R, centroid=True,
                                              return_matrix=True)
        # Get U for each bead in shape (n_states, n_states)
        U_cen = dia2ad.get_transformation_matrix(V_d_cen)
        dU_cen_dt = dia2ad.get_time_derivative_of_U(V_d_cen, dV_d_cen,
                                                    np.mean(vel, axis=1))

        # This is ns*ns
        drho_cb_dt = np.matmul(dU_cen_dt, np.matmul(rho_diab, U_cen.T))
        drho_cb_dt += np.conj(drho_cb_dt.T)
        drho_cb_dt += np.matmul(U_cen, np.matmul(drho_diab_dt, U_cen.T))

        return drho_cb_dt

    def _get_b_jk(self, time, time_propagate, c):
        """ Calculate b_jk  where k is the current state and j is any other
        state except k, as defined in J. Chem. Phys. 93, 1061 (1990).

        TODO : add an equation for it. For density_matrix, it is averaged.
        b_jk = 2/ hbar Im(a_kj V_jk) - 2 Re(a_kj \\dot{R} d_kj)
        where k is the current state and a_kj is the density matrix.

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        c : (n_beads, n_states) ndarray of complex
                if rpsh_type == 'density_matrix'
            /or/ (1, n_states) ndarray of complex
                if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.

        Returns
        -------
        b_jk : (n_states) ndarray of floats
            b_jk array
        """
        t_frac = time / time_propagate
        H = mtools.linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)

        # Transposes need to be done for proper broadcasting to give a_kj of
        # shape (1 or max_n_beads, n_states).
        if (self.evolution_picture == 'schroedinger'):
            a_kj = (c[:, self.current_state] * np.conj(c).T).T

        elif (self.evolution_picture == 'interaction'):
            phase = self._phase + 0.5 * time * (self._old_diff_diag_V
                                                + mtools.linear_interpolation_1d(
                                                  t_frac, self._old_diff_diag_V,
                                                  self._diff_diag_V))
            a_kj = (c[:, self.current_state]
                    * (np.conj(c) * np.exp(1j * phase[:, self.current_state, :])).T).T

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

    def _integrator_runga_kutta(self, time, c_coeff, timestep, time_propagate,
                                prop_func):
        """ Propagate the electronic coefficients by one electronic time step
        using the classic 4th order Runga-Kutta method.
        Reference: http://mathworld.wolfram.com/Runge-KuttaMethod.html

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex
                      if rpsh_type == 'density_matrix'
                  /or/ (1, n_states) ndarray of complex
                      if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        timestep : float
            Electronic time step in au.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients. For
            now it refers to propagation in interaction or Schroedinger picture.

        Returns
        -------
        c_new : ndarray of complex with the same shape as `c_coeff`
            New electronic coefficients obtained after propagation.
        """
        c_1 = timestep * prop_func(time, c_coeff, time_propagate)
        c_2 = timestep * prop_func((time + 0.5 * timestep),
                                   c_coeff + 0.5 * c_1, time_propagate)
        c_3 = timestep * prop_func((time + 0.5 * timestep),
                                   c_coeff + 0.5 * c_2, time_propagate)
        c_4 = timestep * prop_func((time + timestep),
                                   c_coeff + c_3, time_propagate)

        c_new = c_coeff + 1.0 / 6.0 * (c_1 + c_4) + 1.0 / 3.0 * (c_2 + c_3)

        # Normalizing by hand to conserve norm.
        for i in c_new:
            normalization = np.linalg.norm(i)
            i /= normalization

        return c_new

    def _integrator_scipy(self, time, c_coeff, timestep, time_propagate,
                          prop_func):
        """ Propagate the electronic coefficients by electronic time step using
        ode integrator from scipy.

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex
                      if rpsh_type == 'density_matrix'
                  /or/ (1, n_states) ndarray of complex
                      if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        timestep : float
            Electronic time step in au.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients. For
            now it refers to propagation in interaction or Schroedinger picture.

        Returns
        -------
        c_new : ndarray of complex with the same shape as `c_coeff`
            New electronic coefficients obtained after propagation.
        """
        c_new = []

        solver = ode(prop_func).set_integrator('zvode', first_step=timestep,
                                               method='bdf')

        for i, c in enumerate(c_coeff):
            solver.set_f_params(time_propagate, i)
            solver.set_initial_value(c, time)
            while solver.successful() and solver.t < (time + timestep):
                solver.integrate(solver.t + timestep)

            # Normalizing by hand to conserve norm.
            normalization = np.linalg.norm(solver.y)
            c_new.append(solver.y / normalization)

        return np.array(c_new)

    def _integrator_unitary(self, time, c_coeff, timestep, time_propagate,
                            prop_func=None):
        """ Propagate the electronic coefficients by electronic time step using
        unitary evolution matrix at the midpoint. This is a 3rd order method.
        For unitary evolution, the coefficients have to be in Schroedinger
        picture.
        Reference: Chemical Physics 349, 334 (2008)

        Parameters
        ----------
        time : float
            Current time in this propagation in au.
        c_coeff : (n_beads, n_states) ndarray of complex
                      if rpsh_type == 'density_matrix'
                  /or/ (1, n_states) ndarray of complex
                      if rpsh_type == 'bead' or 'centroid'
            Electronic wavefuntion coefficients in Schroedinger picture.
        timestep : float
            Electronic time step in au.
        time_propagate : float
            Total time to advance the electrons in this propagation step in au.
        prop_func : function
            Function to use for propagating the electronic coefficients.
            This is not required for this method so default is None.

        Returns
        -------
        c_new : ndarray of complex with the same shape as `c_coeff`
            New electronic coefficients obtained after propagation.
        """
        t_frac_mid = (time + 0.5 * timestep) / time_propagate

        H_mid = mtools.linear_interpolation_1d(t_frac_mid, self._old_H_e,
                                               self._H_e_total)
        propagator = np.array([expm(-1j * h * timestep) for h in H_mid])

        c_new = np.matmul(propagator,
                          np.expand_dims(c_coeff, axis=-1)).reshape(-1, self.pes.n_states)

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
        c : (n_beads, n_states) ndarray of complex
                if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
            /or/ (1, n_states) ndarray of complex
                if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
            /or/ (n_states) ndarray of complex
                if ode_solver == 'scipy'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        t_prop : float
            Total time to advance the electrons in au.
        i : int
            Index for electronic coefficients; needed if ode_solver == 'scipy'.

        Returns
        -------
        ndarray of complex with the same shape as `c`
            Time derivative of electronic coefficients at time `t`.
        """
        # Time fraction in the propagation step, needed for interpolation.
        t_frac = t / t_prop
        if i is None:
            H = mtools.linear_interpolation_1d(t_frac, self._old_H_e,
                                               self._H_e_total)
            return (-1.0j * np.matmul(H, np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            H = mtools.linear_interpolation_1d(t_frac, self._old_H_e[i],
                                               self._H_e_total[i])
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
        c : (n_beads, n_states) ndarray of complex
                if rpsh_type == 'density_matrix' and ode_solver == 'runga_kutta'
            /or/ (1, n_states) ndarray of complex
                if rpsh_type == 'bead' or 'centroid', and ode_solver == 'runga_kutta'
            /or/ (n_states) ndarray of complex
                if ode_solver == 'scipy'
            Electronic wavefuntion coefficients in interaction or Schroedinger
            picture depending upon 'evolution_picture' selected.
        t_prop : float
            Total time to advance the electrons in au.
        i : int
            Index for electronic coefficients; needed if ode_solver == 'scipy'.

        Returns
        -------
        ndarray of complex with the same shape as `c`
            Time derivative of electronic coefficients at time `t`.
        """
        # Time fraction in the propagation step, needed for interpolation.
        t_frac = t / t_prop

        if i is None:
            H = mtools.linear_interpolation_1d(t_frac, self._old_H_e,
                                               self._H_e_total)

            phase = self._phase \
                + 0.5 * t * (self._old_diff_diag_V +
                             mtools.linear_interpolation_1d(t_frac, self._old_diff_diag_V,
                                                            self._diff_diag_V))

            # 'np.expand_dims' used to add a dimension for proper
            # matrix multiplication.
            return (-1.0j * np.matmul((H * np.exp(-1.0j * phase)),
                                      np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))
        else:
            H = mtools.linear_interpolation_1d(t_frac, self._old_H_e[i],
                                               self._H_e_total[i])
            phase = self._phase[i] \
                + 0.5 * t * (self._old_diff_diag_V[i] +
                             mtools.linear_interpolation_1d(t_frac, self._old_diff_diag_V[i],
                                                            self._diff_diag_V[i]))

            return (-1.0j * np.matmul((H * np.exp(-1.0j * phase)), c))

    def _surface_hopping(self, R, P, time_array, a_kk_initial, b_list):
        """ Perform surface hopping if required by comparing the probability
        to hop with a random number drawn from a uniform distribution between 0
        and 1. Based on the random number, decide whether to stay on the same
        surface or to hop. If there is a hop, then check for energy
        conservation and rescale momenta of nuclei accordingly.

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
            Probability density of current state before the propagation.
        b_list : (n_steps + 1) list of (n_states) ndarray of floats
                 /or/ (1) list of (n_states) ndarray of floats if rpsh_type == 'dm_cb'
            List of b_jk at times given in `t_list`.
        """
        if (self.rpsh_type != 'dm_cb'):
            # Perform trapezoidal integration to get hopping probability for each
            # state, so 'prob' is (n_state) ndarray of floats
            prob = np.trapz(np.array(b_list),
                            np.array(time_array), axis=0) / a_kk_initial
        else:
            # Here b_list only has a single entry, and assumes that there's
            # only a single e- timestep.
            prob = b_list[0] * time_array[-1] / a_kk_initial

        # Setting negative probability and hopping to current state to 0.
        prob[prob < 0] = 0.
        prob[self.current_state] = 0.

        if np.sum(prob) > 1.:
            raise RuntimeError("\nXPACDT: Total hopping probability is more"
                               " than 1. Try again with smaller nuclear timestep.")

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
                self.__hop_status = 'Successful hop from state {:d} to state {:d}'.format(
                    self.current_state, new_state)
                self.__current_state = new_state
            else:
                self.__hop_status = 'Attempted hop from state {:d} to state {:d}'.format(
                    self.current_state, new_state)

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

        assert (new_state != self.current_state),\
               ("New state to hop cannot be the same as the old state.")

        if (self.rpsh_rescaling == 'centroid'):
            centroid = True
        else:
            centroid = False

        # Obtain change in potential for each bead or centroid to give
        # (nbeads) ndarray of floats or a float value
        if (self.basis == 'adiabatic'):
            diff_V = self.pes.adiabatic_energy(R, self.current_state, centroid=centroid, return_matrix=False) \
                    - self.pes.adiabatic_energy(R, new_state, centroid=centroid, return_matrix=False)
        else:
            diff_V = self.pes.diabatic_energy(R, self.current_state, self.current_state, centroid=centroid, return_matrix=False) \
                    - self.pes.diabatic_energy(R, new_state, new_state, centroid=centroid, return_matrix=False)

        # Obtain projection vector for each dof and beads or centroid, which
        # gives (ndof, nbeads) or (ndof) array
        if (self.rescaling_type == 'nac'):
            proj_vec = self.pes.nac(R, self.current_state, new_state,
                                    centroid=centroid, return_matrix=False)
        elif (self.rescaling_type == 'gradient'):
            # Get gradient differences
            if (self.basis == 'adiabatic'):
                proj_vec = self.pes.adiabatic_gradient(R, self.current_state, centroid=centroid, return_matrix=False) \
                          - self.pes.adiabatic_gradient(R, new_state, centroid=centroid, return_matrix=False)
            else:
                proj_vec = self.pes.diabatic_gradient(R, self.current_state, self.current_state, centroid=centroid, return_matrix=False) \
                          - self.pes.diabatic_gradient(R, new_state, new_state, centroid=centroid, return_matrix=False)

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
            assert (A_kj != 0.0),\
                   ("A_kj in momentum rescaling is 0. Try different rescaling"
                    " type. Currently it is: " + self.rescaling_type + " in"
                    " basis: " + self.basis)

            # Take the least possible change
            if (B_kj < 0.):
                factor = (B_kj + math.sqrt(inside_root)) / (2. * A_kj)
            else:
                factor = (B_kj - math.sqrt(inside_root)) / (2. * A_kj)

            # Changing P here should also change in nuclei
            for i, p_A in enumerate(P):
                p_A -= factor * proj_vec[i]

            return True
