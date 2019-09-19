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

import numpy as np

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

    References
    ----------
    .. [1] J. Chem. Phys. 93, 1061 (1990)
    .. [2] J. Chem. Phys. 137, 22S549 (2012)

    """

    def __init__(self, parameters, n_beads):

        electronic_parameters = parameters.get("SurfaceHoppingElectrons")
        basis = electronic_parameters.get("basis", "adiabatic")

        electrons.Electrons.__init__(self, "SurfaceHoppingElectrons",
                                     parameters, n_beads, basis)
        print("Initiating surface hopping in ", self.basis, " basis")

        # TODO: decide what should be the default, for nb=1 does that shouldn't matter though
        self.rpsh_type = electronic_parameters.get("rpsh_type", "bead")
        self.rpsh_rescaling = electronic_parameters.get("rpsh_rescaling", "bead")
        self.rescaling_type = electronic_parameters.get("rescaling_type", "nac")

        self.__masses_nuclei = parameters.masses
        # Add try and exception here for the factor?
        timestep_scaling_factor = float(electronic_parameters.get("timestep_scaling_factor"))
        nuclear_timestep = units.parse_time(parameters.get("nuclei_propagator").get("timestep"))
        self.__timestep = timestep_scaling_factor * nuclear_timestep

        self.evolution_picture = electronic_parameters.get("evolution_picture", "interaction")
        self.ode_solver = electronic_parameters.get("ode_solver", "rk4")

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
                self._phase = np.zeros(max_n_beads, n_states, n_states, dtype=complex)
                self._diff_diag_V = np.zeros(max_n_beads, n_states, n_states, dtype=complex)
        else:
            self._c_coeff = np.zeros((1, n_states), dtype=complex)
            self._D = np.zeros((1, n_states, n_states), dtype=float)
            self._H_e_total = np.zeros((1, n_states, n_states), dtype=complex)
            if (self.evolution_picture == 'interaction'):
                self._phase = np.zeros((1, n_states, n_states), dtype=complex)
                self._diff_diag_V = np.zeros((1, n_states, n_states), dtype=complex)

        self._c_coeff[:, self.current_state] = 1.0 + 0.0j

        positions = parameters.coordinates
        momenta = parameters.momenta
        self._old_D = self._get_kinetic_coupling_matrix(positions, momenta)
        self._old_H_e = self._get_H_matrix(positions, self._old_D)
        if (self.evolution_picture == 'interaction'):
            self._old_diff_diag_V = self._get_diff_diag_V_matrix(positions)

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
        """{'bead', 'centroid', 'density_matrix'} : Type of ring polymer
        surface hopping (RPSH) to be used."""
        # TODO: Possibly add 'individual'; does having individual bead hops make sense?
        return self.__rpsh_type

    @rpsh_type.setter
    def rpsh_type(self, r):
        assert (r in ['bead', 'centroid', 'density_matrix']),\
               ("Ring polymer surface hopping (RPSH) type not available.")
        self.__rpsh_type = r

    @property
    def rpsh_rescaling(self):
        """{'bead', 'centroid'} : Type of RPSH rescaling to be used;
        this can be either to conserve bead or centroid energy."""
        return self.__rpsh_rescaling

    @rpsh_rescaling.setter
    def rpsh_rescaling(self, r):
        assert (r in ['bead', 'centroid']),\
               ("RPSH rescaling type not available.")
        self.__rpsh_rescaling = r

    @property
    def rescaling_type(self):
        """{'nac', 'gradient'} : Type of momentum rescaling to be used."""
        # TODO: possibly add 'nac_with_momenta_reversal' if needed
        return self.__rescaling_type

    @rescaling_type.setter
    def rescaling_type(self, r):
        assert (r in ['nac', 'gradient']),\
               ("Momentum rescaling type not available.")
        self.__rescaling_type = r

    @property
    def evolution_picture(self):
        """{'interaction', 'schroedinger'} : Representation/picture for quantum evolution."""
        # TODO: possibly add 'schrodinger' if needed
        return self.__evolution_picture

    @evolution_picture.setter
    def evolution_picture(self, p):
        assert (p in ['interaction', 'schroedinger']),\
               ("Evolution picture not available.")
        self.__evolution_picture = p

    @property
    def ode_solver(self):
        """{'rk4'} : Type of velocity rescaling to be used."""
        # TODO: 'scipy_solver' if needed
        return self.__ode_solver

    @rescaling_type.setter
    def ode_solver(self, s):
        assert (s in ['rk4']),\
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
              Also rename it as derivative coupling

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

                # D here is (n_beads) list of (n_states, n_states)
                D_list = []
                for i, v in enumerate(vel):
                    D_list.append(np.dot(nac[i], v))

                if (self.rpsh_type == 'bead'):
                    D = np.mean(np.array(D_list), axis=0)
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
            # Is this assert needed as this function is only accessed internally.
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
        diff : (n_beads, n_states, n_states) ndarray of floats if
            /or/ (1, n_states, n_states) ndarray of floats
            Diagonal energy difference matrix.
        """
        # Maybe put the assert in the initialization
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
        by propagating electronic wavefunction coefficients.

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
        """
        # TODO: again too many asserts?
        assert ('step_index' in kwargs), ("Step index not provided for surface"
                                          " hopping propagtion.")

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
        n_steps = int((time_propagate + 1e-8) // self.timestep)
        # Only relative time matters, right? set initial t = 0
        t_list = [0.0]
        c_list = [self._c_coeff]
        H_list = [self._H_e_total]
        
        # Using for loop has better accuracy than while loop with adding step, right?
        for i in range(n_steps):
            if (self.ode_solver == 'rk4'):
                if (self.evolution_picture == 'interaction'):
                    prop_func = self._propagation_equation_interaction_picture
                elif (self.evolution_picture == 'schroedinger'):
                    prop_func = self._propagation_equation_schroedinger_picture

                self._c_coeff = self._integrator_runga_kutta(i * self.timestep, time_propagate,
                                                             self._c_coeff, prop_func)
                t_list.append(i * self.timestep)
                # Maybe just get b_jk here??!
                
                c_list.append(self._c_coeff)
                # How to avoid interpolating again and store at once? Maybe make a self._H_list?
                
                # Make this as a attribute
                H_list.append(self._linear_interpolation_1d(i * self.timestep / time_propagate, self._old_H_e, self._H_e_total))
                # !!! Need phase list as well
                
        # Perform Surface hops, for this need to get integrated b_jk for which we need list of c_coeff and H at those steps
        # However only need to calculate prob from current state to all other states.
        # !!! maybe better to give b_jk and a_jk??
        self._surface_hopping(R, P, t_list, c_list, H_list)
        
        if (self.evolution_picture == 'interaction'):
            # At the end, add to phase (by trapezoidal integration) and update values
            self._phase = self._phase + 0.5 * time_propagate * (self._old_diff_diag_V + self._diff_diag_V)
            self._old_diff_diag_V = self._diff_diag_V.copy()

        self._old_D = self._D.copy()
        self._old_H_e = self._H_e_total.copy()

        return

    def _integrator_runga_kutta(self, t, time_propagate, c, prop_func):
        # Classical Runga-Kutta RK4 algorithm for electronic quantum coefficients
        # Should rk4 calculate and return all of the interpolated values???
        t_step = self.timestep
        c_1 = t_step * prop_func(t, time_propagate, c)
        c_2 = t_step * prop_func((t + 0.5 * t_step), time_propagate, c + 0.5 * c_1)
        c_3 = t_step * prop_func((t + 0.5 * t_step), time_propagate, c + 0.5 * c_2)
        c_4 = t_step * prop_func((t + t_step), time_propagate, c + c_3)

        c_new = c + 1.0 / 6.0 * (c_1 + c_4) + 1.0 / 3.0 * (c_2 + c_3)

        return c_new
    
    # Get wrapper function to give interpolated H
    def _integrator_scipy(self, ):
        return

    def _propagation_equation_interaction_picture(self, t, t_prop, c):
        """Propagation equation in interaction picture.

        Parameters
        ----------
        c : (n_beads, n_states) ndarray of floats
            /or/ (1, n_states) ndarray of floats
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
        # 'np.expand_dims' used to add a dimension for proper matrix multiplication
        return (-1.0j * np.matmul(H, np.matmul(np.exp(-1.0j * phase),
                                               np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states)))

    def _propagation_equation_schroedinger_picture(self, t, t_prop, c):
        # Matrix product of -i/hbar np.matmul(H, c)
        t_frac = t / t_prop
        H = self._linear_interpolation_1d(t_frac, self._old_H_e, self._H_e_total)
        return (-1.0j * np.matmul(H, np.expand_dims(c, axis=-1)).reshape(-1, self.pes.n_states))

    # Maybe put this also in math module
    def _linear_interpolation_1d(self, x_fraction, y_initial, y_final):
        # x_fraction = (x_required - x_initial) / (x_final - x_initial)
        # y can be any dimension array, x_fraction is float
        return ((1. - x_fraction) * y_initial + x_fraction * y_final)

    def _surface_hopping(self, R, P, t_list, c_list, H_list):
        # Perform surface hopping if required
        # Create density matrix/ reduced (averaged one)
        # Integrate to get b_jk
        # Get prob to hop, set to 0 if less than zero
        # State switch if needed, have another function for momentum rescaling
        
        # Here k is the current state
        # First get a_kj
        # a_jk = c * (np.conj(c[self.current_state]) * exp(1j * phase[self.current_state])
        # Is it better to integrate using scipy.integrate.trapez
        # b_jk = 2 * (np.conj(a) * V[:, self.current_state]).imag if diabatic
        # b_jk = 2 * (np.conj(a) * D[:, self.current_state]).real if adiabatic
        
        
        return
    
    def _momentum_rescaling(self,):
        # Perform momentum rescaling if needed.
        return
