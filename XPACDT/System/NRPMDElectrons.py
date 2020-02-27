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

""" 
In this module the quasiclassical mapping of electronic states within the RPMD
framework is implemented in a matrixcalculus fashion regarding the energie, 
the gradient, and die propagation of the electronic coordinates an momenta. 
"""

import numpy as np
import XPACDT.System.Electrons as electrons
import XPACDT.Tools.Units as units
from scipy.linalg import sinm, cosm


class NRPMDElectrons(electrons.Electrons):
        
    """
    This module extends the description of electrons in the systems under
    consideration by mapping the electronic states by harmonic oscillators
    and thus enables the description within the RPMD model also for electrons.
    For reference the paper of ..... has to be mentioned.

    The implementation follows closely the formlulation of the electronic
    mapping by S. Chowdhury and P. Huo in reference [1]. Refering to this 
    work the initialisation, the calculation of the energie, gradient and 
    the population can be found there.
    The propagation in time was implementet acording to the work of 
    J. Richardson and M. Thoss in reference [2] and was also implementet 
    in an matrixcalculus manner.

    Parameters
    ----------
    parameters: XPACT.Input.Inputfile
    R(, P): (n_dof, n_beads) ndarray of floats.
            The (ring-polymer) positions 'R' (and momenta 'P') representing
            the system nuclei in a.u.
        
    Attributes
    ----------
    q
    p

    References
    ----------
    .. [1] J. Chem. Phys. 150, 244102 (2019)
    .. [2] Chem. Phys. 482, 124-134 (2017)
    """

    def __init__(self, parameters, n_beads, R=None, P=None):

        electrons.Electrons.__init__(self, "NRPMDElectrons", parameters, n_beads, 'diabatic')
        initstate = int(parameters.get("NRPMDElectrons").get("initial_states"))
        self.tstep = units.parse_time(parameters.get("NRPMDElectrons").get("timestep"))

        self.q = np.zeros((self.pes.n_states, self.pes.max_n_beads))
        self.p = np.zeros((self.pes.n_states, self.pes.max_n_beads))

        #initialisation corresponding to reference [1] 
        #sampling with an random angle 
        #it is ensured that 1/2[(q_j)²+(p_j)²-1] equals delta(j,i) for the
        #occupied state |i> 
        angles = 2.0*np.pi*np.random.random((self.pes.n_states, self.pes.max_n_beads))

        self.q = np.sin(angles)
        self.p = np.cos(angles)

        self.q[initstate, :] *= np.sqrt(3)
        self.p[initstate, :] *= np.sqrt(3)

    def step(self, R, time_propagate, **kwargs):

        """
        Calculate the stepwise propagation of positions and momenta of the
        system electrons as defined by the systems PES.
        
        Here the exact step is implemented in a matrixcalculus format following
        the work of J. richardson und M. Thoss in reference [2]:
            
        :math: ´\cvect{\text{\textbf{q}}_{i,\alpha}+\frac{1}{2}
        \dot{\text{\textbf{q}}}_{i,\alpha}\delta t}{\text{\textbf{p}}_{i,\alpha}
        +\frac{1}{2}\dot{\text{\textbf{p}}}_{i,\alpha}\delta t} = 
        \begin{MATRIX}\cos\left(\frac{V(q_{i})\delta t}{2\hbar}\right) & 
        \sin\left(\frac{V(q_{i})\delta t}{2\hbar}\right) \\ 
        -\sin\left(\frac{V(q_{i})\delta t}{2\hbar}\right) & 
        \cos\left(\frac{V(q_{i})\delta t}{2\hbar}\right) \end{MATRIX} 
        \cvect{\text{\textbf{q}}_{i,\alpha}}{\text{\textbf{p}}_{i,\alpha}}`

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        time_propagate:
            Number in atomic time units for the propagation of the electronic
            coordinates and momenta. 
        """
        
        #the exact step from reference [2]
        potential = self.pes.diabatic_energy(R, return_matrix=True).transpose(2, 0, 1)
        q_mat = np.zeros_like(self.q)
        p_mat = np.zeros_like(self.p)
        for i in range(self.pes.max_n_beads):
            Potential = potential[i]

            q_mat[:, i] = (np.matmul((cosm(Potential*0.5*time_propagate)),
                                     np.expand_dims(self.q[:, i], axis=-1)) +
                           np.matmul((sinm(Potential*0.5*time_propagate)),
                                     np.expand_dims(self.p[:, i], axis=-1)))\
                           .reshape(-1)
            p_mat[:, i] = (np.matmul((cosm(Potential*0.5*time_propagate)),
                                     np.expand_dims(self.p[:, i], axis=-1)) -
                           np.matmul((sinm(Potential*0.5*time_propagate)),
                                     np.expand_dims(self.q[:, i], axis=-1)))\
                           .reshape(-1)

        #renaming for later circles
        self.q = q_mat
        self.p = p_mat

    def energy(self, R, centroid=False):

        """
        Calculate the electronic energy at the current geometry as defined
        by the systems PES.
        
        The electronic energie part of the whole hamiltonian as in reference 
        [1] is calcualted acording to:
            
        :math:´\hat{H}_{el}=\frac{1}{2\hbar}\Sum{\alpha=1}{N}{\Sum{nm}{}{V_{nm}
        (\text{\textbf{R}}_{\alpha})\times([\text{\textbf{q}}_{\alpha}]_{n}
        [\text{\textbf{q}}_{\alpha}]_{m}+[\text{\textbf{p}}_{\alpha}]_{n}
        [\text{\textbf{p}}_{\alpha}]_{m}-\delta_{nm}\hbar)}}´   

        where alpha is the bead-index and is running to N=self.pes.max_n_beads
        and V is the diabatic potential energie matrix for n_states electronic 
        states.
        
        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_beads) ndarray of float /or/ float
        The energy of the systems PES at each bead position or at the centroid
        in hartree. 
        """
        # comment on centroid is not implementet yet 

        #shape (ns,ns,nb) => (nb,ns,ns)
        Potential = self.pes.diabatic_energy(R, return_matrix=True).transpose(2, 0, 1)
        #calculate the energy within matrixcalkulus
        Elek_mat_energy = 0.5 * (np.matmul(np.matmul(np.expand_dims(self.q, axis=-1)
                                                     .transpose(1, 2, 0), Potential),
                                                     np.expand_dims(self.q, axis=-1)
                                                     .transpose(1, 0, 2))
                                                     .reshape(-1, self.pes.max_n_beads) +
                                 np.matmul(np.matmul(np.expand_dims(self.p, axis=-1)
                                                     .transpose(1, 2, 0), Potential), 
                                                     np.expand_dims(self.p, axis=-1)
                                                     .transpose(1, 0, 2))
                                                     .reshape(-1, self.pes.max_n_beads) -
                                 np.trace(Potential, axis1=1, axis2=2)
                                 .reshape(-1, self.pes.max_n_beads))
        #reshape the vector to max_n_beads only
        Elek_mat_energy = Elek_mat_energy.reshape(self.pes.max_n_beads)

        return Elek_mat_energy

    def gradient(self, R, centroid=False):

        """Calculate the gradient of the electronic energy at the current
        geometry as defined by the systems PES.
        
        The electronic part of the gradient for the nuclear motion is calculated 
        as in reference [1]:
            
        :math:´\dot{\text{\textbf{P}}}_{\alpha}^{el}=\frac{1}{2\hbar}
        \Sum{\alpha=1}{N}{\Sum{nm}{}{\nabla_{\text{\textbf{R}}_{\alpha}}
        V_{nm}(\text{\textbf{R}}_{\alpha})\times([\text{\textbf{q}}_{\alpha}]_{n}
        [\text{\textbf{q}}_{\alpha}]_{m}+[\text{\textbf{p}}_{\alpha}]_{n}
        [\text{\textbf{p}}_{\alpha}]_{m}-\delta_{nm}\hbar)}}´

        where alpha is the bead-index and is running to N=self.pes.max_n_beads
        and :math:´\nabla_{\text{\textbf{R}}_{\alpha}}
        V_{nm}(\text{\textbf{R}}_{\alpha})´ is the diabatic potential energie
        gradient matrix for n_states electronic states.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
        The gradient of the systems PES at each bead position or at the
        centroid in hartree/au.
        """

        #shape (ns,ns,nb,ndof) => (nb,ndof,ns,ns)
        Gradient = self.pes.diabatic_gradient(R, return_matrix=True).transpose(2, 3, 0, 1) 
        #calculate the gradient within matrixcalkulus
        Elek_mat_gradient = -0.5 * (np.matmul(np.matmul(np.expand_dims(self.q, axis=-1)
                                                        .transpose(1, 2, 0), Gradient),
                                                        np.expand_dims(self.q, axis=-1)
                                                        .transpose(1, 0, 2))
                                                        .reshape(-1, self.pes.max_n_beads) +
                                    np.matmul(np.matmul(np.expand_dims(self.p, axis=-1)
                                                        .transpose(1, 2, 0), Gradient),
                                                        np.expand_dims(self.p, axis=-1)
                                                        .transpose(1, 0, 2))
                                                        .reshape(-1, self.pes.max_n_beads) -
                                    np.trace(Gradient, axis1=2, axis2=3)
                                    .reshape(-1, self.pes.max_n_beads))

        return Elek_mat_gradient

    def get_population(self, proj, basis_requested):

        """ Get electronic population for a certain adiabatic or diabatic
        state. Diabatic populations can only be obtained for potentials that
        are based on a diabatic model.

        The Population of the electronic state j is calculated within the
        framework of the reference [1]:
        
        :math:´\bar{\mathcal{P}}_{j}=\frac{1}{N}\Sum{\alpha}{}{\mathcal{P}_{j}
        (\alpha)}=\frac{1}{N} \Sum{\alpha=1}{N}{\frac{1}{2}([\text{\textbf{q}}
        _{\alpha}]_{j}^{2}+[\text{\textbf{p}}_{\alpha}]_{j}^{2}-1})´

        Parameters
        ----------
        proj : int
            State to be projected onto in the basis given by `basis_requested`.
        basis_requested : str
            Electronic basis to be used. Can be "diabatic".
            "adiabatic" will raise a "NotImplementedError"

        Returns
        -------
        population : float
            Electronic population value.
        """
        if basis_requested == "diabatic":

            population = np.zeros(self.pes.n_states)
            for i in range(self.pes.n_states):
                for j in range(self.pes.max_n_beads):
                    population[i] += (1 / (2 * self.pes.max_n_beads)) * \
                    (self.q[i, j]*self.q[i, j] + self.p[i, j]*self.p[i, j] - 1.0)

        else:
            raise NotImplementedError

        return population[proj]


