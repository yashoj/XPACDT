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


import numpy as np
import XPACDT.System.Electrons as electrons
import XPACDT.Tools.Units as units
import scipy.linalg as sl


class NRPMDElectrons(electrons.Electrons):

    def __init__(self, parameters, n_beads, R=None, P=None):

        electrons.Electrons.__init__(self, "NRPMDElectrons", parameters, n_beads, 'diabatic')
        self.initstate = int(parameters.get("NRPMDElectrons").get("initial_states"))
        self.tstep = units.parse_time(parameters.get("NRPMDElectrons").get("timestep"))

        self.q = np.zeros((self.pes.n_states, self.pes.max_n_beads))
        self.p = np.zeros((self.pes.n_states, self.pes.max_n_beads))
        self.Einsmatrix = np.identity(self.pes.n_states)
        angles = 2.0*np.pi*np.random.random((self.pes.n_states, self.pes.max_n_beads))

        self.q = np.sin(angles)
        self.p = np.cos(angles)

        self.q[self.initstate, :] *= np.sqrt(3)
        self.p[self.initstate, :] *= np.sqrt(3)

    def step(self, R, **kwargs):

        """Calculate the stepwise propagation of position and momentum of the
        system electrons as defined by the systems PES.

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
        (n_states, n_beads) ndarrays of float /or/ float
        The position and momenta of the systems electrons of the systems PES at
        each bead position or at the centroid in hartree.
        """

#      the exact step from Richardson and Thoss
        potential = self.pes.diabatic_energy(R, return_matrix=True).transpose(2, 0, 1)
        q_mat = np.zeros_like(self.q)
        p_mat = np.zeros_like(self.p)
        for i in range(self.pes.max_n_beads):
            Potential = potential[i]
#            print("Potential[i]", Potential)
#            print("q[i]",self.q[i])
            q_mat[:, i] = (np.matmul((sl.cosm(Potential*self.tstep)),
                                     np.expand_dims(self.q[:, i], axis=-1)) +
                           np.matmul((sl.sinm(Potential*self.tstep)),
                                     np.expand_dims(self.p[:, i], axis=-1)))\
                .reshape(-1)
            p_mat[:, i] = (np.matmul((sl.cosm(Potential*self.tstep)),
                                     np.expand_dims(self.p[:, i], axis=-1)) -
                           np.matmul((sl.sinm(Potential*self.tstep)),
                                     np.expand_dims(self.q[:, i], axis=-1)))\
                .reshape(-1)

        print("q_mat", q_mat)
        print("p_mat", p_mat)
#       renaming for later circles
        self.q = q_mat
        self.p = p_mat

        return self.q, self.p

    def energy(self, R, centroid=False):

        """Calculate the electronic energy at the current geometry as defined
        by the systems PES.

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

#       shape (ns,ns,nb) => (nb,ns,ns)
        Potential = self.pes.diabatic_energy(R, return_matrix=True).transpose(2, 0, 1)
#       calculate the energy within matrixcalkulus
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
#       reshape the vector to max_n_beads only
        Elek_mat_energy = Elek_mat_energy.reshape(self.pes.max_n_beads)

        return Elek_mat_energy

    def gradient(self, R, centroid=False):

        """Calculate the gradient of the electronic energy at the current
        geometry as defined by the systems PES.

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

#       shape (ns,ns,nb,ndof) => (nb,ndof,ns,ns)
        Gradient = self.pes.diabatic_gradient(R, return_matrix=True).transpose(2, 3, 0, 1) 
#       calculate the gradient within matrixcalkulus
        Elek_mat_gradient = -0.5 * (np.matmul(np.matmul(np.expand_dims(self.q, axis=-1)
            .transpose(1, 2, 0), Gradient), np.expand_dims(self.q, axis=-1)
            .transpose(1, 0, 2)).reshape(-1, self.pes.max_n_beads) +
            np.matmul(np.matmul(np.expand_dims(self.p, axis=-1)
            .transpose(1, 2, 0), Gradient), np.expand_dims(self.p, axis=-1)
            .transpose(1, 0, 2)).reshape(-1, self.pes.max_n_beads) -
            np.trace(Gradient, axis1=2, axis2=3).reshape(-1, self.pes.max_n_beads))
#       reshape the vector to max_n_beads only
        Elek_mat_gradient = Elek_mat_gradient.reshape(-1)

        return Elek_mat_gradient

    def get_population(self, R, centroid=False):

        """Calculate the population estimator of the electronic mapping states
        at the current geometry as defined by the systems PES.

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
        (n_states) ndarray of floats
        """

        self.Estimator = np.zeros(self.pes.n_states)
        for i in range(self.pes.n_states):
            for j in range(self.pes.max_n_beads):
                self.Estimator[i] += (1 / (2 * self.pes.max_n_beads)) * \
                (self.q[i, j]*self.q[i, j] + self.p[i, j]*self.p[i, j] - 1.0)

#        print("Estimator",self.Estimator)

        return self.Estimator


# ========================================================================================

# alte Implementierung für die Steps

#        for i in range(self.pes.n_states):
#            for j in range(self.pes.n_states):
#                self.p[i,:] -= 0.5*self.tstep*self.pes.diabatic_energy(R,i,j)*self.q[j,:]
#        for i in range(self.pes.n_states):
#            for j in range(self.pes.n_states):
#                self.q[i,:] += self.tstep*self.pes.diabatic_energy(R,i,j)*self.p[j,:]
#        for i in range(self.pes.n_states):
#            for j in range(self.pes.n_states):
#                self.p[i,:] -= 0.5*self.tstep*self.pes.diabatic_energy(R,i,j)*self.q[j,:]
#        print("q_alt",self.q.shape)
#        print("alt_q",self.q)
#        print("alt_p",self.p)    

# alte Implementierung für die Energie

#
#                Elek_energy=np.zeros(self.pes.max_n_beads)
#
#        for i in range(self.pes.n_states):
#            for j in range(self.pes.n_states):
#          #Potential=n_beads-vector
#                Potential=self.pes.diabatic_energy(R,i,j)
#                Elek_energy[:] += 0.5*Potential[:]*\
#                (self.q[i,:]*self.q[j,:] + self.p[i,:]*self.p[j,:] - self.Einsmatrix[i,j])


# alte Implemtierung für den Gradienten

#        IMP=np.zeros_like(R)
#
#       for i in range(self.pes.n_states):
#            for j in range(self.pes.n_states):
#                Gradient=self.pes.diabatic_gradient(R,i,j)
#                for n in range(self.pes.n_dof):
#                    IMP[n,:] -= (0.5*Gradient[n,:]*\
#                     (self.q[i,n]*self.q[j,n] + self.p[i,n]*self.p[j,n] - self.Einsmatrix[i,j]))


# Zeug für die Population

#        self.m_d_0=0.0
#        for i in range(self.pes.max_n_beads):
#            for j in range(self.pes.n_states):
#                self.m_d_0*=(0.5*(self.q[i,j]*self.q[i,j] + self.p[i,j]*self.p[i,j] - 1.0)\
#                             -np.kron(j,self.initstate))

#        pop=np.zeros(self.pes.n_states)
#        for i in range(self.pes.n_states):
#        pop[i]=m_d_0*initRPdensity*Estimator[i]
