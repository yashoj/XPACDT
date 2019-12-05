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

"""TODO"""
import numpy as np
import XPACDT.System.Electrons as electrons
import XPACDT.Tools.Units as units


class NRPMDElectrons(electrons.Electrons):
    """ TODO
    """

    
    def __init__(self, parameters, n_beads, R=None, P=None):


        electrons.Electrons.__init__(self, "NRPMDElectrons", parameters, n_beads, 'diabatic')

#        self.n_states=int(parameters.get("NRPMDElektrons").get("n_states"))
#        self.n_beads=int(parameters.get("Nuclei").get("n_beads"))
        self.initstate=int(parameters.get("NRPMDElectrons").get("initial_states"))
        self.tstep=units.parse_time(parameters.get("NRPMDElectrons").get("timestep"))
#        self.inittime=int(parameters.get("system").get("time"))
#        self.n_atoms=int(parameters.get("Nuclei").get("n_atoms"))
        
        self.q = np.zeros((self.pes.n_states,self.pes.max_n_beads))
        self.p = np.zeros((self.pes.n_states,self.pes.max_n_beads))
        self.Einsmatrix = np.identity(self.pes.n_states)
        angles = 2.0*np.pi*np.random.random((self.pes.n_states,self.pes.max_n_beads))           
       
        self.q = np.sin(angles)
        self.p = np.cos(angles)
        
        self.q[self.initstate,:] *= np.sqrt(3)
        self.p[self.initstate,:] *= np.sqrt(3)

#        print(self.q)
#        print(self.p)
       
    def step(self, R, **kwargs):
        """Calculate the stepwise propagation of position and momentum of the system electrons as defined
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
        (n_states, n_beads) ndarrays of float /or/ float
        The position and momenta of the systems electrons of the systems PES at each bead position or at the centroid
        in hartree.
        """
# doing the exact step from Richardson and Thoss
        
        for i in range(self.pes.n_states):
            for j in range(self.pes.n_states):
                self.p[i,:] -= 0.5*self.tstep*self.pes.diabatic_energy(R,i,j)*self.q[j,:]
        for i in range(self.pes.n_states):
            for j in range(self.pes.n_states):
                self.q[i,:] += self.tstep*self.pes.diabatic_energy(R,i,j)*self.p[j,:]
        for i in range(self.pes.n_states):
            for j in range(self.pes.n_states):
                self.p[i,:] -= 0.5*self.tstep*self.pes.diabatic_energy(R,i,j)*self.q[j,:]

 #       print(self.q)
 #       print(self.p)

        return self.q,self.p

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
        Elek_energy=np.zeros(self.pes.max_n_beads)

        for i in range(self.pes.n_states):
            for j in range(self.pes.n_states):
          #Potential=n_beads-vector
                Potential=self.pes.diabatic_energy(R,i,j)
                Elek_energy[:] += 0.5*Potential[:]*\
                (self.q[i,:]*self.q[j,:] + self.p[i,:]*self.p[j,:] - self.Einsmatrix[i,j])
        
  #      print(self.pes.diabatic_energy)
    #    print(Elek_energy)
            
        return Elek_energy

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
        IMP=np.zeros_like(R)

        for i in range(self.pes.n_states):
            for j in range(self.pes.n_states):
                Gradient=self.pes.diabatic_gradient(R,i,j)
                for n in range(self.pes.n_dof):
                    IMP[n,:] -= (0.5*Gradient[n,:]*\
                    (self.q[i,n]*self.q[j,n] + self.p[i,n]*self.p[j,n] - self.Einsmatrix[i,j]))

        return IMP
 