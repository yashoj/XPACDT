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

""" This module connects to XMolecule.
"""

import numpy as np
import os
import warnings
import XPACDT.Tools.Units as units
from XPACDT.Interfaces.InterfaceTemplate import PotentialInterface

try:
    import python_xmolecule_interface as XM
except ImportError:
    warnings.warn("\nXPACDT: XMOLECULE interface could not be loaded!")
    pass


class XMolecule(PotentialInterface):
    """
    XMolecule

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.

    Other Parameters (as given in the input file)
    ----------------


    Attributes
    ----------

    """

    def __init__(self, parameters, **kwargs):
        super().__init__('XMolecule',
                         parameters.n_dof,
                         # max_n_beads=max(parameters.n_beads),
                         # primary_basis='adiabatic'
                         )
        atom_symbols = [units.atom_symbol(m) for m in parameters.masses]
        pos = parameters.coordinates
        n_atom = parameters.n_dof // 3

        # Write an appropriate XMolecule input file from all set parameters
        ifile = open('input.in', 'w')
        for i in range(n_atom):
            ifile.write(
                f'{atom_symbols[i*3]}    {pos[i*3,0]:+.10f}    {pos[i*3+1,0]:+.10f}    {pos[i*3+2,0]:+.10f}\n')
        if "n_dof" in parameters['XMolecule'].keys():
            parameters['XMolecule'].pop('n_dof')
        # Iterate over entries and set in XMolecule input file.
        for k in parameters["XMolecule"].keys():
            ifile.write(k + '=' + parameters["XMolecule"][k] + ' \n')
        ifile.close()

        self.xmol = XM.xmol()
        self.xmol.init()

    def _calculate_adiabatic_all(self, R, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        -----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
            Please note that Cartesian coordinates of the atoms are used here.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        self._adiabatic_energy = np.zeros((1, R.shape[1]))
        self._adiabatic_gradient = np.zeros_like(R[np.newaxis, :])

        for bead in range(R.shape[1]):

            self.xmol.set_geom(R[:, bead].tolist())
            self._adiabatic_energy[0, bead] = self.xmol.energy
            grad = np.array(self.xmol.gradient())
            self._adiabatic_gradient[0, :, bead] = grad[:]

        self._adiabatic_energy_centroid = self._adiabatic_energy[0, 0]
        self._adiabatic_gradient_centroid = self._adiabatic_gradient[0, :, 0]

        return


############ FROM CDTK


    def hole_efields(self, X, x0):
        """
        Energy and electric field for single-hole Koopmans' theorem calculations.
        
        Input:
        X      -- 3N array of atomic positions, in Bohr (au)
        x0     -- array of dimension 3 of electronic position, in Bohr (au)
        
        Output:
        e      -- Array of energies for all single-hole states in Hartree (au)
        efield -- Electric field at the electron position
        
        """
        self.xmolecule.set_geom(X.tolist())
        l = self.xmolecule.hole_efields(x0.tolist())
        nstates = len(l)
        ncoords = len(x0.tolist())

        occ = []
        e = np.zeros(nstates)
        efield = np.zeros((nstates,ncoords))

        for i, li in enumerate(l):
            occi = li[0]
            ei = li[1]
            efieldi = li[2]

            occ.append(occi)

            e[i] = ei
            efield[i] = efieldi

        return occ, e, efield

    def set_occ(self, state):
        """
        Set occupation number to given state.
        """

        self.xmolecule.set_occ(state)

    def get_occ(self):
        """
        Get occupation number 
        """

        return self.xmolecule.get_occ()

    def set_geom(self, X):
        """
        Set geometry to X

        X   -- 3N array of atomic positions, in Bohr (au)
        """

        # pass on geometry in Bohr





    def energy_HF(self, X, Time=0.0, **opts):
        """
        Hartree-Fock energy calculation. 

        Input:
        X   -- 3N array of atomic positions, in Bohr (au)

        Output:
        e   -- SCF energy in atomic units
        """

        # pass on geometry in Bohr
        self.xmolecule.set_geom(X.tolist())
        e = self.xmolecule.energy

        return e

    def energy_gradient_HF(self, X, Time=0.0, **opts):
        """
        Hartree-Fock energy and gradient calculation.

        Input:
        X   -- 3N array of atomic positions, in Bohr (au)
        
        Output:
        e   -- SCF energy in atomic units
        g   -- 3N SCF gradient in atomic units
        """

        # pass on geometry in Bohr
        self.xmolecule.set_geom(X.tolist())
        e = self.xmolecule.energy
        g = np.array(self.xmolecule.gradient())

        return e, g

    def hole_properties(self, X, Time=0.0, **opts):
        """
        Energy, gradient, non-adiabatic coupling for single-hole Koopmans' theorem calculations.

        Input:
        X   -- 3N array of atomic positions, in Bohr (au)
        
        Output:
        e   -- Array of energies for all single-hole states in atomic units (hartree)
        g   -- Array of 3N gradient for all single-hole states in atomic units
        nac -- 2D array of 3N non-adiabatic coupling between all single-hole states in atomic units
        """

        # pass on geometry in Bohr
        self.xmolecule.set_geom(X.tolist())
        l = self.xmolecule.hole_properties()
        
        nstates = len(l)
        ncoords = len(X.tolist())

        occ = []
        e = np.zeros(nstates)
        g = np.zeros((nstates, ncoords))
        nac = {}

        for i, li in enumerate(l):
            occi = li[0]
            ei = li[1]
            gi = li[2]
            naci = li[3]
            occ.append(occi)
            e[i] = ei
            g[i] = np.array(gi)
            for j, nacij in enumerate(naci):
                nac[(i,j)] = np.array(nacij)
#            nac[i] = np.array(naci)
        return occ, e, g, nac

    def CI_energy(self, X, Time=0.0,state=None, **opts):        
        self.xmolecule.set_geom(X.tolist())
        en=np.array(self.xmolecule.CI_energies())
        if(state != None):
            return(en)
        else:
            return(en[state])

    def CI_gradient(self, X, Time=0.0,state=None, **opts):        
        if state==None:
            state=0
        # only along z-direction
        g=nderiv.gradient(lambda x: self.CI_energy(x,state=state), X ,ms=np.array([[0,0,1,0,0,-1]]),step=0.005,fast=True)
        G=np.zeros(len(X))
        G[2]=0.5*g[0]
        G[5]=-0.5*g[0]
        return(G)                

    def CIS_properties(self, X, Time=0.0,state=0, **opts):        
        """
        Energy, gradient, non-adiabatic coupling for CIS states.

        Input:
        X   -- 3N array of atomic positions, in Bohr (au)
        state   -- state index
        
        Output:
        e   -- Array of energies for all single-hole states in atomic units (hartree)
        g   -- Array of 3N gradient for all single-hole states in atomic units
        nac -- 2D array of 3N non-adiabatic coupling between all single-hole states in atomic units
        """
        # pass on geometry in Bohr
        self.xmolecule.set_geom(X.tolist())
        nstates = self.nstates
        ncoords = len(X.tolist())

        en,gr,nacs=self.xmolecule.CIS_energy_gradient_nacs(state)        
        state_labels = []
        nac = {}
        e = en[0:nstates].copy()
        g = gr[0:nstates,:].copy()
        
        for i in range(0,nstates):
            state_labels.append(i)
            for j in range(0,nstates):
                nac[(i,j)] = nacs[i,j,:].copy()
        del nacs
        del en
        del gr

#        import gc
#        gc.collect()

############# test
        # e = np.zeros(nstates)
        # g = np.zeros((nstates, ncoords))
        # nac = {}
        # state_labels = []
        # for i in range(0,nstates):
        #     state_labels.append(i)
        #     for j in range(0,nstates):
        #         nac[(i,j)] = 0.0


        return state_labels, e, g, nac

    def getGamma(self):
        """
        Obtain rates and cross sections at current position.
        """

        return self.xmolecule.process()


    def getPartialCharges(self,chargetype='mulliken'):
        """
        Returns current partial charges of the molecule accoring to given method.
        """
        
        if chargetype == 'mulliken':
            return self.xmolecule.mullikencharge()
        else:
            print("Charge type not implemented in XMolecule Interface")
            exit(-1)

    def getPopulation(self):
        """
        Returns current Population Analysis data.
        """        
        return self.xmolecule.population()

    def getOrbitalEnergies(self):
        """
        Returns current Orbital Energies.
        """        
        return self.xmolecule.orbital_energies()

    def getBondOrders(self,botype='Meyer'):
        """
        Returns current BondOrder Matrix of the molecule accoring to given method.
        """        
        if botype == 'Meyer':
            return self.xmolecule.bondorder()
        else:
            print("Charge type not implemented in XMolecule Interface")
            exit(-1)

    def getAbsorptionCrossSections(self):
        """
        Returns absorption cross sections between current molecular orbitals

        aCS[i,j] = absorption cross section from MO i to j

        Spin orbital factor included.

        The corresponding energy for the absorption can be obtained from the MO energy difference.
        """

        return self.xmolecule.absorptioncs()

    def getFluorescenceRates(self):
        """
        Returns fluorescence rates between current molecular orbitals

        fluoRate[i,j] = fluorescence rate from MO i to j

        Spin orbital factor included.

        The corresponding energy for the fluorescence photon can be obtained from the MO energy difference.
        """

        return self.xmolecule.fluorescencerates()
