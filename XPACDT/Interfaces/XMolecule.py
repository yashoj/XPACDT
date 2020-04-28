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
import functools

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

    def __determineYesNo(self, val):
        valU = val.upper()
        if \
                valU == "YES" \
                or valU == "Y" \
                or valU == "ON" \
                or valU == "T" \
                or valU == "TRUE":
            return True
        elif \
                valU == "NO" \
                or valU == "N" \
                or valU == "OFF" \
                or valU == "F" \
                or valU == "FALSE":
            return False
        raise ValueError(f" Boolean option {val} unclear.")

    def __getstate__(self):
        """
        get dictionary of all class elements.
        instead of xmol return xmol_state.
        xmol will be initialized from xmol_state at a later point.
        """
        state = {}
        for k, v in self.__dict__.items():
            if k == "xmol":
                state["xmol_state"] = self.xmol.__getstate__()
            else:
                state[k] = self.__dict__[k]
        return state

    def __setstate__(self, state):
        """
        set all class elements.
        """
        for k, v in state.items():
            self.__dict__[k] = state[k]

    def __check_xmol(func):
        """
        This decorate makes sure that xmol object is properly set up.
        If the Xmolecule class is copied, copying of self.xmol is prohibited.
        Instead a 'pickled' version of self.xmol is copied (self.xmol_state).
        See implementation of __setstate__ / __getstate__.
        In case a proper initialisation of xmol object is required
        this decorator takes care of it by calling
        xmol.__set_state__(self.xmol_state)
        """
        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            self = args[0]
            if not hasattr(self, "xmol"):
                print("setting up xmolecule")
                self.xmol = XM.xmol()
                self.xmol.__setstate__(self.xmol_state)
            value = func(*args, **kwargs)
            return value
        return wrapper_decorator

    def __init__(self, parameters, **kwargs):
        atom_symbols = [units.atom_symbol(m) for m in parameters.masses]
        pos = parameters.coordinates
        n_atom = parameters.n_dof // 3
        self.calcType = "singleState"
        # todo: xmolecule does not distinguish between upper / lower case
        # xpacdt does!
        if "CIS" in parameters['XMolecule'].keys():
            if self.__determineYesNo(parameters['XMolecule']["CIS"]):
                self.calcType = "CIS"
        if "gs_occ" in parameters['XMolecule'].keys():
            if self.__determineYesNo(parameters['XMolecule']["gs_occ"]):
                self.calcType = "KoopmanHole"

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

        # determine number of states
        n_states = 0
        if "nstates" in parameters['XMolecule'].keys():
            n_states = int(parameters['XMolecule']["nstates"])
        else:
            if self.calcType == "CIS":
                energies, grads, nacs = self.xmol.CIS_energy_gradient_nacs(
                    state)
                n_states = energies.shape
            elif self.calcType == "Koopmanhole":
                holeProperties = self.xmol.hole_properties()
                n_states = len(holeProperties)
            elif self.calcType == "singleState":
                n_states = 1

        super().__init__('XMolecule',
                         parameters.n_dof,
                         n_states=n_states,
                         max_n_beads=max(parameters.n_beads),
                         primary_basis='adiabatic'
                         )

    @__check_xmol
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

        self._adiabatic_energy = np.zeros((self.n_states, R.shape[1]))
        self._adiabatic_gradient = np.zeros((self.n_states, self.n_dof,
                                             self.max_n_beads))

        print(f"state : {S}")
        for bead in range(R.shape[1]):
            self.xmol.set_geom(R[:, bead].tolist())
            if self.calcType == "Koopmanhole":
                holeProperties = self.xmol.hole_properties()
                assert len(holeProperties) >= self.n_states, \
                    f"Number of available holes in Koopmans space is smaller than requested number of states ( {len(holeProperties)} < {self.n_states})."
                for i, (occi, ei, gi, naci) in enumerate(holeProperties):
                    if i >= self.n_states:
                        break
                    self._adiabatic_energy[i, bead] = ei
                    self._adiabatic_gradient[i, :, bead] = np.array(gi)
                    for j, nacij in enumerate(naci):
                        self._nac[i, j, :, bead] = nacij[:].copy()
            elif self.calcType == "CIS":
                # if option CIS was selected, we need to have a valid state argument
                if S is None:
                    state = 0
                else:
                    try:
                        state = int(S)
                    except:
                        raise ValueError(f"CIS state index {S} cannot be read")
                energies, grads, nacs = self.xmol.CIS_energy_gradient_nacs(
                    state)
                assert energies.shape[0] >= self.n_states, \
                    f"Number of available states in CIS space smaller than requested ( {en.shape[0]} < {self.n_states})."
                self._adiabatic_energy[:,
                                       bead] = energies[0:self.n_states].copy()
                self._adiabatic_gradient[:, :,
                                         bead] = grads[0:self.n_states, :].copy()
                self._nac[:, :, :, bead] = nacs[0:self.n_states,
                                                0:self.n_states, :].copy()
            elif self.calcType == "singleState":
                self._adiabatic_energy[0, bead] = self.xmol.energy
                grad = np.array(self.xmol.gradient())
                self._adiabatic_gradient[0, :, bead] = grad[:]
        # set centroid values to values of the first bead
        self._adiabatic_energy_centroid = self._adiabatic_energy[:, 0]
        self._adiabatic_gradient_centroid = self._adiabatic_gradient[:, :, 0]
        if self.n_states > 1:
            self._nac_centroid = self._nac[:, :, :, 0]
        return

    @__check_xmol
    def set_occ(self, occ):
        """
        Set occupation number to given configuration.

        Input Parameter:
        -------------------
        occ: list of occupation numbers with values = 0,1,2
        """
        # make sure that occ has the same shape as what is returned from get_occ()
        currentOcc = self.xmol.get_occ()
        occ = (occ + len(currentOcc) * [0])[:len(currentOcc)]
        for o in occ:
            assert isinstance(o, int), f"configuration {occ} not valid."
            assert 0 <= o, f"configuration {occ} not valid."
            assert o <= 2, f"configuration {occ} not valid."

        self.xmol.set_occ(occ)

    @__check_xmol
    def get_occ(self):
        """
        Returns occupation number.

        Returns:
        ----------
        occ: list of occupation numbers with values = 0,1,2
        """
        return self.xmol.get_occ()

    @__check_xmol
    def getPartialCharges(self, chargetype='mulliken'):
        """
        Returns current partial charges of the molecule accoring to given method.
        Returns:
        ----------
        partialcharges : list of floats len= number of atoms
                         partial charge for each atom
        """
        if chargetype == 'mulliken':
            return self.xmol.mullikencharge()
        else:
            print("Charge type not implemented in XMolecule Interface")
            exit(-1)

    @__check_xmol
    def getPopulation(self):
        """
        Returns current Population Analysis data.
        Returns:
        ----------
        population : list of populations for each molecular orbital:
                     [pop_orb] (len = n_orb),
                     where pop_orb is a list of floats, len = n_atoms,
                     where the floats are atomic populations for the
                     respective orbital
        """
        return self.xmol.population()

    @__check_xmol
    def getOrbitalEnergies(self):
        """
        Returns current Orbital Energies.
        Returns:
        ----------
        orbital_energies : list of orbital energies. len = number of orbitals
        """
        return self.xmol.orbital_energies()

    @__check_xmol
    def getBondOrders(self, botype='Meyer'):
        """
        Returns current BondOrder Matrix of the molecule accoring to given method.
        Returns:
        ---------
        bondorder_values: 2-dimensional list, shape=(n_atom,n_atom)
                          containing bondorder values between atoms
        """
        if botype == 'Meyer':
            return self.xmol.bondorder()
        else:
            print("Charge type not implemented in XMolecule Interface")
            exit(-1)

    @__check_xmol
    def getAbsorptionCrossSections(self):
        """
        Returns absorption cross sections between current molecular orbitals
        Returns:
        ----------
        aCS : 2-dimensional list (shape = n_orbitals,n_orbitals)
              aCS[i][j] is the absorption cross section from MO i to j
              Spin orbital factor included.
              The corresponding energy for the absorption can be obtained from the MO energy difference.
        """

        return self.xmol.absorptioncs()

    @__check_xmol
    def getFluorescenceRates(self):
        """
        Returns fluorescence rates between current molecular orbitals
        Returns:
        ----------
        fluoRate : 2-dimensional list (shape = n_orbitals,n_orbitals)
              fluoRate[i][j] = fluorescence rate from MO i to j
              Spin orbital factor included.
              The corresponding energy for the absorption can be obtained from the MO energy difference.
        """
        return self.xmol.fluorescencerates()

    @__check_xmol
    def getRates(self):
        """
        Obtain rates and cross sections at current position.
        Returns:
        ----------
        process_data: list of tuples (tag, final_occ, value, E_eV)
                      tag (char) inidcates the process type ('P','A','F')
                      final_occ (list of ints) are the final occupation numbers
                      values (float) is the rate or cross section
                      E_eV (int) is the transition energy in eV
        """

        return self.xmol.process()


# FROM CDTK
# things below need to be implemented

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
        efield = np.zeros((nstates, ncoords))

        for i, li in enumerate(l):
            occi = li[0]
            ei = li[1]
            efieldi = li[2]

            occ.append(occi)

            e[i] = ei
            efield[i] = efieldi

        return occ, e, efield

    def CI_energy(self, X, Time=0.0, state=None, **opts):
        self.xmolecule.set_geom(X.tolist())
        en = np.array(self.xmolecule.CI_energies())
        if(state != None):
            return(en)
        else:
            return(en[state])

    def CI_gradient(self, X, Time=0.0, state=None, **opts):
        if state == None:
            state = 0
        # only along z-direction
        g = nderiv.gradient(lambda x: self.CI_energy(x, state=state), X, ms=np.array(
            [[0, 0, 1, 0, 0, -1]]), step=0.005, fast=True)
        G = np.zeros(len(X))
        G[2] = 0.5*g[0]
        G[5] = -0.5*g[0]
        return(G)
