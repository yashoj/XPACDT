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

""" This module connects to any triatomic fitted PES.

Implemented are:

H3 - BKMP2:
A. I. Boothroyd, W. J. Keogh, P. G. Martin, and M. R. Peterson,
J. Chem. Phys. 104, 7139 (1996).

ClH2 - CW:
G. Capecchi and H. J. Werner, Phys. Chem. Chem. Phys. 6, 4975 (2004).

FH2 - LWAL:
G. Li, H.-J. Werner, F. Lique, and M. H. Alexander, J. Chem. Phys. 127, 174302 (2007).
"""

import importlib
import numpy as np
import os

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Tools.Geometry as geom
import XPACDT.Tools.Units as units

from XPACDT.Input.Error import XPACDTInputError


class Triatomic(itemplate.PotentialInterface):
    """
    Triatomic PES.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.

    Other Parameters (as given in the input file)
    ----------------
    name : string
        The name of the PES requested.

    Attributes
    ----------
    pes_name
    available_pes
    """
    def __init__(self, n_dof=9, **parameters):
        self.__data_path = os.path.dirname(pot.__file__) + "/"
        pot.pes_init()
        if n_dof != 3 and n_dof != 9:
            raise XPACDTInputError(
                f"Inferred number of degree of freedom is {n_dof}, but "
                "should be either 3 or 9 for CW.",
                section="CW")

        super().__init__("CW",
                         n_dof=9, n_states=1, primary_basis='adiabatic',
                         **parameters)
        # For proper Hessian derivatives! Numerically tested for stability!
        self._DERIVATIVE_STEPSIZE = 7e-3

        pes_parameters = parameters.get(self.name)
        self.__pes_name = pes_parameters.get('name')

        if self.pes_name not in self.available_pes:
            raise RuntimeError("\nXPACDT: The requested triatomic pes is not implemented: " + self.pes_name
                               + " Available: " + str(self.available_pes.keys()))

        try:
            self.__pot = importlib.import_module("XPACDT.Interfaces."+ self.pes_name + "_module.pot")
        except ModuleNotFoundError as e:
            raise type(e)(str(e) + "\nXPACDT: One of the compiled triatomic PES ("
                          + self.pes_name + ") could not be imported. Please make sure"
                          " that it was properly compiled.")

        self.__pot.pes_init()

        self.__data_path = os.path.dirname(self.__pot.__file__) + "/"
        self.__masses = self.available_pes.get(self.pes_name).get('masses')

        if self.pes_name == 'CW':
            # For proper Hessian derivatives! Numerically tested for stability!
            self._DERIVATIVE_STEPSIZE = 7e-3

    @property
    def pes_name(self):
        """ str : Name of the instantiated triatomic PES."""
        return self.__pes_name

    @property
    def available_pes(self):
        """ Dictonary of the implemented PES routines. The keys are the names 
        of the implemented PES. Each value will be a dictonary, that holds
        the masses of the associated atoms in au as a list."""
        return {'BKMP2': { 'masses': [units.atom_mass('H'), units.atom_mass('H'), units.atom_mass('H')]},
                 'LWAL': { 'masses': [units.atom_mass('F'), units.atom_mass('H'), units.atom_mass('H')]},
                 'CW': { 'masses': [units.atom_mass('Cl'), units.atom_mass('H'), units.atom_mass('H')]}
               }



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

        self._adiabatic_energy_centroid = np.zeros(1)
        self._adiabatic_gradient_centroid = np.zeros((1, R.shape[0]))

        # centroid part if more than 1 bead
        if R.shape[1] > 1:
            centroid = np.mean(R, axis=1)
            self._adiabatic_energy_centroid[0], self._adiabatic_gradient_centroid[0] = self.__pot.pot(centroid, self.__data_path)

        for i, r in enumerate(R.T):
            self._adiabatic_energy[0, i], self._adiabatic_gradient[0, :, i] = self.__pot.pot(r, self.__data_path)

        if R.shape[1] == 1:
            self._adiabatic_energy_centroid = self._adiabatic_energy[:, 0]
            self._adiabatic_gradient_centroid = self._adiabatic_gradient[:, :, 0]

        return

    def _from_cartesian_to_internal(self, R):
        """Transform from full cartesian coordinates to internal Jacobi
        coordinates. The Jacobi coordinates are defined as follows:
            r = internal[0] = Distance between the second and third atom in au.
            R = internal[1] = Distance between the first atom and the center of
                              mass of the last two atoms in au.
            phi = internal[2] = angle between the two vectors that define
                                r and R.

        Parameters
        ----------
        R: (9) ndarray of floats
            Values of the cartesian coordinates in au.

        Returns
        -------
        internal : (3) ndarray of floats
            Values of the Jacobi coordinates associated with the input
            cartesian coordinates in au.
        """
        internal = np.zeros(3)

        # r (Always the last two atoms).
        r_vec = R[3:6]-R[6:9]
        internal[0] = np.linalg.norm(r_vec)
        
        # R 
        R_vec = (1.0/(self.__masses[1]+self.__masses[2])) * (self.__masses[1]*R[3:6] + self.__masses[2]*R[6:9]) - R[0:3]
        internal[1] = np.linalg.norm(R_vec)

        # phi
        internal[2] = geom.angle(r_vec, R_vec)
        # Correct angle definition to range :math:`0 : 2\pi`
        if R[1] < 0.0:
            internal[2] = 2.0*np.pi-internal[2]

        return internal

    def _from_internal(self, internal):
        return self._from_internal_to_cartesian(internal)

    def _from_internal_to_cartesian(self, internal):
        """Transform from Jacobi coordinates to full cartesian coordinates. The
        Jacobi coordinates are defined as follows:
            r = internal[0] = Distance between the second and third atom in au.
            R = internal[1] = Distance between the first atom and the center of
                              mass of the last two atoms in au.
            phi = internal[2] = angle between the two vectors that define
                                r and R.
        The output cartesian coordinates are in the xy plane. The center of
        mass of the last two atoms is fixed to the origin. The last two atoms are 
        displaced along the x-axis in negative and positive direction, respectively. 
        The first atom is then placed in the xy-plane according to 'R' and 'phi'.

        Parameters
        ----------
        internal: (3) ndarray of floats
            Values of the Jacobi coordinates in au.

        Returns
        -------
        R : (9) ndarray of floats
            Cartesian coordinates associated with the given internal
            coordinates with the orientation defined above.
        """

        R = np.zeros(9)

        # from r, fixed to x-axis
        R[3] = -self.__masses[1]*internal[0] / (self.__masses[1] + self.__masses[2])
        R[6] = self.__masses[2]*internal[0] / (self.__masses[1] + self.__masses[2])

        # from R and phi, fixed to x-y-plane
        R[0] = (internal[1]) * np.cos(internal[2])
        R[1] = (internal[1]) * np.sin(internal[2])

        return R
