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

""" This module connects to the LWAL F+H2 PES.
G. Li, H.-J. Werner, F. Lique, and M. H. Alexander, J. Chem. Phys. 127, 174302 (2007).
"""

import numpy as np
import os
import XPACDT.Interfaces.LWAL_module.pot as pot

import XPACDT.Interfaces.InterfaceTemplate as itemplate
import XPACDT.Tools.Geometry as geom


class LWAL(itemplate.PotentialInterface):
    """
    LWAL PES. No parameters required.

    The ordering of the atoms is as follows:
    F, H, H

    G. Li, H.-J. Werner, F. Lique, and M. H. Alexander, J. Chem. Phys. 127, 174302 (2007).
    """
    def __init__(self, max_n_beads=1, **kwargs):
        self.__data_path = os.path.dirname(pot.__file__) + "/"
        pot.pes_init()
        itemplate.PotentialInterface.__init__(self, "LWAL", 9, 1,
                                              max_n_beads, 'adiabatic')

    def _calculate_adiabatic_all(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        -----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
            Please note that Cartesian coordinates of the atoms are used and 
            have to be ordered in the following way: F, H, H
        P : (n_dof, n_beads) ndarray of floats, optional
            The momenta of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads. This is not
            used in this potential and thus defaults to None.
        S : int, optional
            The current electronic state. This is not used in this potential
            and thus defaults to None.
        """

        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"

        self._adiabatic_energy = np.zeros((1, R.shape[1]))
        self._adiabatic_gradient = np.zeros_like(R[np.newaxis, :])

        self._adiabatic_energy_centroid = np.zeros(1)
        self._adiabatic_gradient_centroid = np.zeros((1, R.shape[0]))

        # centroid part if more than 1 bead
        if R.shape[1] > 1:
            centroid = np.mean(R, axis=1)
            self._adiabatic_energy_centroid[0], self._adiabatic_gradient_centroid[0] = pot.pot(centroid, self.__data_path)

        for i, r in enumerate(R.T):
            self._adiabatic_energy[0, i], self._adiabatic_gradient[0, :, i] = pot.pot(r, self.__data_path)

        if R.shape[1] == 1:
            self._adiabatic_energy_centroid = self._adiabatic_energy[:, 0]
            self._adiabatic_gradient_centroid = self._adiabatic_gradient[:, :, 0]

        return

    def _from_cartesian_to_internal(self, R):
        """Transform from full cartesian coordinates to internal Jacobi
        coordinates. The order of the atoms has to be F, H, H.
        The Jacobi coordinates are defined as follows:
            r = internal[0] = Distance between the first and second H in au.
            R = internal[1] = Distance between the F and the center of
                              the two H's in au.
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

        # r
        r_vec = R[3:6]-R[6:9]
        internal[0] = np.linalg.norm(r_vec)

        # R
        R_vec = 0.5 * (R[3:6]+R[6:9]) - R[0:3]
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
            r = internal[0] = Distance between the first and second H in au.
            R = internal[1] = Distance between the F and the center of
                              the two H's in au.
            phi = internal[2] = angle between the two vectors that define
                                r and R.
        The output cartesian coordinates are in the xy plane. The center of 
        mass of H2 is fixed to the origin. The two H are displaced along the
        x-axis in negative and positive direction, respectively. The F is then
        placed in the xy-plane according to 'R' and 'phi'.

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
        R[3] = -0.5 * internal[0]
        R[6] = 0.5 * internal[0]

        # from R and phi, fixed to x-y-plane
        R[0] = (internal[1]) * np.cos(internal[2])
        R[1] = (internal[1]) * np.sin(internal[2])

        return R


# if __name__ == "__main__":
#     pes = LWAL()
#     print(pes.name)
#     x=np.zeros(9)
#     for i in range(1000):
#         phi = np.random.rand(1)*2.0*np.pi
#         x[0] = 3.0*np.cos(phi)
#         x[1] = 3.0*np.sin(phi)
#         x[2] = 0.0 
#         x[3] = -1.0
#         x[4] = 0.0
#         x[5] = 0.0
#         x[6] = 1.0
#         x[7] = 0.0
#         x[8] = 0.0
        
        
# #        if phi > -11.0:
# #            inte = pes._from_cartesian_to_internal(x)
# ##            print(phi, inte[2], 2*np.pi-inte[2], inte[2]+phi, inte[2]-phi)
# #            y = pes._from_internal_to_cartesian(inte)
# ##            print(x, y)
# #   
# #            print((abs(x-y) < 1e-8).all())
# ##            print()
    
#     pes._calculate_all(x[:, None])
#     print(pes._adiabatic_energy, pes._adiabatic_gradient)
#     print(pes.adiabatic_energy(x[:, None]))
#     internal = np.array([2.0, 5.0, 0.0])
#     pes.plot_1D(internal, 1, 4.0, 9.0, 0.1, relax=False, internal=True)
# #    pes.plot_1D(internal, 0, 2.0, 10.0, 0.1, relax=True)
    
#     pes.plot_2D(internal, 0, 1, (1.0, 2.0), (3.0, 9.0), (0.2, 0.2), relax=False, internal=True)
# #    pes.plot_2D(internal, 0, 1, (0.5, 2.0), (3.5, 7.0), (0.2, 0.2), relax=True, internal=True)
# #    pes.plot_2D(internal, 2, 0.0, 2*np.pi, 0.1, relax=True, internal=True)
