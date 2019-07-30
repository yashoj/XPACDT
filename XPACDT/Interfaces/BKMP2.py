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

""" This module connects to the BKMP2 H3 PES."""

import numpy as np
import XPACDT.Interfaces.BKMP2_module.pot as pot

import XPACDT.Interfaces.InterfaceTemplate as itemplate


class BKMP2(itemplate.PotentialInterface):
    """
    BKMP2 PES
    """
    def __init__(self, **kwargs):
        pot.pes_init()
        itemplate.PotentialInterface.__init__(self, "BKMP2")

    def _calculate_all(self, R, P=None, S=None):
        """
        Calculate the value of the potential and the gradient at positions R.

        Parameters:
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
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

        self._energy = np.zeros(1)
        self._gradient = np.zeros((1, 9))

        # centroid part if more than 1 bead
        if R.shape[1] > 0: #1:
            centroid = np.mean(R, axis=1)

            self._energy_centroid, self._gradient_centroid = pot.pot(centroid)
            self._energy[0], self._gradient[0] = self._energy_centroid, self._gradient_centroid

        # beads part
        pass

        return


if __name__ == "__main__":
    pes = BKMP2()
    print(pes.name)
    x=np.zeros(9)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0 
    x[3] = 1.7566 + 0.0001
    x[4] = 0.0
    x[5] = 0.0
    x[6] = 2.0 * x[3]
    x[7] = 0.0
    x[8] = 0.0
    
    pes._calculate_all(x[:, None])
    print(pes._energy, pes._gradient)
    print(pes.energy(x[:, None]))
    pes.plot_1D(x, 6, 3.0, 10.0, 0.1)
    pes.plot_1D(x, 6, 3.0, 10.0, 0.1, relax=True)