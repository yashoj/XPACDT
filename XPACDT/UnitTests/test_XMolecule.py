#!/usr/bin/env python3
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


import numpy as np
import XPACDT.Tools.Units as unit
import unittest
from XPACDT.Interfaces.XMolecule import XMolecule
from XPACDT.Input import Inputfile
import numpy as np
from copy import deepcopy


class XMoleculeTest(unittest.TestCase):
    def setUp(self):

        self.parameters = Inputfile.Inputfile(
            "FilesForTesting/InterfaceTests/input_Xmolecule_1.in")
        self.XMolecule = XMolecule(self.parameters)
        return

    def test_deepcopy(self):

        XMolecule_copy = deepcopy(self.XMolecule)
        R = self.parameters.coordinates

        self.XMolecule._calculate_adiabatic_all(R)
        XMolecule_copy._calculate_adiabatic_all(R)

        self.assertTrue(
            np.allclose(self.XMolecule._adiabatic_energy,
                        XMolecule_copy._adiabatic_energy
                        ))
        self.assertTrue(
            np.allclose(self.XMolecule._adiabatic_gradient,
                        XMolecule_copy._adiabatic_gradient
                        ))

        return

    def test_getPartialCharges(self):
        self.assertTrue(
            np.allclose(
                self.XMolecule.getPartialCharges(chargetype='mulliken'),
                np.array(
                    [0.5011398635072679, 0.46777795273720535, -0.968917816244451])
            ))

    def test_calculate_adiabatic_all(self):
        R = self.parameters.coordinates

        self.XMolecule._calculate_adiabatic_all(R)
        self.assertTrue(
            np.allclose(self.XMolecule._adiabatic_energy,
                        np.array([[-75.94374855]])))
        self.assertTrue(
            np.allclose(self.XMolecule._adiabatic_gradient,
                        np.array([[[1.09948008e-01],
                                   [0.0],
                                   [2.99179881e-02],
                                   [1.49343067e-02],
                                   [0.0],
                                   [-3.29896073e-02],
                                   [-1.24882314e-01],
                                   [0.0],
                                   [3.07161924e-03]]])
                        ))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(XMoleculeTest)
    unittest.TextTestRunner().run(suite)
