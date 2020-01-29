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
import unittest

import XPACDT.Interfaces.TullyModel as tullym
import XPACDT.Input.Inputfile as infile


class TullyModelTest(unittest.TestCase):

    def setUp(self):
        self.pes_A_1_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyA_1.in"))
        self.pes_A_2_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyA_2.in"))

        self.pes_B_1_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyB_1.in"))
        self.pes_B_2_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyB_2.in"))

        self.pes_C_1_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyC_1.in"))
        self.pes_C_2_nb = tullym.TullyModel(infile.Inputfile("FilesForTesting/InterfaceTests/input_TullyC_2.in"))

        return

    def test_creation(self):
        self.assertEqual(self.pes_A_1_nb.name, 'TullyModel')
        self.assertEqual(self.pes_A_1_nb.model_type, 'model_A')

        return

    def test_calculate_diabatic_all(self):
        # Test correct diabatic potential values and gradients for all states

        # Model A: 1 bead
        self.pes_A_1_nb._calculate_diabatic_all(np.array([[0.0]]))
        np.testing.assert_allclose(
                self.pes_A_1_nb._diabatic_energy, [[[0.0], [0.005]],
                                                   [[0.005], [0.0]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_1_nb._diabatic_gradient, [[[[0.016]], [[0.0]]],
                                                     [[[0.0]], [[-0.016]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_1_nb._diabatic_energy_centroid, [[0.0, 0.005], [0.005, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_1_nb._diabatic_gradient_centroid, [[[0.016], [0.0]],
                                                              [[0.0], [-0.016]]], rtol=1e-7)

        # Model A: 2 beads
        self.pes_A_2_nb._calculate_diabatic_all(np.array([[1.0e5, -1.0e5]]))
        np.testing.assert_allclose(
                self.pes_A_2_nb._diabatic_energy, [[[0.01, -0.01], [0.0, 0.0]], 
                                                   [[0.0, 0.0], [-0.01, 0.01]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_2_nb._diabatic_gradient, [[[[0.0, 0.0]], [[0.0, 0.0]]], 
                                                     [[[0.0, 0.0]], [[0.0, 0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_2_nb._diabatic_energy_centroid, [[0.0, 0.005], [0.005, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_A_2_nb._diabatic_gradient_centroid, [[[0.016], [0.0]],
                                                              [[0.0], [-0.016]]], rtol=1e-7)
    
        # Model B: 1 bead
        self.pes_B_1_nb._calculate_diabatic_all(np.array([[0.0]]))
        np.testing.assert_allclose(
                self.pes_B_1_nb._diabatic_energy, [[[0.0], [0.015]],
                                                   [[0.015], [-0.05]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_1_nb._diabatic_gradient, [[[[0.0]], [[0.0]]],
                                                     [[[0.0]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_1_nb._diabatic_energy_centroid, [[0.0, 0.015],
                                                            [0.015, -0.05]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_1_nb._diabatic_gradient_centroid, [[[0.0], [0.0]],
                                                              [[0.0], [0.0]]], rtol=1e-7)

        # Model B: 2 beads
        self.pes_B_2_nb._calculate_diabatic_all(np.array([[1.0e5, -1.0e5]]))
        np.testing.assert_allclose(
                self.pes_B_2_nb._diabatic_energy, [[[0.0, 0.0], [0.0, 0.0]], 
                                                   [[0.0, 0.0], [0.05, 0.05]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_2_nb._diabatic_gradient, [[[[0.0, 0.0]], [[0.0, 0.0]]], 
                                                     [[[0.0, 0.0]], [[0.0, 0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_2_nb._diabatic_energy_centroid, [[0.0, 0.015],
                                                            [0.015, -0.05]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_B_2_nb._diabatic_gradient_centroid, [[[0.0], [0.0]],
                                                              [[0.0], [0.0]]], rtol=1e-7)

        # Model C: 1 bead
        self.pes_C_1_nb._calculate_diabatic_all(np.array([[0.0]]))
        np.testing.assert_allclose(
                self.pes_C_1_nb._diabatic_energy, [[[0.0006], [0.1]],
                                                   [[0.1], [-0.0006]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_1_nb._diabatic_gradient, [[[[0.0]], [[0.09]]],
                                                     [[[0.09]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_1_nb._diabatic_energy_centroid, [[0.0006, 0.1],
                                                            [0.1, -0.0006]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_1_nb._diabatic_gradient_centroid, [[[0.0], [0.09]],
                                                              [[0.09], [0.0]]], rtol=1e-7)

        # Model C: 2 beads
        self.pes_C_2_nb._calculate_diabatic_all(np.array([[1.0e5, -1.0e5]]))
        np.testing.assert_allclose(
                self.pes_C_2_nb._diabatic_energy, [[[0.0006, 0.0006], [0.2, 0.0]], 
                                                   [[0.2, 0.0], [-0.0006, -0.0006]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_2_nb._diabatic_gradient, [[[[0.0, 0.0]], [[0.0, 0.0]]], 
                                                     [[[0.0, 0.0]], [[0.0, 0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_2_nb._diabatic_energy_centroid, [[0.0006, 0.1],
                                                            [0.1, -0.0006]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_C_2_nb._diabatic_gradient_centroid, [[[0.0], [0.09]],
                                                              [[0.09], [0.0]]], rtol=1e-7)

    def test_get_V_dV_11(self):
        raise NotImplementedError("Please implement a test here while"
                                  " implmenting the function!!")

    def test_get_V_dV_22(self):
        raise NotImplementedError("Please implement a test here while"
                                  " implmenting the function!!")

    def test_get_V_dV_12(self):
        raise NotImplementedError("Please implement a test here while"
                                  " implmenting the function!!")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TullyModelTest)
    unittest.TextTestRunner().run(suite)
