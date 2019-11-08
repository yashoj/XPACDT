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
import math
import unittest

import XPACDT.Interfaces.InterfaceTemplate as IT
import XPACDT.Interfaces.TullyModel as tullym
import XPACDT.Interfaces.MorseDiabatic as morsedia


class InterfaceTemplateTest(unittest.TestCase):

    def setUp(self):
        # todo create input file here.
        self.interface = IT.PotentialInterface("dummyTemplate", 4, n_states=1,
                                               max_n_beads=3)
        return

    def test_changed(self):
        self.assertEqual(self.interface.name, "dummyTemplate")

        # test the given parameters
        with self.assertRaises(AssertionError):
            self.interface._changed([0.0], None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([0.0]), None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[[0.0]]]), None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), [0.0])

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), np.array([0.0]))

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), np.array([[[0.0]]]))

        # initial storage
        R = np.random.rand(12).reshape(4, 3)
        self.assertTrue(self.interface._changed(R, None))
        self.assertTrue(np.alltrue(self.interface._old_R == R))

        # not changed
        self.assertFalse(self.interface._changed(R, None))

        # changed
        R = np.random.rand(12).reshape(4, 3)
        self.assertTrue(self.interface._changed(R, None))
        self.assertTrue(np.alltrue(self.interface._old_R == R))

        return

    def test_recalculate_adiabatic(self):
        # test for 2 state potential
        pes = tullym.TullyModel(1, **{'model_type': 'model_C'})

        ### Testing if trying to access adiabatic energy runs
        # '_recalculate_adiabatic' function which in turn calculates all the
        # adiabatic properties.
        R = np.array([[0.]])
        V = pes.diabatic_energy(R, SI=None, SJ=None, centroid=False,
                                return_matrix=True)

        # Checking if all diabatic properties have been calculated
        np.testing.assert_allclose(pes._diabatic_energy, [[[0.0006], [0.1]],
                                                          [[0.1], [-0.0006]]], rtol=1e-7)
        np.testing.assert_allclose(pes._diabatic_gradient, [[[[0.0]], [[0.09]]],
                                                            [[[0.09]], [[0.0]]]], rtol=1e-7)
        # Now accessing adiabatic energy to calculate all adiabatic properties
        V_ad = pes.adiabatic_energy(R, 0, centroid=False, return_matrix=True)

        np.testing.assert_allclose(V_ad, [[-math.sqrt(0.01 + 3.6e-07)],
                                          [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy, [[-math.sqrt(0.01 + 3.6e-07)],
                                        [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient, [[[-0.009 / math.sqrt(0.01 + 3.6e-07)]],
                                          [[0.009 / math.sqrt(0.01 + 3.6e-07)]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac, [[[[0.0]], [[-2.7e-05/(0.01 + 3.6e-07)]]],
                           [[[2.7e-05/(0.01 + 3.6e-07)]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy_centroid, [-math.sqrt(0.01 + 3.6e-07),
                                                 math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient_centroid, [[-0.009 / math.sqrt(0.01 + 3.6e-07)],
                                                   [0.009 / math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac_centroid, [[[0.0], [-2.7e-05/(0.01 + 3.6e-07)]],
                                    [[2.7e-05/(0.01 + 3.6e-07)], [0.0]]], rtol=1e-7)

        ### Testing again with different position whether the change is computed.
        R = np.array([[-1.0e5]])
        # This doesn't change the 'old_R' value yet.
        pes._calculate_diabatic_all(R)

        # Checking if at this point the adiabatic properties haven't been changed yet.
        np.testing.assert_allclose(pes._adiabatic_energy,
                                   [[-math.sqrt(0.01 + 3.6e-07)],
                                    [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)

        # Now checking if the adiabatic properties have been changed due to
        # trying to access the adiabatic energy
        V_ad = pes.adiabatic_energy(R, 0, centroid=False, return_matrix=True)

        np.testing.assert_allclose(V_ad, [[-0.0006], [0.0006]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy, [[-0.0006], [0.0006]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient, [[[0.0]], [[0.0]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac, [[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy_centroid, [-0.0006, 0.0006], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient_centroid, [[0.0], [0.0]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac_centroid, [[[0.0], [0.0]], [[0.0], [0.0]]], rtol=1e-7)

        ### Testing if changing position again and accessing diabatic
        # energy, which changes 'old_R', also leads to change in adiabatic
        # properties if they are accessed
        R = np.array([[0.]])
        # This changes the 'old_R' value
        V = pes.diabatic_energy(R, SI=None, SJ=None, centroid=False, return_matrix=True)

        # Checking if at this point the adiabatic properties haven't been
        # changed but the diabatic have.
        np.testing.assert_allclose(pes._diabatic_energy, [[[0.0006], [0.1]],
                                                          [[0.1], [-0.0006]]], rtol=1e-7)
        np.testing.assert_allclose(pes._diabatic_gradient, [[[[0.0]], [[0.09]]],
                                                            [[[0.09]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(pes._adiabatic_energy, [[-0.0006], [0.0006]], rtol=1e-7)

        # Now checking if the adiabatic properties have been changed due to
        # trying to access the adiabatic energy
        V_ad = pes.adiabatic_energy(R, 0, centroid=False, return_matrix=True)
        
        np.testing.assert_allclose(V_ad, [[-math.sqrt(0.01 + 3.6e-07)],
                                          [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy, [[-math.sqrt(0.01 + 3.6e-07)],
                                        [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient, [[[-0.009 / math.sqrt(0.01 + 3.6e-07)]],
                                          [[0.009 / math.sqrt(0.01 + 3.6e-07)]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac, [[[[0.0]], [[-2.7e-05/(0.01 + 3.6e-07)]]],
                           [[[2.7e-05/(0.01 + 3.6e-07)]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy_centroid, [-math.sqrt(0.01 + 3.6e-07),
                                                 math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient_centroid, [[-0.009 / math.sqrt(0.01 + 3.6e-07)],
                                                   [0.009 / math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac_centroid, [[[0.0], [-2.7e-05/(0.01 + 3.6e-07)]],
                                    [[2.7e-05/(0.01 + 3.6e-07)], [0.0]]], rtol=1e-7)

        return

    def test_get_adiabatic_from_diabatic(self):
        # test for 2 state potential
        pes = tullym.TullyModel(1, **{'model_type': 'model_C'})
        R = np.array([[0.]])
        pes._calculate_diabatic_all(R)
        pes._get_adiabatic_from_diabatic(R)

        np.testing.assert_allclose(
                pes._adiabatic_energy, [[-math.sqrt(0.01 + 3.6e-07)],
                                        [math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient, [[[-0.009 / math.sqrt(0.01 + 3.6e-07)]],
                                          [[0.009 / math.sqrt(0.01 + 3.6e-07)]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac, [[[[0.0]], [[-2.7e-05/(0.01 + 3.6e-07)]]],
                           [[[2.7e-05/(0.01 + 3.6e-07)]], [[0.0]]]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_energy_centroid, [-math.sqrt(0.01 + 3.6e-07),
                                                 math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(
                pes._adiabatic_gradient_centroid, [[-0.009 / math.sqrt(0.01 + 3.6e-07)],
                                                   [0.009 / math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(
                pes._nac_centroid, [[[0.0], [-2.7e-05/(0.01 + 3.6e-07)]],
                                    [[2.7e-05/(0.01 + 3.6e-07)], [0.0]]], rtol=1e-7)

        # test for 3 state potential
        pes = morsedia.MorseDiabatic(1, **{'n_states': '3', 'model_type': 'model_1'})
        R = np.array([[0.]])
        pes._calculate_diabatic_all(R)

        with self.assertRaises(AssertionError):
            pes._get_adiabatic_from_diabatic(R)

        # TODO: add more tests with more beads and 3 state test using morse diabatic

        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(InterfaceTemplateTest)
    unittest.TextTestRunner().run(suite)
