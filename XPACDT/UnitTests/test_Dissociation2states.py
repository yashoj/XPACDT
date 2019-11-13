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
import unittest

import XPACDT.Interfaces.Dissociation2states as diss2S


class Dissociation2statesTest(unittest.TestCase):
    
    def setUp(self):
        self.pes_strong_1_nb = diss2S.Dissociation2states(1, **{'model_type': 'strong_coupling'})
        self.pes_strong_2_nb = diss2S.Dissociation2states(2, **{'model_type': 'strong_coupling'})
        self.pes_weak_1_nb = diss2S.Dissociation2states(1, **{'model_type': 'weak_coupling'})
        self.pes_weak_2_nb = diss2S.Dissociation2states(2, **{'model_type': 'weak_coupling'})

        return

    def test_creation(self):
        with self.assertRaises(AssertionError):
            pes = diss2S.Dissociation2states(1)

        self.assertEqual(self.pes_strong_1_nb.name, 'Dissociation2states')
        self.assertEqual(self.pes_strong_1_nb.model_type, 'strong_coupling')

        return

    def test_calculate_diabatic_all(self):
        # Test correct diabatic potential values and gradients for all states
        
        # Strong coupling model: 1 bead
        self.pes_strong_1_nb._calculate_diabatic_all(np.array([[2.0]]))
        np.testing.assert_allclose(
                self.pes_strong_1_nb._diabatic_energy, [[[0.0], [1.59483967e-21]],
                                                        [[1.59483967e-21], [0.03622797]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_strong_1_nb._diabatic_gradient, [[[[0.0]], [[1.18400898e-19]]],
                                                          [[[1.18400898e-19]], [[-0.06398139]]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_strong_1_nb._diabatic_energy_centroid, [[0.0, 1.59483967e-21],
                                                                 [1.59483967e-21, 0.03622797]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_strong_1_nb._diabatic_gradient_centroid, [[[0.0], [1.18400898e-19]],
                                                                   [[1.18400898e-19], [-0.06398139]]], rtol=1e-7)
        
        # Weak coupling model: 1 bead
        self.pes_weak_1_nb._calculate_diabatic_all(np.array([[2.0]]))
        np.testing.assert_allclose(
                self.pes_weak_1_nb._diabatic_energy, [[[0.0], [1.27096733e-08]],
                                                      [[1.27096733e-08], [0.03959738]]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_weak_1_nb._diabatic_gradient, [[[[0.0]], [[1.16928994e-07]]],
                                                        [[[1.16928994e-07]], [[-0.02771817]]]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_weak_1_nb._diabatic_energy_centroid, [[0.0, 1.27096733e-08],
                                                               [1.27096733e-08, 0.03959738]], rtol=1e-7)
        np.testing.assert_allclose(
                self.pes_weak_1_nb._diabatic_gradient_centroid, [[[0.0], [1.16928994e-07]],
                                                                 [[1.16928994e-07], [-0.02771817]]], rtol=1e-6)

        # Strong coupling model: 2 beads
        self.pes_strong_2_nb._calculate_diabatic_all(np.array([[2.0, 3.16]]))
        np.testing.assert_allclose(
                self.pes_strong_2_nb._diabatic_energy, [[[0.0, 0.00685996], [1.59483967e-21, 0.008]], 
                                                        [[1.59483967e-21, 0.008], [0.03622797, 0.006834]]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_strong_2_nb._diabatic_gradient, [[[[0.0, 0.00773923]], [[1.18400898e-19, 0.0]]], 
                                                          [[[1.18400898e-19, 0.0]], [[-0.06398139, -0.00666315]]]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_strong_2_nb._diabatic_energy_centroid, [[0.00245461, 1.69042827e-07],
                                                                 [1.69042827e-07, 0.01400544]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_strong_2_nb._diabatic_gradient_centroid, [[[0.00689168], [6.27486975e-06]],
                                                                   [[6.27486975e-06], [-0.02064746]]], rtol=1e-6)
        
        # Weak coupling model: 2 beads
        self.pes_weak_2_nb._calculate_diabatic_all(np.array([[2.0, 4.3]]))
        np.testing.assert_allclose(
                self.pes_weak_2_nb._diabatic_energy, [[[0.0, 0.00788315], [1.27096733e-08, 0.0005]],\
                                                      [[1.27096733e-08, 0.0005], [0.03959738, 0.00791503]]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_weak_2_nb._diabatic_gradient, [[[[0.0, 0.00137031]], [[1.16928994e-07, 0.0]]],
                                                        [[[1.16928994e-07, 0.0]], [[-0.02771817, -0.00554052]]]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_weak_2_nb._diabatic_energy_centroid, [[0.00493148, 3.55026769e-05],
                                                               [3.55026769e-05, 0.01770351]], rtol=1e-6)
        np.testing.assert_allclose(
                self.pes_weak_2_nb._diabatic_gradient_centroid, [[[0.00410023], [1.63312314e-04]],
                                                                 [[1.63312314e-04], [-0.01239246]]], rtol=1e-6)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Dissociation2statesTest)
    unittest.TextTestRunner().run(suite)
