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

import XPACDT.Interfaces.Morse1D as morse1d


class Morse1DTest(unittest.TestCase):

    def setUp(self):
        De = 0.02278
        a = 0.686
        re = 2.0
        b = 0.0

        self.pes_1_nb = morse1d.Morse1D(1, **{'De': De, 'a': a, 're': re, 'b': b})
        self.pes_2_nb = morse1d.Morse1D(2, **{'De': De, 'a': a, 're': re, 'b': b})

        return

    def test_creation(self):
        self.assertEqual(self.pes_1_nb.name, 'Morse1D')
        self.assertEqual(self.pes_1_nb.De, 0.02278)
        self.assertEqual(self.pes_1_nb.a, 0.686)
        self.assertEqual(self.pes_1_nb.re, 2.0)
        self.assertEqual(self.pes_1_nb.b, 0.0)

        return

    def test_calculate_adiabatic_all(self):

        # test correct potential values and gradients

        # At x = re, energy = -b and gradient = 0
        self.pes_1_nb._calculate_adiabatic_all(np.array([[2.0]]))
        self.assertSequenceEqual(self.pes_1_nb._adiabatic_energy, [[0.0]])
        self.assertSequenceEqual(self.pes_1_nb._adiabatic_gradient, [[[0.0]]])
        self.assertSequenceEqual(self.pes_1_nb._adiabatic_energy_centroid, [0.0])
        self.assertSequenceEqual(self.pes_1_nb._adiabatic_gradient_centroid, [[0.0]])

        # At large x, energy should be approx. (De - b) and zero gradient
        self.pes_1_nb._calculate_adiabatic_all(np.array([[100000.0]]))
        np.testing.assert_allclose(self.pes_1_nb._adiabatic_energy, [[0.02278]],  rtol=1e-7)
        np.testing.assert_allclose(self.pes_1_nb._adiabatic_gradient, [[[0.0]]],  rtol=1e-7)
        np.testing.assert_allclose(self.pes_1_nb._adiabatic_energy_centroid, [0.02278],  rtol=1e-7)
        np.testing.assert_allclose(self.pes_1_nb._adiabatic_gradient_centroid, [[0.0]],  rtol=1e-7)

        # test for multiple beads
        self.pes_2_nb._calculate_adiabatic_all(np.array([[1.5, 2.5]]))
        np.testing.assert_allclose(self.pes_2_nb._adiabatic_energy, [[0.00381381, 0.00192058]],  atol=1e-8)
        np.testing.assert_allclose(self.pes_2_nb._adiabatic_gradient, [[[-0.01802077, 0.00643998]]],  atol=1e-8)
        np.testing.assert_allclose(self.pes_2_nb._adiabatic_energy_centroid, [0.0],  rtol=1e-7)
        np.testing.assert_allclose(self.pes_2_nb._adiabatic_gradient_centroid, [[0.0]],  rtol=1e-7)        

    def test_minimize_geom(self):
        e_reference = 0.0
        R_reference = np.array([2.0])

        R0 = np.array([1.0])
        e, R = self.pes_1_nb.minimize_geom(R0)
        np.testing.assert_allclose(e, e_reference, atol=1e-8)
        np.testing.assert_allclose(R, R_reference, rtol=1e-6)

    def test_get_Hessian(self):
        Hessian = self.pes_1_nb.get_Hessian(np.array([2.0]))
        Hessian_reference = np.array([[2. * 0.686**2 * 0.02278]])
        np.testing.assert_allclose(Hessian, Hessian_reference)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Morse1DTest)
    unittest.TextTestRunner().run(suite)
