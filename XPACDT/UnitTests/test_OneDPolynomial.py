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
import os
import unittest

import XPACDT.Interfaces.OneDPolynomial as oneDP
import XPACDT.Input.Inputfile as infile


class OneDPolynomialTest(unittest.TestCase):

    def test_creation(self):
        with self.assertRaises(AssertionError):
            pes = oneDP.OneDPolynomial(1)

        pes = oneDP.OneDPolynomial(1, **{'a': '0.0 0.0 0.5'})
        self.assertEqual(pes.name, 'OneDPolynomial')
        self.assertEqual(pes.x0, 0.0)
        self.assertSequenceEqual(pes.a, [0.0, 0.0, 0.5])

        pes = oneDP.OneDPolynomial(1, **{'a': '1.0 0.0 0.5 0.1', 'x0': '-1.0'})
        self.assertEqual(pes.name, 'OneDPolynomial')
        self.assertEqual(pes.x0, -1.0)
        self.assertSequenceEqual(pes.a, [1.0, 0.0, 0.5, 0.1])

        with self.assertRaises(ValueError):
            pes = oneDP.OneDPolynomial(1, **{'a': 'miep'})

        with self.assertRaises(ValueError):
            pes = oneDP.OneDPolynomial(1, **{'x0': 'miep', 'a': '0.0'})

        return

    def test_calculate_adiabatic_all(self):
        pes = oneDP.OneDPolynomial(1, **{'a': '0.0 0.0 0.5'})

        # test correct potential values and gradients
        pes._calculate_adiabatic_all(np.array([[0.0]]), None)
        self.assertSequenceEqual(pes._adiabatic_energy, [[0.0]])
        self.assertSequenceEqual(pes._adiabatic_gradient, [[[0.0]]])
        self.assertSequenceEqual(pes._adiabatic_energy_centroid, [0.0])
        self.assertSequenceEqual(pes._adiabatic_gradient_centroid, [[0.0]])

        pes = oneDP.OneDPolynomial(1, **{'a': '1.0 0.0 0.5 0.1', 'x0': '-1.0'})
        pes._calculate_adiabatic_all(np.array([[-1.0]]), None)
        self.assertSequenceEqual(pes._adiabatic_energy, [[1.0]])
        self.assertSequenceEqual(pes._adiabatic_gradient, [[[0.0]]])
        self.assertSequenceEqual(pes._adiabatic_energy_centroid, [1.0])
        self.assertSequenceEqual(pes._adiabatic_gradient_centroid, [[0.0]])

        pes._calculate_adiabatic_all(np.array([[1.0]]), None)
        self.assertSequenceEqual(pes._adiabatic_energy, [[1.0+2.0+0.8]])
        self.assertSequenceEqual(pes._adiabatic_gradient, [[[2.0+1.2]]])
        self.assertSequenceEqual(pes._adiabatic_energy_centroid, [1.0+2.0+0.8])
        self.assertSequenceEqual(pes._adiabatic_gradient_centroid, [[2.0+1.2]])

        # test for multiple beads
        pes_3_nb = oneDP.OneDPolynomial(3, **{'a': '1.0 0.0 0.5 0.1', 'x0': '-1.0'})

        pes_3_nb._calculate_adiabatic_all(np.array([[1.0, -2.0, -1.0]]), None)
        self.assertTrue(
                np.alltrue(pes_3_nb._adiabatic_energy
                           == np.array([[1.0+2.0+0.8, 1.0+0.5-0.1, 1.0]])))
        self.assertTrue(
                np.alltrue(pes_3_nb._adiabatic_gradient
                           == np.array([[[2.0+1.2, -1.0+0.3, 0.0]]])))

        np.testing.assert_allclose(pes_3_nb._adiabatic_energy_centroid, np.array([0.05555555555+0.0037037+1.0]))
        np.testing.assert_allclose(pes_3_nb._adiabatic_gradient_centroid, np.array([[0.33333333+0.0333333333]]))

    def test_minimize_geom(self):
        # Harmonic oscillator
        pes = oneDP.OneDPolynomial(1, **{'a': '0.0 0.0 0.5'})
        e_reference = 0.0
        R_reference = np.array([0.0])

        R0 = np.array([1.0])
        e, R = pes.minimize_geom(R0)
        np.testing.assert_allclose(e, e_reference)
        np.testing.assert_allclose(R, R_reference)

        # Shifted harmonic oscillator
        pes = oneDP.OneDPolynomial(1, **{'a': '1.0 0.0 0.5', 'x0': '2.0'})
        e_reference = 1.0
        R_reference = np.array([2.0])

        R0 = np.array([-10.0])
        e, R = pes.minimize_geom(R0)
        np.testing.assert_allclose(e, e_reference)
        np.testing.assert_allclose(R, R_reference)

    def test_plot1d(self):
        pes = oneDP.OneDPolynomial(1, **{'a': '1.0 0.0 0.5', 'x0': '2.0'})
        pes.plot_1D(np.array([0.0]), 0, -10.0, 10.0, 2.0)

        points_reference = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])
        values_reference = np.array([73., 51., 33., 19., 9., 3., 1., 3., 9., 19., 33.])

        points_values = np.loadtxt('pes_1d.dat')
        np.testing.assert_allclose(points_values[:, 0], points_reference)
        np.testing.assert_allclose(points_values[:, 1], values_reference)
        os.remove('pes_1d.dat')
        os.remove("pes_1d.plt")

    def test_get_Hessian(self):
        pes = oneDP.OneDPolynomial(1, **{'a': '0.0 0.0 0.5'})
        Hessian_reference = np.array([[1.0]])
        R = np.array([0.0])

        Hessian = pes.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)

        pes = oneDP.OneDPolynomial(1, **{'a': '0.0 0.0 0.5 0.1 0.01'})
        Hessian_reference = np.array([[1.0+0.6+0.12]])
        R = np.array([1.0])

        Hessian = pes.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)

        Hessian_reference = np.array([[1.0-0.6+0.12]])
        R = np.array([-1.0])

        Hessian = pes.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(OneDPolynomialTest)
    unittest.TextTestRunner().run(suite)
