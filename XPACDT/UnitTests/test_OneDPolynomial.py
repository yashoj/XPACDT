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
import os
import unittest

import XPACDT.Interfaces.OneDPolynomial as oneDP
import XPACDT.Input.Inputfile as infile

from XPACDT.Input.Error import XPACDTInputError


class OneDPolynomialTest(unittest.TestCase):

    def setUp(self):

        self.pes1D_harmonic_classical = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/harmonic.in"))
        self.pes1D_shifted_harmonic_classical = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/harmonic_shifted.in"))
        self.pes1D_shifted_anharmonic_classical = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/anharmonic_shifted.in"))
        self.pes1D_anharmonic_classical = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/anharmonic.in"))
        self.pes1D_quartic_classical = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/quartic.in"))


        self.pes1D_harmonic_4_nb = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/harmonic_4.in"))
        self.pes1D_shifted_anharmonic_4_nb = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/anharmonic_shifted_4.in"))
        self.pes1D_anharmonic_4_nb = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/anharmonic_4.in"))
        self.pes1D_quartic_4_nb = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/quartic_4.in"))

        return

    def test_creation(self):
        self.assertEqual(self.pes1D_harmonic_classical.name, 'OneDPolynomial')
        self.assertEqual(self.pes1D_harmonic_classical.x0, 0.0)
        self.assertSequenceEqual(self.pes1D_harmonic_classical.a, [0.0, 0.0, 0.5])

        self.assertEqual(self.pes1D_shifted_anharmonic_classical.name, 'OneDPolynomial')
        self.assertEqual(self.pes1D_shifted_anharmonic_classical.x0, -1.0)
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical.a, [1.0, 0.0, 0.5, 0.1, 0.01])

        with self.assertRaises(XPACDTInputError):
            pes = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/harmonic_fail1.in"))

        with self.assertRaises(XPACDTInputError):
            pes = oneDP.OneDPolynomial(**infile.Inputfile("FilesForTesting/InterfaceTests/harmonic_fail1.in"))

        return

    def test_calculate_adiabatic_all(self):
        # test correct potential values and gradients
        self.pes1D_harmonic_classical._calculate_adiabatic_all(np.array([[0.0]]), None)
        self.assertSequenceEqual(self.pes1D_harmonic_classical._adiabatic_energy, [[0.0]])
        self.assertSequenceEqual(self.pes1D_harmonic_classical._adiabatic_gradient, [[[0.0]]])
        self.assertSequenceEqual(self.pes1D_harmonic_classical._adiabatic_energy_centroid, [0.0])
        self.assertSequenceEqual(self.pes1D_harmonic_classical._adiabatic_gradient_centroid, [[0.0]])

        self.pes1D_shifted_anharmonic_classical._calculate_adiabatic_all(np.array([[-1.0]]), None)
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_energy, [[1.0]])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_gradient, [[[0.0]]])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_energy_centroid, [1.0])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_gradient_centroid, [[0.0]])

        self.pes1D_shifted_anharmonic_classical._calculate_adiabatic_all(np.array([[1.0]]), None)
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_energy, [[1.0+2.0+0.8+0.16]])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_gradient, [[[2.0+1.2+0.32]]])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_energy_centroid, [1.0+2.0+0.8+0.16])
        self.assertSequenceEqual(self.pes1D_shifted_anharmonic_classical._adiabatic_gradient_centroid, [[2.0+1.2+0.32]])

        # test for multiple beads
        self.pes1D_shifted_anharmonic_4_nb._calculate_adiabatic_all(np.array([[1.0, -2.0, -1.0, 0.0]]), None)
        self.assertTrue(
                np.alltrue(self.pes1D_shifted_anharmonic_4_nb._adiabatic_energy
                           == np.array([[1.0+2.0+0.8+0.16, 1.0+0.5-0.1+0.01, 1.0, 1.0+0.5+0.1+0.01]])))
        self.assertTrue(
                np.alltrue(self.pes1D_shifted_anharmonic_4_nb._adiabatic_gradient
                           == np.array([[[2.0+1.2+0.32, -1.0+0.3-0.04, 0.0, 1+0.1*3.+0.01*4.]]])))

        np.testing.assert_allclose(self.pes1D_shifted_anharmonic_4_nb._adiabatic_energy_centroid, np.array([1.138125]))
        np.testing.assert_allclose(self.pes1D_shifted_anharmonic_4_nb._adiabatic_gradient_centroid, np.array([[0.58]]))

    def test_optimize_geometry(self):
        # Harmonic oscillator
        e_reference = 0.0
        R_reference = np.array([0.0])

        R0 = np.array([1.0])
        e, R = self.pes1D_harmonic_classical.optimize_geometry(R0)
        np.testing.assert_allclose(e, e_reference)
        np.testing.assert_allclose(R, R_reference)

        # Shifted harmonic oscillator
        e_reference = 1.0
        R_reference = np.array([2.0])

        R0 = np.array([-10.0])
        e, R = self.pes1D_shifted_harmonic_classical.optimize_geometry(R0)
        np.testing.assert_allclose(e, e_reference)
        np.testing.assert_allclose(R, R_reference)

    def test_plot1d(self):
        self.pes1D_shifted_harmonic_classical.plot_1D(np.array([0.0]), 0, -10.0, 10.0, 2.0)

        points_reference = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])
        values_reference = np.array([73., 51., 33., 19., 9., 3., 1., 3., 9., 19., 33.])

        points_values = np.loadtxt('pes_1d.dat')
        np.testing.assert_allclose(points_values[:, 0], points_reference)
        np.testing.assert_allclose(points_values[:, 1], values_reference)
        os.remove('pes_1d.dat')
        os.remove("pes_1d.plt")

    def test_get_Hessian(self):
        Hessian_reference = np.array([[1.0]])
        R = np.array([0.0])

        Hessian = self.pes1D_harmonic_classical.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)

        Hessian_reference = np.array([[1.0+0.6+0.12]])
        R = np.array([1.0])

        Hessian = self.pes1D_anharmonic_classical.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)

        Hessian_reference = np.array([[1.0-0.6+0.12]])
        R = np.array([-1.0])

        Hessian = self.pes1D_anharmonic_classical.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(OneDPolynomialTest)
    unittest.TextTestRunner().run(suite)
