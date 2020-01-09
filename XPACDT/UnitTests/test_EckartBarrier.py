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

import XPACDT.Interfaces.EckartBarrier as eckart


class EckartBarrierTest(unittest.TestCase):

    def test_creation(self):
        with self.assertRaises(RuntimeError):
            eckart.EckartBarrier()

        pes_asym = eckart.EckartBarrier(**{'A': -18.0/np.pi, 'B': 54.0/np.pi, 'L': 4.0/np.sqrt(3.0*np.pi)})
        pes_asym2 = eckart.EckartBarrier(**{'d': -18.0/np.pi, 'h': 6.0/np.pi, 'w': 1.0, 'm': 1.0})
        self.assertEqual(pes_asym.name, 'EckartBarrier')
        self.assertEqual(pes_asym.A, -18.0/np.pi)
        self.assertEqual(pes_asym.B, 54.0/np.pi)
        self.assertEqual(pes_asym.L, 4.0/np.sqrt(3.0*np.pi))

        self.assertEqual(pes_asym2.name, 'EckartBarrier')
        self.assertAlmostEqual(pes_asym2.A, -18.0/np.pi)
        self.assertAlmostEqual(pes_asym2.B, 54.0/np.pi)
        self.assertAlmostEqual(pes_asym2.L, 4.0/np.sqrt(3.0*np.pi))

        pes_sym = eckart.EckartBarrier(**{'A': 0, 'B': 0.0363857, 'L': 0.330235})
        pes_sym2 = eckart.EckartBarrier(**{'d': 0, 'h': 0.0363857/4.0, 'w': 0.00476288299, 'm': 1836.0})
        self.assertEqual(pes_sym.name, 'EckartBarrier')
        self.assertEqual(pes_sym.A, 0)
        self.assertEqual(pes_sym.B, 0.0363857)
        self.assertEqual(pes_sym.L, 0.330235)

        self.assertEqual(pes_sym2.name, 'EckartBarrier')
        self.assertAlmostEqual(pes_sym2.A, 0)
        self.assertAlmostEqual(pes_sym2.B, 0.0363857)
        self.assertAlmostEqual(pes_sym2.L, 0.330235, places=3)

        with self.assertRaises(ValueError):
            eckart.EckartBarrier(**{'A': 'miep', 'B': 2.0, 'L': 1.0})

        with self.assertRaises(AssertionError):
            eckart.EckartBarrier(**{'A': '2.0', 'B': 2.0, 'L': 1.0})

        with self.assertRaises(AssertionError):
            eckart.EckartBarrier(**{'A': '-2.0', 'B': -2.0, 'L': 1.0})

        with self.assertRaises(AssertionError):
            eckart.EckartBarrier(**{'A': '-2.0', 'B': 2.0, 'L': -1.0})

        return

    def test_calculate_adiabatic_all(self):
        pes_asym = eckart.EckartBarrier(**{'A': -18.0/np.pi, 'B': 54.0/np.pi, 'L': 4.0/np.sqrt(3.0*np.pi)})

        # test the given parameters
        with self.assertRaises(AssertionError):
            pes_asym._calculate_adiabatic_all([0.0], None)

        with self.assertRaises(AssertionError):
            pes_asym._calculate_adiabatic_all(np.array([0.0]), None)

        with self.assertRaises(AssertionError):
            pes_asym._calculate_adiabatic_all(np.array([[[0.0]]]), None)

        # test correct potential values and gradients
        pes_asym._calculate_adiabatic_all(np.array([[-100.0]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, [[0.0]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, [0.0], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_asym._calculate_adiabatic_all(np.array([[+100.0]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, [[-18.0/np.pi]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, [-18.0/np.pi], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_asym._calculate_adiabatic_all(np.array([[-4.0*np.log(2.0)/np.sqrt(3.0*np.pi)]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, [[6.0/np.pi]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, [6.0/np.pi], atol=1e-10)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_asym._calculate_adiabatic_all(np.array([[-4.0/np.sqrt(3.0*np.pi)]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, [[1.83859]], atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, [[[0.334034]]], atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, [1.83859], atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, [[0.334034]], atol=1e-6)

        pes_asym._calculate_adiabatic_all(np.array([[4.0/np.sqrt(3.0*np.pi)]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, [[-0.809147]], atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, [[[-2.06321]]], atol=1e-5)
        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, [-0.809147], atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, [[-2.06321]], atol=1e-5)

        # test for multiple beads
        pes_asym._calculate_adiabatic_all(np.array([[-100.0, 100.0, -4.0/np.sqrt(3.0*np.pi), 4.0/np.sqrt(3.0*np.pi)]]), None)
        np.testing.assert_allclose(pes_asym._adiabatic_energy, np.array([[0.0, -18.0/np.pi, 1.83859, -0.809147]]), atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient, np.array([[[0.0, 0.0, 0.334034, -2.06321]]]), atol=1e-5)

        np.testing.assert_allclose(pes_asym._adiabatic_energy_centroid, np.array([4.5/np.pi]), atol=1e-6)
        np.testing.assert_allclose(pes_asym._adiabatic_gradient_centroid, np.array([[-1.09936]]), atol=1e-5)

        pes_sym = eckart.EckartBarrier(**{'A': 0, 'B': 0.0363857, 'L': 0.330235})

        with self.assertRaises(AssertionError):
            pes_sym._calculate_adiabatic_all([0.0], None)

        with self.assertRaises(AssertionError):
            pes_sym._calculate_adiabatic_all(np.array([0.0]), None)

        with self.assertRaises(AssertionError):
            pes_sym._calculate_adiabatic_all(np.array([[[0.0]]]), None)

        # test correct potential values and gradients
        pes_sym._calculate_adiabatic_all(np.array([[-70.0]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, [[0.0]], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, [0.0], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_sym._calculate_adiabatic_all(np.array([[+70.0]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, [[0.0]], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, [0.0], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_sym._calculate_adiabatic_all(np.array([[0.0]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, [[0.0363857/4.0]], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, [[[0.0]]], atol=1e-10)
        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, [0.0363857/4.0], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, [[0.0]], atol=1e-10)

        pes_sym._calculate_adiabatic_all(np.array([[-0.66047]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, [[0.0090964/np.cosh(1)**2]], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, [[[0.00881034]]], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, [0.0090964/np.cosh(1)**2], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, [[0.00881034]], atol=1e-6)

        pes_sym._calculate_adiabatic_all(np.array([[0.66047]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, [[0.0090964/np.cosh(1)**2]], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, [[[-0.00881034]]], atol=1e-5)
        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, [0.0090964/np.cosh(1)**2], atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, [[-0.00881034]], atol=1e-5)

        # test for multiple beads
        pes_sym._calculate_adiabatic_all(np.array([[-70.0, 70.0, -0.66047, 0.66047]]), None)
        np.testing.assert_allclose(pes_sym._adiabatic_energy, np.array([[0.0, 0.0, 0.0090964/np.cosh(1)**2, 0.0090964/np.cosh(1)**2]]), atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient, np.array([[[0.0, 0.0, 0.00881034, -0.00881034]]]), atol=1e-5)

        np.testing.assert_allclose(pes_sym._adiabatic_energy_centroid, np.array([0.0363857/4.0]), atol=1e-6)
        np.testing.assert_allclose(pes_sym._adiabatic_gradient_centroid, np.array([[0.0]]), atol=1e-5)

    def test_get_Hessian(self):
        pes_asym = eckart.EckartBarrier(**{'A': -18.0/np.pi, 'B': 54.0/np.pi, 'L': 4.0/np.sqrt(3.0*np.pi)})
        Hessian_reference_asym = np.array([[-1.0]])
        R = np.array([-4.0/np.sqrt(3.0*np.pi) * np.log(2)])
        Hessian = pes_asym.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference_asym)

        pes_sym = eckart.EckartBarrier(**{'A': 0, 'B': 0.0363857, 'L': 0.330235})
        Hessian_reference_sym = np.array([[-0.0363857/(8*0.330235*0.330235)]])
        R = np.array([0.0])
        Hessian = pes_sym.get_Hessian(R)
        np.testing.assert_allclose(Hessian, Hessian_reference_sym)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(EckartBarrierTest)
    unittest.TextTestRunner().run(suite)
