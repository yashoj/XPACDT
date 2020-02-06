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

import XPACDT.Interfaces.BKMP2 as bkmp2
import XPACDT.Tools.NormalModes as nm
import XPACDT.Tools.Units as units
import XPACDT.Input.Inputfile as infile


class BKMP2Test(unittest.TestCase):

    def setUp(self):
        self.pes = bkmp2.BKMP2(**infile.Inputfile("FilesForTesting/InterfaceTests/input_bkmp2.in"))

    def test_creation(self):
        self.assertEqual(self.pes.name, 'BKMP2')

    def test_calculate_adiabatic_all(self):
        # Full Asymptote
        energy_ref = np.zeros((1, 1))
        gradient_ref = np.zeros((1, 9, 1))
        self.pes._calculate_adiabatic_all(self.pes._from_internal([400.0, 800.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes._adiabatic_gradient, gradient_ref, atol=1e-10)

        # H2 minimum
        energy_ref = np.zeros((1, 1)) + -0.17449577
        gradient_ref = np.zeros((1, 9, 1))
        self.pes._calculate_adiabatic_all(self.pes._from_internal([1.4014718, 80.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes._adiabatic_energy, energy_ref)
        np.testing.assert_allclose(self.pes._adiabatic_gradient, gradient_ref, atol=1e-6)

        # TST
        energy_ref = np.zeros((1, 1)) + -0.17449577 + 0.01532
        gradient_ref = np.zeros((1, 9, 1))
        self.pes._calculate_adiabatic_all(self.pes._from_internal([1.757, 2.6355, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes._adiabatic_gradient, gradient_ref, atol=1e-10)

        # RPMD
        x0 = self.pes._from_internal([1.757, 2.6355, 0.0])
        x1 = self.pes._from_internal([400.0, 800.0, 0.0])
        x2 = self.pes._from_internal([1.4014718, 80.0, 0.0])
        x3 = self.pes._from_internal([1.757, 2.6355, 0.0])
        x = np.column_stack((x0, x1, x2, x3))
        self.pes._calculate_adiabatic_all(x, None)
        energy_ref = np.array([[-0.15917577, 0.0, -0.17449577, -0.15917577]])
        gradient_ref = np.zeros((1, 9, 4))
        np.testing.assert_allclose(self.pes._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes._adiabatic_gradient, gradient_ref, atol=1e-6)

    def test_minimize(self):
        fun_ref = -0.17449577
        x_ref = np.array([0.70073594, 0.0, 0.0, -0.70073594, 0.0, 0.0, 40.0, 0.0, 0.0])
        fun, x = self.pes.minimize_geom(self.pes._from_internal([2.0, 40.0, 0.0]))
        self.assertAlmostEqual(fun_ref, fun)
        np.testing.assert_allclose(x, x_ref)

    def test_get_Hessian(self):
        freq_ref = np.zeros(9)
        freq_ref[8] = 4403
        r = self.pes._from_internal([1.4014718, 80.0, 0.0])
        hessian = self.pes.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('H')]*9)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=3.0)

        freq_ref = np.zeros(9)
        freq_ref[0] = -1510
        freq_ref[6] = 907
        freq_ref[7] = 907
        freq_ref[8] = 2055
        r = self.pes._from_internal([1.757, 2.6355, 0.0])
        hessian = self.pes.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('H')]*9)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=3.0)

    def test_from_internal(self):
        # colinear
        internal = np.array([2.0, 4.0, 0.0])
        cartesian_ref = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        cartesian = self.pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, self.pes._from_cartesian_to_internal(cartesian))

        # perpendicular
        internal = np.array([2.0, 4.0, np.pi/2.0])
        cartesian_ref = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0])
        cartesian = self.pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref, atol=1e-10)
        np.testing.assert_allclose(internal, self.pes._from_cartesian_to_internal(cartesian))

        # 45 degrees off
        internal = np.array([2.0, 4.0, np.pi/4.0])
        cartesian_ref = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0/np.sqrt(2.0), 4.0/np.sqrt(2.0), 0.0])
        cartesian = self.pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, self.pes._from_cartesian_to_internal(cartesian))

        # -45 degrees off
        internal = np.array([2.0, 4.0, 2.0*np.pi-np.pi/4.0])
        cartesian_ref = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0/np.sqrt(2.0), -4.0/np.sqrt(2.0), 0.0])
        cartesian = self.pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, self.pes._from_cartesian_to_internal(cartesian))

    def test_from_cartesian_to_internal(self):
        # colinear
        cartesian = np.array([-1.2, 0.0, 0.0, 1.2, 0.0, 0.0, 3.8, 0.0, 0.0])
        internal_ref = np.array([2.4, 3.8, 0.0])
        internal = self.pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)
        np.testing.assert_allclose(cartesian, self.pes._from_internal(internal))

        # perpendicular
        cartesian = np.array([0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0])
        internal_ref = np.array([2.0, 4.0, np.pi/2.0])
        internal = self.pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)

        # 'random' in space, 3rd H along axis
        cartesian = np.array([1.0, -1.0, 2.0, 0.5, -1.5, 2.0-1.0/np.sqrt(2.0), -1.25 , -3.25, -1.18198052])
        internal_ref = np.array([1.0, 4.0, 2.0*np.pi])
        internal = self.pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)

        # -45 degrees off
        cartesian = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0/np.sqrt(2.0), -4.0/np.sqrt(2.0), 0.0])
        internal_ref = np.array([2.0, 4.0, 2.0*np.pi-np.pi/4.0])
        internal = self.pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)
        np.testing.assert_allclose(cartesian, self.pes._from_internal(internal))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(BKMP2Test)
    unittest.TextTestRunner().run(suite)
