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
import warnings

import XPACDT.Interfaces.Triatomic as triatomic
import XPACDT.Tools.NormalModes as nm
import XPACDT.Tools.Units as units
import XPACDT.Input.Inputfile as infile


def setUpModule():
    try:
        triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_bkmp2.in"))
    except ModuleNotFoundError as e:
        print("BKMP2 PES could not be imported. Is it compiled properly? Will skip all related tests.")
        
    try:            
        triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_cw.in"))
    except ModuleNotFoundError as e:
        print("CW PES could not be imported. Is it compiled properly? Will skip all related tests.")
        
    try:
        triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_lwal.in"))
    except ModuleNotFoundError as e:
        print("LWAL PES could not be imported. Is it compiled properly? Will skip all related tests.")
        
    return


class TriatomicTest(unittest.TestCase):

    def setUp(self):
        try:
            self.pes_bkmp2 = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_bkmp2.in"))
            self.pes_bkmp2_4 = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_bkmp2_4.in"))
        except ModuleNotFoundError as e:
            self.pes_bkmp2 = None
            self.pes_bkmp2_4 = None
            
        try:            
            self.pes_cw = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_cw.in"))
            self.pes_cw_4 = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_cw_4.in"))
        except ModuleNotFoundError as e:
            self.pes_cw = None
            self.pes_cw_4 = None

        try:
            self.pes_lwal = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_lwal.in"))
            self.pes_lwal_4 = triatomic.Triatomic(infile.Inputfile("FilesForTesting/InterfaceTests/input_lwal_4.in"))
        except ModuleNotFoundError as e:
            self.pes_lwal = None
            self.pes_lwal_4 = None

    def test_creation(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.assertEqual(self.pes_bkmp2.name, 'Triatomic')

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.assertEqual(self.pes_cw.name, 'Triatomic')

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.assertEqual(self.pes_lwal.name, 'Triatomic')

    def test_calculate_adiabatic_all(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.__bkmp2_calculate_adiabatic_all()

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.__cw_calculate_adiabatic_all()

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.__lwal_calculate_adiabatic_all()

    def __bkmp2_calculate_adiabatic_all(self):
        # BKMP2
        # Full Asymptote
        energy_ref = np.zeros((1, 1))
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_bkmp2._calculate_adiabatic_all(self.pes_bkmp2._from_internal([400.0, 800.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_gradient, gradient_ref, atol=1e-10)

        # H2 minimum 
        energy_ref = np.zeros((1, 1)) + -0.17449577
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_bkmp2._calculate_adiabatic_all(self.pes_bkmp2._from_internal([1.4014718, 80.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_energy, energy_ref)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_gradient, gradient_ref, atol=1e-6)

        # TST
        energy_ref = np.zeros((1, 1)) + -0.17449577 + 0.01532
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_bkmp2._calculate_adiabatic_all(self.pes_bkmp2._from_internal([1.757, 2.6355, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes_bkmp2._adiabatic_gradient, gradient_ref, atol=1e-10)

        # RPMD
        x0 = self.pes_bkmp2_4._from_internal([1.757, 2.6355, 0.0])
        x1 = self.pes_bkmp2_4._from_internal([400.0, 800.0, 0.0])
        x2 = self.pes_bkmp2_4._from_internal([1.4014718, 80.0, 0.0])
        x3 = self.pes_bkmp2_4._from_internal([1.757, 2.6355, 0.0])
        x = np.column_stack((x0, x1, x2, x3))
        self.pes_bkmp2_4._calculate_adiabatic_all(x, None)
        energy_ref = np.array([[-0.15917577, 0.0, -0.17449577, -0.15917577]])
        gradient_ref = np.zeros((1, 9, 4))
        np.testing.assert_allclose(self.pes_bkmp2_4._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_bkmp2_4._adiabatic_gradient, gradient_ref, atol=1e-6)

    def __lwal_calculate_adiabatic_all(self):
        # LWAL
        # H2 minimum
        energy_ref = np.zeros((1, 1)) 
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_lwal._calculate_adiabatic_all(self.pes_lwal._from_internal([1.401168, 40.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_gradient, gradient_ref, atol=1e-5)

        # HF minimum
        energy_ref = np.zeros((1, 1)) - 0.050454
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_lwal._calculate_adiabatic_all(self.pes_lwal._from_internal([80.0, 40.0+1.7335, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_gradient, gradient_ref, atol=1e-4)

        # co-linear TST
        energy_ref = np.zeros((1, 1)) + 0.003428
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_lwal._calculate_adiabatic_all(self.pes_lwal._from_internal([1.443, 2.936+0.5*1.443, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_energy, energy_ref, atol=1e-5)
        save_adiabatic_gradient = self.pes_lwal._adiabatic_gradient_centroid

        # TST
        energy_ref = np.zeros((1, 1)) + 0.002616
        gradient_ref = np.zeros((1, 9, 1))
        R = np.sqrt( (1.457/2.0)**2 + 2.932**2 - 1.457*2.932*np.cos(np.pi*113.0/180.0))
        gamma = np.arccos((2.932**2 - R**2 - (1.457/2.0)**2) / (-R * 1.457))
        self.pes_lwal._calculate_adiabatic_all(self.pes_lwal._from_internal([1.457, R, gamma]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes_lwal._adiabatic_gradient, gradient_ref, atol=1e-4)

        # RPMD
        x0 = self.pes_lwal_4._from_internal([1.401168, 40.0, 0.0])
        x1 = self.pes_lwal_4._from_internal([80.0, 40.0+1.7335, 0.0])
        x2 = self.pes_lwal_4._from_internal([1.443, 2.936+0.5*1.443, 0.0])
        x3 = self.pes_lwal_4._from_internal([1.457, R, gamma])
        x = np.column_stack((x0, x1, x2, x3))
        self.pes_lwal_4._calculate_adiabatic_all(x, None)
        energy_ref = np.array([[0.0, -0.050454, 0.003428, 0.002616]])
        gradient_ref = np.zeros((1, 9, 4))
        gradient_ref[0, :, 2] = save_adiabatic_gradient[0]
        np.testing.assert_allclose(self.pes_lwal_4._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_lwal_4._adiabatic_gradient, gradient_ref, atol=1e-4)

    def __cw_calculate_adiabatic_all(self):
        # CW
        # H2 minimum 
        energy_ref = np.zeros((1, 1)) - 0.00139  # lowered by SO
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_cw._calculate_adiabatic_all(self.pes_cw._from_internal([1.4005706, 40.0, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_cw._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes_cw._adiabatic_gradient, gradient_ref, atol=1e-4)

        # HCl minimum 
        energy_ref = np.zeros((1, 1)) + 0.004114
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_cw._calculate_adiabatic_all(self.pes_cw._from_internal([120.0, 60.0+2.41003, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_cw._adiabatic_energy, energy_ref, atol=1e-6)
        np.testing.assert_allclose(self.pes_cw._adiabatic_gradient, gradient_ref, atol=1e-4)

        # TST
        energy_ref = np.zeros((1, 1)) + 0.012107
        gradient_ref = np.zeros((1, 9, 1))
        self.pes_cw._calculate_adiabatic_all(self.pes_cw._from_internal([1.854, 3.631, 0.0]).reshape(-1, 1), None)
        np.testing.assert_allclose(self.pes_cw._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_cw._adiabatic_gradient, gradient_ref, atol=1e-4)

        # RPMD
        x0 = self.pes_cw_4._from_internal([1.854, 3.631, 0.0])
        x1 = self.pes_cw_4._from_internal([1.4005706, 40.0, 0.0])
        x2 = self.pes_cw_4._from_internal([120.0, 60.0+2.41003, 0.0])
        x3 = self.pes_cw_4._from_internal([1.854, 3.631, 0.0])
        x = np.column_stack((x0, x1, x2, x3))
        self.pes_cw_4._calculate_adiabatic_all(x, None)
        energy_ref = np.array([[0.012107, -0.00139, 0.004114, 0.012107]])
        gradient_ref = np.zeros((1, 9, 4))
        np.testing.assert_allclose(self.pes_cw_4._adiabatic_energy, energy_ref, atol=1e-5)
        np.testing.assert_allclose(self.pes_cw_4._adiabatic_gradient, gradient_ref, atol=1e-4)
        
    def test_minimize(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.__bkmp2_minimize()

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.__cw_minimize()

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.__lwal_minimize()


    def __bkmp2_minimize(self):
        # BKMP2
        fun_ref = -0.17449577
        x_ref = np.array([40.0, 0.0, 0.0, 0.70073594, 0.0, 0.0, -0.70073594, 0.0, 0.0])
        fun, x = self.pes_bkmp2.minimize_geom(self.pes_bkmp2._from_internal([2.0, 40.0, 0.0]))
        self.assertAlmostEqual(fun_ref, fun)
        np.testing.assert_allclose(x, x_ref)

    def __lwal_minimize(self):
        # LWAL
        fun_ref = 0.0
        x_ref = np.array([100.0, 0.0, 0.0, -0.700584, 0.0, 0.0, 0.700584, 0.0, 0.0])
        fun, x = self.pes_lwal.minimize_geom(self.pes_lwal._from_internal([1.4, 100.0, 0.0]))
        np.testing.assert_allclose(fun_ref, fun, atol=1e-5)
        np.testing.assert_allclose(x, x_ref, atol=1e-5)

    def __cw_minimize(self):
        # CW
        # H2
        fun_ref = - 0.00139 # lowered by SO
        x_ref = np.array([40.0, 0.0, 0.0, -0.700279, 0.0, 0.0, 0.700279, 0.0, 0.0])
        fun, x = self.pes_cw.minimize_geom(self.pes_cw._from_internal([1.4, 40.0, 0.0]))
        np.testing.assert_allclose(fun_ref, fun, atol=1e-6)
        np.testing.assert_allclose(x, x_ref, atol=1e-3)

        # HCl
        fun_ref = 0.004114
        x_ref = np.array([62.41003, 0.0, 0.0, -60.0, 0.0, 0.0, 60.0, 0.0, 0.0])
        fun, x = self.pes_cw.minimize_geom(self.pes_cw._from_internal([120.0, 60.0+2.410, 0.0]))
        np.testing.assert_allclose(fun_ref, fun, atol=1e-6)
        np.testing.assert_allclose(x, x_ref, atol=1e-3)
        
    def test_get_Hessian(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.__bkmp2_get_Hessian()

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.__cw_get_Hessian()

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.__lwal_get_Hessian()

    def __bkmp2_get_Hessian(self):
        # BKMP2
        freq_ref = np.zeros(9)
        freq_ref[8] = 4403
        r = self.pes_bkmp2._from_internal([1.4014718, 80.0, 0.0])
        hessian = self.pes_bkmp2.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('H')]*9)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=4.0)

        freq_ref = np.zeros(9)
        freq_ref[0] = -1510
        freq_ref[6] = 907
        freq_ref[7] = 907
        freq_ref[8] = 2055
        r = self.pes_bkmp2._from_internal([1.757, 2.6355, 0.0])
        hessian = self.pes_bkmp2.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('H')]*9)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=4.0)

    def __lwal_get_Hessian(self):
        # LWAL 
        freq_ref = np.zeros(9)
        freq_ref[8] = 4406
        r = self.pes_lwal._from_internal([1.401168, 300.0, 0.0])
        hessian = self.pes_lwal.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('F')]*3 + [units.atom_mass('H')]*6)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=5.0)

        # TODO: Same as from instanton code - but different from paper; why?
        freq_ref = np.zeros(9)
        freq_ref[0] = -715.8
        freq_ref[7] = 326.4
        freq_ref[8] = 3859.1
        R = np.sqrt( (1.457/2.0)**2 + 2.932**2 - 1.457*2.932*np.cos(np.pi*113.0/180.0))
        gamma = np.arccos((2.932**2 - R**2 - (1.457/2.0)**2) / (-R * 1.457))
        r = self.pes_lwal._from_internal([1.457, R, gamma])
        hessian = self.pes_lwal.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('F')]*3 + [units.atom_mass('H')]*6)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq[[0, 7, 8]], freq_ref[[0, 7, 8]], atol=5.0)

    def __cw_get_Hessian(self):
        # CW
        # HCl
        freq_ref = np.zeros(9)
        freq_ref[8] = 2989
        r = self.pes_cw._from_internal([120.0, 60.0+2.41003, 0.0])
        hessian = self.pes_cw.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('Cl')]*3 + [units.atom_mass('H')]*6)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=10.0)
        
        freq_ref = np.zeros(9)
        freq_ref[8] = 4403
        r = self.pes_cw._from_internal([1.4005706, 21.0, 0.0])
        hessian = self.pes_cw.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('Cl')]*3 + [units.atom_mass('H')]*6)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=10.0)

        freq_ref = np.zeros(9)
        freq_ref[0] = -1296
        freq_ref[6] = 540
        freq_ref[7] = 540
        freq_ref[8] = 1360
        r = self.pes_cw._from_internal([1.854, 3.631, 0.0])
        hessian = self.pes_cw.get_Hessian(r)
        freq = nm.get_normal_modes(hessian, [units.atom_mass('Cl')]*3 + [units.atom_mass('H')]*6)[0]*units.nm_to_cm
        np.testing.assert_allclose(freq, freq_ref, atol=10.0)
        
    def test_from_internal(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.__from_internal(self.pes_bkmp2)

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.__from_internal(self.pes_cw)

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.__from_internal(self.pes_lwal)

    def __from_internal(self, pes):
        # colinear
        internal = np.array([2.0, 4.0, 0.0])
        cartesian_ref = np.array([4.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        cartesian = pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, pes._from_cartesian_to_internal(cartesian))

        # perpendicular
        internal = np.array([2.0, 4.0, np.pi/2.0])
        cartesian_ref = np.array([ 0.0, 4.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        cartesian = pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref, atol=1e-10)
        np.testing.assert_allclose(internal, pes._from_cartesian_to_internal(cartesian))

        # 45 degrees off
        internal = np.array([2.0, 4.0, np.pi/4.0])
        cartesian_ref = np.array([4.0/np.sqrt(2.0), 4.0/np.sqrt(2.0), 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        cartesian = pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, pes._from_cartesian_to_internal(cartesian))

        # -45 degrees off
        internal = np.array([2.0, 4.0, 2.0*np.pi-np.pi/4.0])
        cartesian_ref = np.array([4.0/np.sqrt(2.0), -4.0/np.sqrt(2.0), 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        cartesian = pes._from_internal(internal)
        np.testing.assert_allclose(cartesian, cartesian_ref)
        np.testing.assert_allclose(internal, pes._from_cartesian_to_internal(cartesian))

    def test_from_cartesian_to_internal(self):
        with self.subTest():
            if self.pes_bkmp2 is None:
                self.skipTest("BKMP2 PES not compiled properly.")
            self.__from_cartesian_to_internal(self.pes_bkmp2)

        with self.subTest():
            if self.pes_cw is None:
                self.skipTest("CW PES not compiled properly.")
            self.__from_cartesian_to_internal(self.pes_cw)

        with self.subTest():
            if self.pes_lwal is None:
                self.skipTest("LWAL PES not compiled properly.")
            self.__from_cartesian_to_internal(self.pes_lwal)

    def __from_cartesian_to_internal(self, pes):
        # colinear
        cartesian = np.array([3.8, 0.0, 0.0, -1.2, 0.0, 0.0, 1.2, 0.0, 0.0])
        internal_ref = np.array([2.4, 3.8, 0.0])
        internal = pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)
        np.testing.assert_allclose(cartesian, pes._from_internal(internal))

        # perpendicular
        cartesian = np.array([4.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0])
        internal_ref = np.array([2.0, 4.0, np.pi/2.0])
        internal = pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)

        # 'random' in space, 1st H along axis
        cartesian = np.array([-1.25 , -3.25, -1.18198052, 1.0, -1.0, 2.0, 0.5, -1.5, 2.0-1.0/np.sqrt(2.0)])
        internal_ref = np.array([1.0, 4.0, 2.0*np.pi])
        internal = pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)

        # -45 degrees off
        cartesian = np.array([4.0/np.sqrt(2.0), -4.0/np.sqrt(2.0), 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        internal_ref = np.array([2.0, 4.0, 2.0*np.pi-np.pi/4.0])
        internal = pes._from_cartesian_to_internal(cartesian)
        np.testing.assert_allclose(internal, internal_ref)
        np.testing.assert_allclose(cartesian, pes._from_internal(internal))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TriatomicTest)
    unittest.TextTestRunner().run(suite)
