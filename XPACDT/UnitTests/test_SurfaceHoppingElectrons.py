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

import math
import numpy as np
import random
import unittest

import XPACDT.System.SurfaceHoppingElectrons as sh
import XPACDT.Input.Inputfile as infile
import XPACDT.System.Nuclei as nuclei


class SurfaceHoppingTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.param_classical = infile.Inputfile("FilesForTesting/SystemTests/input_SH_classical.in")
        self.param_rpmd = infile.Inputfile("FilesForTesting/SystemTests/input_SH_rpmd.in")

    def test_creation(self):
        param = self.param_classical
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        self.assertEqual(sh_electrons_classical.name, 'SurfaceHoppingElectrons')
        self.assertEqual(sh_electrons_classical.basis, 'adiabatic')
        self.assertEqual(sh_electrons_classical.current_state, 0)
        self.assertEqual(sh_electrons_classical.rpsh_type, 'bead')
        self.assertEqual(sh_electrons_classical.rpsh_rescaling, 'bead')
        self.assertEqual(sh_electrons_classical.rescaling_type, 'nac')
        self.assertEqual(sh_electrons_classical.evolution_picture, 'schroedinger')
        self.assertEqual(sh_electrons_classical.ode_solver, 'runga_kutta')
        self.assertEqual(sh_electrons_classical.hop_status, 'No hop')
        return

    def test_energy(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # Adiabatic basis
        R = np.array([[0.]])

        param = self.param_classical
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=False),
                                   [-math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=True),
                                   -math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=False),
                                   [math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=True),
                                   math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=False),
                                   [0.0006], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=True),
                                   0.0006, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=False),
                                   [-0.0006], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.energy(R, centroid=True),
                                   -0.0006, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        # Adiabatic basis
        R = np.array([[-1.0e05, 1.0e05]])

        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=False),
                                   [-0.0006, -math.sqrt(0.04 + 3.6e-07)],
                                   rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=True),
                                   -math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=False),
                                   [0.0006, math.sqrt(0.04 + 3.6e-07)],
                                   rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=True),
                                   math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=False),
                                   [0.0006, 0.0006], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=True),
                                   0.0006, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=False),
                                   [-0.0006, -0.0006], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.energy(R, centroid=True),
                                   -0.0006, rtol=1e-7)

        return

    def test_gradient(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # Adiabatic basis
        R = np.array([[0.]])

        param = self.param_classical
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=False),
                                   [[-0.009 / math.sqrt(0.01 + 3.6e-07)]],
                                   rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=True),
                                   [-0.009 / math.sqrt(0.01 + 3.6e-07)],
                                   rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.009 / math.sqrt(0.01 + 3.6e-07)]],
                                   rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=True),
                                   [0.009 / math.sqrt(0.01 + 3.6e-07)],
                                   rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_classical.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        # Adiabatic basis
        R = np.array([[-1.0e05, 1.0e05]])

        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=True),
                                   [-0.009 / math.sqrt(0.01 + 3.6e-07)],
                                   rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.009 / math.sqrt(0.01 + 3.6e-07)],
                                   rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        return

    def test_get_velocity(self):
        # 1 bead, 1 dof test
        P = np.array([[2.0]])
        param = self.param_classical
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_velocity(P),
                                   [[1.0]], rtol=1e-7)
        p_centroid = np.array([2.0])
        np.testing.assert_allclose(sh_electrons_classical._get_velocity(p_centroid),
                                   [1.0], rtol=1e-7)

        # 2 beads, 1 dof test
        P = np.array([[2.0, 1.0]])
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_velocity(P),
                                   [[1.0, 0.5]], rtol=1e-7)

        # TODO: add test for more than 1 dof once higher dimensional potentials are available
        return

    def test_get_modified_V(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        param = self.param_classical

        # Adiabatic basis
        V_ref = np.array([[[-math.sqrt(0.01 + 3.6e-07), 0.],
                           [0., math.sqrt(0.01 + 3.6e-07)]]])
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        # Diabatic basis
        V_ref = np.array([[[0.0006, 0.1], [0.1, -0.0006]]])
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])
        param = self.param_rpmd

        # Adibatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[-V_ad_ref, 0.],
                                     [0., V_ad_ref]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[-math.sqrt(0.01 + 3.6e-07), 0.],
                                     [0., math.sqrt(0.01 + 3.6e-07)]]],
                                   rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[-0.0006, 0.], [0., 0.0006]],
                                    [[-math.sqrt(0.04 + 3.6e-07), 0.],
                                     [0., math.sqrt(0.04 + 3.6e-07)]]],
                                   rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.1],
                                     [0.1, -0.0006]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.1],
                                     [0.1, -0.0006]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.0], [0.0, -0.0006]],
                                    [[0.0006, 0.2], [0.2, -0.0006]]],
                                   rtol=1e-7)

        return

    def test_get_kinetic_coupling_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        P = np.array([[4.]])
        nac_ref = -2.7e-05/(0.01 + 3.6e-07)
        param = self.param_classical

        # Adiabatic basis
        D_ref = np.array([[[0., 2.0 * nac_ref], [-2.0 * nac_ref, 0.]]])
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        # Diabatic basis - should just return None for all cases.
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        self.assertIsNone(sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        self.assertIsNone(sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        self.assertIsNone(sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        ### test for 2 state potential, 1 dof, 2 beads
        # Adibatic basis
        R = np.array([[-1.0e05, 1.0e05]])
        P = np.array([[2., 6.]])
        param = self.param_rpmd

        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., 2.0 * nac_ref],
                                     [-2.0 * nac_ref, 0.]]], rtol=1e-7)

        R = np.array([[0., 1.0e05]])
        P = np.array([[4., 2.]])
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., nac_ref],
                                     [-nac_ref, 0.]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., 2.0 * nac_ref], [-2.0 * nac_ref, 0.]],
                                    [[0., 0.], [0., 0.]]], rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        self.assertIsNone(sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P))

        # TODO: add test for more than 1 dof once higher dimensional potentials are available

        return

    def test_get_H_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        P = np.array([[4.]])
        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        D_ref = -2. * 2.7e-05/(0.01 + 3.6e-07)
        param = self.param_classical

        # Adiabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        H_ref_schroedinger = np.array([[[-V_ad_ref + 0.0j, -1j * D_ref],
                                        [1j * D_ref, V_ad_ref + 0.0j]]])
        H_ref_interaction = np.array([[[0.0 + 0.0j, -1j * D_ref],
                                       [1j * D_ref, 0.0 + 0.0j]]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        D = sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        D = sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        D = sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        H_ref_schroedinger = np.array([[[0.0006 + 0.0j, 0.1 + 0.0j],
                                        [0.1 + 0.0j, -0.0006 + 0.0j]]])
        H_ref_interaction = np.array([[[0.0 + 0.0j, 0.1 + 0.0j],
                                       [0.1 + 0.0j, 0.0 + 0.0j]]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])
        P = np.array([[2., 6.]])
        param = self.param_rpmd

        # Adibatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        D = sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-V_ad_ref + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, V_ad_ref + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        D_ref = -2. * 2.7e-05/(0.01 + 3.6e-07)
        D = sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-V_ad_ref + 0.0j, -1j * D_ref],
                                     [1j * D_ref, V_ad_ref + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, -1j * D_ref],
                                     [1j * D_ref, 0.0 + 0.0j]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        D = sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-0.0006 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0006 + 0.0j]],
                                    [[-math.sqrt(0.04 + 3.6e-07) + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, math.sqrt(0.04 + 3.6e-07) + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0 + 0.0j]],
                                    [[0.0 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, -0.0006 + 0.0j]],
                                    [[0.0006 + 0.0j, 0.2 + 0.0j],
                                     [0.2 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, 0.0 + 0.0j]],
                                    [[0.0 + 0.0j, 0.2 + 0.0j],
                                     [0.2 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)
        return

    def test_get_diff_diag_V_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        param = self.param_classical

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        with self.assertRaises(AssertionError):
            sh_electrons_classical._get_diff_diag_V_matrix(R)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        # Adiabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'

        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        diff_ref = np.array([[[0., 2. * V_ad_ref],
                              [-2. * V_ad_ref, 0.]]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)

        # Diabatic basis
        V_ref = np.array([[[0., -0.0012], [0.0012, 0.]]])
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        np.testing.assert_allclose(sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        # Adibatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 2. * V_ad_ref],
                                     [-2. * V_ad_ref, 0.]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 2. * math.sqrt(0.01 + 3.6e-07)],
                                     [-2. * math.sqrt(0.01 + 3.6e-07), 0.]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 0.0012], [-0.0012, 0.]],
                                    [[0., 2. * math.sqrt(0.04 + 3.6e-07)],
                                     [-2. * math.sqrt(0.04 + 3.6e-07), 0.]]], rtol=1e-7)

        # Diabatic basis
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'centroid'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        np.testing.assert_allclose(sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]],
                                    [[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)

        return

    def test_get_a_jk(self):
        ### Schroedinger picture
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        # 1 c-coefficient
        c = np.array([[1.+0.j, 0.+0.j]])
        np.testing.assert_allclose(sh_electrons_classical._get_a_kj(c),
                                   [[[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]],
                                   rtol=1e-7)

        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        np.testing.assert_allclose(sh_electrons_classical._get_a_kj(c),
                                   [[[0.5+0.j, 0.+0.5j], [0.-0.5j, 0.5+0.j]]],
                                   rtol=1e-7)

        # 2 c-coefficients
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        np.testing.assert_allclose(sh_electrons_rpmd._get_a_kj(c),
                                   [[[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                                    [[0.5+0.j, 0.+0.5j], [0.-0.5j, 0.5+0.j]]],
                                   rtol=1e-7)

        ### Interaction picture
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        # 1 c-coefficient
        phase = np.array([[[0., 0.], [0., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        np.testing.assert_allclose(sh_electrons_classical._get_a_kj(c, phase),
                                   [[[0.5+0.j, 0.+0.5j], [0.-0.5j, 0.5+0.j]]],
                                   rtol=1e-7)

        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j]])
        np.testing.assert_allclose(sh_electrons_classical._get_a_kj(c, phase),
                                   [[[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]],
                                   rtol=1e-7)
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        np.testing.assert_allclose(sh_electrons_classical._get_a_kj(c, phase),
                                   [[[0.5+0.j, -0.5+0.j], [-0.5+0.j, 0.5+0.j]]],
                                   rtol=1e-7)

        # 2 c-coefficients
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        np.testing.assert_allclose(sh_electrons_rpmd._get_a_kj(c, phase),
                                   [[[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                                    [[0.5+0.j, -0.5+0.j], [-0.5+0.j, 0.5+0.j]]],
                                   rtol=1e-7)
        return

    def test_get_b_jk(self):
        # First test without interpolation (i.e. using initial H).

        ### Adiabatic basis, Schroedinger picture
        # 1 c-coefficient; checking for rpsh_type == 'bead' and 'density_matrix'
        # 'centroid' isn't checked as it is the same as 'bead' with only 1 c-coeff.
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)

        # 2 c-coefficients; this only makes sense for rpsh_type == 'density_matrix'
        # as other types cannot have multiple c-coefficients.
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., 0.], rtol=1e-7)

        ### Adiabatic basis, interaction picture
        # Note: in interaction picture, diagonal elements of H must be 0.
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., -0.1], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0.1, 0.], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., -0.1], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0.1, 0.], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., -0.05], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0.05, 0.], rtol=1e-7)

        ### Diabatic basis, Schroedinger picture
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.1], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [-0.1, 0.], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.1], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [-0.1, 0.], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]],
                      [[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., 0.05], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [-0.05, 0.], rtol=1e-7)

        ### Diabatic basis, interaction picture
        # Note: in interaction picture, diagonal elements of H must be 0.
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)

        # 2 c-coefficients
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 0.+0.j]],
                      [[0.+0.j, 0.1+0.j], [0.1+0.j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0., 1., c),
                                   [0., 0.], atol=1e-8)

        # Test for proper interpolation at mid-point or quarter-point.
        ### Diabatic basis, Schroedinger picture with interpolation
        # 1 c-coefficient; only rpsh_type == 'bead' is tested as all give same result.
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.5, 1., c),
                                   [0., 0.1], rtol=1e-7)
        sh_electrons_classical._H_e_total = 4. * H
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.25, 1., c),
                                   [0., 0.1], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.5, 1., c),
                                   [-0.1, 0.], rtol=1e-7)
        sh_electrons_classical._H_e_total = 4. * H
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.25, 1., c),
                                   [-0.1, 0.], rtol=1e-7)

        # 2 c-coefficients; only rpsh_type == 'density_matrix' makes sense here.
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["basis"] = 'diabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'

        H = np.array([[[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]],
                      [[0.+0.j, 0.1+0.j], [0.1+0.j, 1.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0.5, 1., c),
                                   [0., 0.05], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0.5, 1., c),
                                   [-0.05, 0.], rtol=1e-7)

        ### Adiabatic basis, interaction picture with interpolation
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'bead'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase doesn't change
        sh_electrons_classical._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_classical._diff_diag_V = 3. * np.ones_like(phase)
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.5, 1., c),
                                   [0., -0.1], rtol=1e-7)
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase doesn't change
        sh_electrons_classical._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_classical._diff_diag_V = 3. * np.ones_like(phase)
        np.testing.assert_allclose(sh_electrons_classical._get_b_jk(0.5, 1., c),
                                   [0.1, 0.], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["basis"] = 'adiabatic'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        sh_electrons_rpmd._phase = phase.copy()
        sh_electrons_rpmd._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_rpmd._diff_diag_V = 3. * np.ones_like(phase)
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0.5, 1., c),
                                   [0., -0.05], rtol=1e-7)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        sh_electrons_rpmd._phase = phase.copy()
        sh_electrons_rpmd._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_rpmd._diff_diag_V = 3. * np.ones_like(phase)
        np.testing.assert_allclose(sh_electrons_rpmd._get_b_jk(0.5, 1., c),
                                   [0.05, 0.], rtol=1e-7)

        return

    @unittest.skip("Please implement a test here.")
    def test_get_drho_dt_dm_cb(self):

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'dm_cb'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        param["SurfaceHoppingElectrons"]["n_steps"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)

        #R = np.array([[-1.0e05, -1.0e05]])
        R = np.array([[-7.5, -7.5]])
        P = np.array([[10.0, 10.0]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        D = sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        H = sh_electrons_rpmd._get_H_matrix(R, D)

        #H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]],
        #              [[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._c_coeff = c.copy()

        # Propagate for a bit
        for i in range(5):
            sh_electrons_rpmd.step(R, P, 1, **{'step_index': 'after_nuclei',
                                               'step_count': i})

        rho = sh_electrons_rpmd._get_rho_dm_cb(sh_electrons_rpmd._c_coeff,
                                               R)
        print(rho.shape)
        print(rho)

        drho_dt = sh_electrons_rpmd._get_drho_dt_dm_cb(sh_electrons_rpmd._c_coeff,
                                                       R, P)
        print(drho_dt.shape)
        print(drho_dt)

        return

    @unittest.skip("Please implement a test here.")
    def test_step(self):
        # TODO: this seems more like a integrated test, what exactly should be tested here?
        # Test all ode solvers give same result in all pictures after long propagation
        # Test norm conservation after long propagation.

        # NAC at R=0 in Tully C is negligible so there should not be any change in momenta.

        # Schroedinger picture
        R = np.array([[-1.0e5]])
        P = np.array([[10.0]])
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        param["SurfaceHoppingElectrons"]["n_steps"] = 100
        #param["TullyModel"]["model_type"] = "model_A"
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)

        with self.assertRaises(AssertionError):
            sh_electrons_classical.step(R, P, 1.)

        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        H_ref = np.array([[[-0.0006+0.j, 0.+0.j], [0.+0.j, 0.0006+0.j]]])
        sh_electrons_classical._old_H_e = np.zeros_like(H_ref)
        sh_electrons_classical._c_coeff = c.copy()


        sh_electrons_classical.step(R, P, 1., **{'step_index': 'after_nuclei'})
#        np.testing.assert_allclose(
#                sh_electrons_classical._integrator_scipy(0., c, math.pi, 2.*math.pi, prop_func),
#                c_ref, rtol=1e-4)

        # No change
        sh_electrons_classical.step(R, P, 1., **{'step_index': 'before_nuclei'})

        # Test for change
        sh_electrons_classical.step(R, P, 1., **{'step_index': 'after_nuclei'})

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        param["SurfaceHoppingElectrons"]["n_steps"] = 100
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)

        R = np.array([[-1.0e5, -1.0e5]])
        P = np.array([[10.0, 10.0]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        H_ref = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                          [[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H_ref)
        sh_electrons_rpmd._c_coeff = c.copy()
        c_ref = [[0.-1.j, 0.+0.j],
                 [(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]

        sh_electrons_rpmd.step(R, P, 1., **{'step_index': 'after_nuclei'})

        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_scipy(0., c, math.pi, 2.*math.pi, prop_func),
                c_ref, rtol=1e-4)

        return

    # Simple exact solutions for the matrix ODE can be obtained for either
    # constant H or if H commutes with its integral over time (which is usually
    # not the case unless it is diagonal). So for these integrator tests below,
    # two distinct cases are tested: 1. when H is constant, and 2. when H is
    # diagonal and linearly varying due to linear interpolation.
    # TODO: how to get exact values for more general case?

    def test_integrator_runga_kutta(self):
        ### When H(t) is diagonal
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_schroedinger_picture
        # Since Runga-Kutta here doesn't have adaptive time step, need smaller
        # step to get result within its error.
        t_step = 0.1
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H

        c_ref = [[(1. / math.sqrt(2)) * np.exp(-1.j*0.5*t_step),
                  -(1.j / math.sqrt(2)) * np.exp(-1.j*t_step)]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, t_step,
                                                               2.*t_step, prop_func),
                c_ref, rtol=1e-5)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        sh_electrons_classical._phase = np.array([[[0., 0.], [0., 0.]]])
        # These values are chosen such that it matches the H in schroedinger
        # picture, although it doesn't really matter.
        sh_electrons_classical._old_diff_diag_V = np.array([[[0., 0.], [0., 0.]]])
        sh_electrons_classical._diff_diag_V = np.array([[[0., 1.], [-1., 0.]]])
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, math.pi,
                                                               2.*math.pi, prop_func),
                c_ref, rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_schroedinger_picture

        t_step = 0.1
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H.copy()
        c_ref = [[np.exp(-1.j*0.5*t_step), 0.+0.j],
                 [(1. / math.sqrt(2)) * np.exp(-1.j*0.5*t_step),
                  -(1.j / math.sqrt(2)) * np.exp(-1.j*t_step)]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_runga_kutta(0., c, t_step,
                                                          2.*t_step, prop_func),
                c_ref, rtol=1e-5)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                      [[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., 0.], [0., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        # These values are chosen such that it matches the H in schroedinger
        # picture, although it doesn't really matter.
        sh_electrons_rpmd._old_diff_diag_V = phase.copy()
        sh_electrons_rpmd._diff_diag_V = np.array([[[0., 1.], [-1., 0.]],
                                                   [[0., 1.], [-1., 0.]]])
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_runga_kutta(0., c, math.pi,
                                                          2.*math.pi, prop_func),
                c_ref, rtol=1e-7)

        ### When H(t) is constant
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_schroedinger_picture

        t_step = 0.1
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        c_ref = [[(1. / math.sqrt(2)) * np.exp(-1.j*t_step),
                  -(1.j / math.sqrt(2)) * np.exp(-1.j*2.*t_step)]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, t_step,
                                                               2.*t_step, prop_func),
                c_ref, rtol=1e-5)

        t_step = 0.1
        H = np.array([[[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()

        exp_plus = np.exp(-1.j*t_step) + np.exp(-3.j*t_step)
        exp_minus = np.exp(-1.j*t_step) - np.exp(-3.j*t_step)
        c_ref = 0.5 / math.sqrt(2) * np.array([[exp_plus + 1.j*exp_minus,
                                                -exp_minus - 1.j*exp_plus]])
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, t_step,
                                                               2.*t_step, prop_func),
                c_ref, rtol=1e-4)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_classical._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_classical._diff_diag_V = np.zeros_like(phase)
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, math.pi/2.,
                                                               math.pi, prop_func),
                c_ref, rtol=1e-7)

        t_step = 0.1
        H = np.array([[[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_classical._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_classical._diff_diag_V = np.zeros_like(phase)

        exp_plus = np.exp(-1.j*t_step) + np.exp(-3.j*t_step)
        exp_minus = np.exp(-1.j*t_step) - np.exp(-3.j*t_step)
        c_ref = 0.5 / math.sqrt(2) * np.exp(2.j*t_step) \
            * np.array([[exp_plus + 1.j*exp_minus, -exp_minus - 1.j*exp_plus]])
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_runga_kutta(0., c, t_step,
                                                               2.*t_step, prop_func),
                c_ref, rtol=1e-5)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_schroedinger_picture

        t_step = 0.1
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()

        exp_plus = np.exp(-1.j*t_step) + np.exp(-3.j*t_step)
        exp_minus = np.exp(-1.j*t_step) - np.exp(-3.j*t_step)
        c_ref = np.array([[(1. / math.sqrt(2)) * np.exp(-1.j*t_step),
                           -(1.j / math.sqrt(2)) * np.exp(-1.j*2.*t_step)],
                          [0.5 / math.sqrt(2) * (exp_plus + 1.j*exp_minus),
                           0.5 / math.sqrt(2) * (-exp_minus - 1.j*exp_plus)]])
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_runga_kutta(0., c, t_step,
                                                          2.*t_step, prop_func),
                c_ref, rtol=1e-4)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_interaction_picture

        t_step = 0.1
        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                      [[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]],
                          [[0., 0.], [0., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_rpmd._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_rpmd._diff_diag_V = np.zeros_like(phase)

        exp_plus = np.exp(-1.j*t_step) + np.exp(-3.j*t_step)
        exp_minus = np.exp(-1.j*t_step) - np.exp(-3.j*t_step)
        c_ref = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                          [0.5 / math.sqrt(2) * np.exp(2.j*t_step) * (exp_plus + 1.j*exp_minus),
                           0.5 / math.sqrt(2) * np.exp(2.j*t_step) * (-exp_minus - 1.j*exp_plus)]])
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_runga_kutta(0., c, t_step,
                                                          2.*t_step, prop_func),
                c_ref, rtol=1e-5)

        return

    def test_integrator_scipy(self):
        ### When H(t) is diagonal
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_schroedinger_picture

        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H

        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi, 2.*math.pi, prop_func),
                c_ref, rtol=1e-4)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        sh_electrons_classical._phase = np.array([[[0., 0.], [0., 0.]]])
        # These values are chosen such that it matches the H in schroedinger
        # picture, although it doesn't really matter.
        sh_electrons_classical._old_diff_diag_V = np.array([[[0., 0.], [0., 0.]]])
        sh_electrons_classical._diff_diag_V = np.array([[[0., 1.], [-1., 0.]]])
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi,
                                                         2.*math.pi, prop_func),
                c_ref, rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_schroedinger_picture

        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H.copy()
        c_ref = [[0.-1.j, 0.+0.j],
                 [(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_scipy(0., c, math.pi,
                                                    2.*math.pi, prop_func),
                c_ref, rtol=1e-4)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                      [[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., 0.], [0., 0.]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        # These values are chosen such that it matches the H in schroedinger
        # picture, although it doesn't really matter.
        sh_electrons_rpmd._old_diff_diag_V = phase.copy()
        sh_electrons_rpmd._diff_diag_V = np.array([[[0., 1.], [-1., 0.]],
                                                   [[0., 1.], [-1., 0.]]])
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_scipy(0., c, math.pi,
                                                    2.*math.pi, prop_func),
                c_ref, rtol=1e-7)

        ### When H(t) is constant
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_schroedinger_picture

        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi/2.,
                                                         math.pi, prop_func),
                c_ref, rtol=1e-5)

        H = np.array([[[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        c_ref = [[(1./math.sqrt(2))+0.j, (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi/2.,
                                                         math.pi, prop_func),
                c_ref, rtol=1e-5)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        prop_func = sh_electrons_classical._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_classical._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_classical._diff_diag_V = np.zeros_like(phase)
        # No change in interaction picture.
        c_ref = c.copy()
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi/2.,
                                                         math.pi, prop_func),
                c_ref, rtol=1e-7)

        H = np.array([[[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]])
        phase = np.array([[[0., 0.], [0., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_classical._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_classical._diff_diag_V = np.zeros_like(phase)

        c_ref = [[(-1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_scipy(0., c, math.pi/2.,
                                                         math.pi, prop_func),
                c_ref, rtol=1e-5)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses, param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_schroedinger_picture

        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))],
                 [(1./math.sqrt(2))+0.j, (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_scipy(0., c, math.pi/2.,
                                                    math.pi, prop_func),
                c_ref, rtol=1e-5)

        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        prop_func = sh_electrons_rpmd._propagation_equation_interaction_picture

        H = np.array([[[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]],
                      [[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]],
                          [[0., 0.], [0., 0.]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        # These values are chosen such that phase is constant.
        sh_electrons_rpmd._old_diff_diag_V = np.zeros_like(phase)
        sh_electrons_rpmd._diff_diag_V = np.zeros_like(phase)

        c_ref = [[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                 [(-1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_scipy(0., c, math.pi/2.,
                                                    math.pi, prop_func),
                c_ref, rtol=1e-5)

        return

    def test_integrator_unitary(self):
        # Unitary evolution requires c-coefficients to be in Schroedinger picture.
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["ode_solver"] = 'unitary'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        with self.assertRaises(ValueError):
            sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                                param.masses,
                                                                param.coordinates,
                                                                param.momenta)

        ### When H(t) is diagonal
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H

        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_unitary(0., c, math.pi,
                                                           2.*math.pi),
                c_ref, rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H.copy()
        c_ref = [[0.-1.j, 0.+0.j],
                 [(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_unitary(0., c, math.pi,
                                                      2.*math.pi),
                c_ref, rtol=1e-7)

        ### When H(t) is constant
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_unitary(0., c, math.pi/2.,
                                                           math.pi),
                c_ref, rtol=1e-7)

        H = np.array([[[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        c_ref = [[(1./math.sqrt(2))+0.j, (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._integrator_unitary(0., c, math.pi/2.,
                                                           math.pi),
                c_ref, rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'schroedinger'
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        H = np.array([[[1.+0.j, 0.+0.j], [0.+0.j, 2.+0.j]],
                      [[2.+0.j, 1.+0.j], [1.+0.j, 2.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        c_ref = [[(-1.j / math.sqrt(2)), (1.j / math.sqrt(2))],
                 [(1./math.sqrt(2))+0.j, (1.j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._integrator_unitary(0., c, math.pi/2.,
                                                      math.pi),
                c_ref, rtol=1e-7)

        return

    def test_propagation_equation_schroedinger_picture(self):
        ### First test without interpolation (i.e. using initial H).
        # 1 c-coefficient
        param = self.param_classical

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        derivative_ref = [[0. + (0.1j / math.sqrt(2)), -0.9 / math.sqrt(2) + 0.j]]
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_schroedinger_picture(0., c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_schroedinger_picture(0., c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        derivative_ref = [[0.+0.j, 0.1+0.j],
                          [0. + (0.1j / math.sqrt(2)), -0.9 / math.sqrt(2) + 0.j]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0., c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0., c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0., c[1], 1., 1),
                derivative_ref[1], rtol=1e-7)

        ### Test for proper interpolation at mid-point.
        # 1 c-coefficient
        param = self.param_classical

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        derivative_ref = [[0. + (0.1j / math.sqrt(2)), -0.9 / math.sqrt(2) + 0.j]]
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_schroedinger_picture(0.5, c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_schroedinger_picture(0.5, c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 1.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])

        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        derivative_ref = [[0.+0.j, 0.1+0.j],
                          [0. + (0.1j / math.sqrt(2)), -0.9 / math.sqrt(2) + 0.j]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0.5, c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0.5, c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_schroedinger_picture(0.5, c[1], 1., 1),
                derivative_ref[1], rtol=1e-7)
        return

    def test_propagation_equation_interaction_picture(self):
        # Note: in interaction picture, diagonal elements of H must be 0.

        ### First test without interpolation (i.e. using initial H).
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])

        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = H.copy()
        sh_electrons_classical._H_e_total = H.copy()
        sh_electrons_classical._phase = phase.copy()
        derivative_ref = [[(0.1 / math.sqrt(2)) + 0.j, 0. + (0.1j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_interaction_picture(0., c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_interaction_picture(0., c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])

        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = H.copy()
        sh_electrons_rpmd._H_e_total = H.copy()
        sh_electrons_rpmd._phase = phase.copy()
        derivative_ref = [[0.+0.j, 0.1+0.j],
                          [(0.1 / math.sqrt(2)) + 0.j, 0. + (0.1j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0., c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0., c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0., c[1], 1., 1),
                derivative_ref[1], rtol=1e-7)

        ### Test for proper interpolation at mid-point.
        # 1 c-coefficient
        param = self.param_classical
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        c = np.array([[(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        phase = np.array([[[0., math.pi/2.], [-math.pi/2., 0.]]])

        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical._old_H_e = np.zeros_like(H)
        sh_electrons_classical._H_e_total = 2. * H
        sh_electrons_classical._phase = phase.copy()
        # These values are chosen such that phase doesn't change
        sh_electrons_classical._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_classical._diff_diag_V = 3. * np.ones_like(phase)

        derivative_ref = [[(0.1 / math.sqrt(2)) + 0.j,
                           0. + (0.1j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_interaction_picture(0.5, c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_classical._propagation_equation_interaction_picture(0.5, c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)

        # 2 c-coefficients
        param = self.param_rpmd
        param["SurfaceHoppingElectrons"]["evolution_picture"] = 'interaction'
        param["SurfaceHoppingElectrons"]["rpsh_type"] = 'density_matrix'

        H = np.array([[[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]],
                      [[0.+0.j, 0.-0.1j], [0.+0.1j, 0.+0.j]]])
        c = np.array([[1.+0.j, 0.+0.j],
                      [(1. / math.sqrt(2))+0.j, 0.-(1.j / math.sqrt(2))]])
        phase = np.array([[[0., 0.], [0., 0.]],
                          [[0., math.pi/2.], [-math.pi/2., 0.]]])

        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd._old_H_e = np.zeros_like(H)
        sh_electrons_rpmd._H_e_total = 2. * H
        sh_electrons_rpmd._phase = phase.copy()
        # These values are chosen such that phase doesn't change
        sh_electrons_rpmd._old_diff_diag_V = -1. * np.ones_like(phase)
        sh_electrons_rpmd._diff_diag_V = 3. * np.ones_like(phase)

        derivative_ref = [[0.+0.j, 0.1+0.j],
                          [(0.1 / math.sqrt(2)) + 0.j,
                           0. + (0.1j / math.sqrt(2))]]
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0.5, c, 1.),
                derivative_ref, rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0.5, c[0], 1., 0),
                derivative_ref[0], rtol=1e-7)
        np.testing.assert_allclose(
                sh_electrons_rpmd._propagation_equation_interaction_picture(0.5, c[1], 1., 1),
                derivative_ref[1], rtol=1e-7)
        return

    def test_surface_hopping(self):
        # Test with 3 states
        params = infile.Inputfile("FilesForTesting/SystemTests/input_SH_classical_3_states.in")

        # R is chosen where coupling in morse diabatic potentials is maximum
        # and P is chosen to be large enough to have enough energy to hop.
        R = np.array([[3.4]])
        P = np.array([[10.]])

        t = [0., 1., 2.]
        a_kk = 0.5
        b_jk = [np.array([0., 0.025, 0.03]), np.array([0., 0.05, 0.1]),
                np.array([0., 0.075, 0.07])]

        ### Hop from 1st state to other states above it
        params["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)

        # Probabilities should be [0, 0.2, 0.3]
        # and random number should be 0.844421851, so no hop
        sh_classical_3_states._surface_hopping(R, P, t, a_kk, b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 0)
        self.assertEqual(sh_classical_3_states.hop_status, 'No hop')

        random.seed(1)
        # Now random number should be 0.13436424; so there is a hop to 2nd
        # state using the same probabilities
        sh_classical_3_states._surface_hopping(R, P, t, a_kk, b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 1)

        hop_status_ref = "Successful hop from state 0 to state 1"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        random.seed(3)
        params["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)
        # Now random number should be 0.23796463.; so there is a hop to 3rd
        # state using the same probabilities
        sh_classical_3_states._surface_hopping(R, P, t, a_kk, b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 2)

        hop_status_ref = "Successful hop from state 0 to state 2"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        ### Hop from 3rd state to other states below
        params["SurfaceHoppingElectrons"]["initial_state"] = 2
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)
        b_jk = [np.array([0.03, 0.0, 0.03]), np.array([0.3, 0.0, 0.1]),
                np.array([0.07, -0.05, 0.07])]
        # Probabilities should be [0.7, 0., 0.]
        # and random number should be 0.54422922, so hop to 1st state
        sh_classical_3_states._surface_hopping(R, P, t, a_kk, b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 0)

        hop_status_ref = "Successful hop from state 2 to state 0"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        params["SurfaceHoppingElectrons"]["initial_state"] = 2
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)
        b_jk = [np.array([0.0, 0.03, 0.03]), np.array([0., 0.3, 0.1]),
                np.array([-0.05, 0.07, 0.07])]
        # Probability should be [0., 0.7, 0.]
        # and random number should be 0.36995517, so hop to 2nd state
        sh_classical_3_states._surface_hopping(R, P, t, a_kk, b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 1)

        hop_status_ref = "Successful hop from state 2 to state 1"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        # Even with P=0, there should still be hops from higher state to lower.
        params["SurfaceHoppingElectrons"]["initial_state"] = 2
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)
        b_jk = [np.array([0.0, 0.03, 0.03]), np.array([0., 0.3, 0.1]),
                np.array([-0.05, 0.07, 0.07])]
        # Again the probability should be [0., 0.7, 0.]
        # and random number should be 0.6039200, so hop to 2nd state
        sh_classical_3_states._surface_hopping(R, np.array([[0.]]), t, a_kk,
                                               b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 1)

        hop_status_ref = "Successful hop from state 2 to state 1"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        ### Attempted hop with P=0 from lower state to higher one.
        params["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_classical_3_states = sh.SurfaceHoppingElectrons(params,
                                                           params.masses,
                                                           params.coordinates,
                                                           params.momenta)
        random.seed(1)
        b_jk = [np.array([0., 0.025, 0.03]), np.array([0., 0.05, 0.1]),
                np.array([0., 0.075, 0.07])]
        # Probabilities should be [0, 0.2, 0.3]
        # And the random number should be 0.13436424; so there is an attempt to
        # hop to 2nd state, but since P=0, it should not be successful.
        sh_classical_3_states._surface_hopping(R, np.array([[0.]]), t, a_kk,
                                               b_jk)
        self.assertTrue(sh_classical_3_states.current_state == 0)

        hop_status_ref = "Attempted hop from state 0 to state 1"
        self.assertEqual(sh_classical_3_states.hop_status, hop_status_ref)

        return

    def test_momentum_rescaling(self):
        # Here momentum rescaling for (A) 1 bead and (B) 2 beads cases are
        # tested. For this, nuclei are initialized instead of electrons to test
        # for conservation of energy after hop by comparing to with initial
        # energy (done using ndarray.copy() for numpy float to copy without
        # aliasing.
        # The different rescalings are tested here, namely, (a) NAC,
        # (b) diabatic gradient and (c) adiabatic gradient rescaling.
        # And also for (1) centroid or (2) bead energy conservation scheme.
        # Note: 1 bead case gives same result for both energy conservation.

        # Here Tully model A and C are used as PES. The diabatic reference
        # values can be seen in the test for Tully models and adiabatic values
        # for model C in test for interface template.
        # For adiabatic reference values for model A:
        # at R = 0:     E = [[-0.005, 0], [0, 0.005]], dE/dR = [[0, 0], [0, 0]]
        #         NAC = [[0, 1.6], [-1.6, 0]]
        # at R = -10^5: E = [[-0.01, 0], [0, 0.01]], dE/dR = [[0, 0], [0, 0]]
        #         NAC = [[0, 0], [0, 0]]

        # TODO: Add test for more than 1 dof once higher dimensional multistate
        #       potentials are available. And check if both energy conservation
        #       schemes conserve both centroid and bead energy simultaneously,
        #       which seems to be the case for 1 dof system (why though?)

        # (A) 1 bead case
        R = np.array([[0.0]])
        param = self.param_classical
        param["TullyModel"]["model_type"] = "model_A"

        # (A)(a) NAC rescaling
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "nac"
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"

        # (A)(a)(1) centroid rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"

        # From state 1 to 2, not always hop allowed due to insufficient energy.
        # For -0.2 < P < 0.2, hops are not allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)
        P = np.array([[0.1]])

        # Hopping to same state is not allowed
        with self.assertRaises(AssertionError):
            nuclei_classical.electrons._momentum_rescaling(R, P, 0)

        E_ref = nuclei_classical.energy_centroid.copy()
        # Here hop not allowed so P should be the same.
        self.assertFalse(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                        1))
        np.testing.assert_allclose(P, [[0.1]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[0.2]])
        E_ref = nuclei_classical.energy_centroid.copy()
        # Here hop is allowed so P changes.
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[0.0]], atol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.2]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[0.2 * math.sqrt(2)]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[-0.2]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[-0.2 * math.sqrt(2)]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[-0.2]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # (A)(a)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"

        # From state 1 to 2, not always hop allowed due to insufficient energy.
        # For -0.2 < P < 0.2, hops are not allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.1]])
        E_ref = nuclei_classical.energy.copy()
        self.assertFalse(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                        1))
        np.testing.assert_allclose(P, [[0.1]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[0.2]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[0.0]], atol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.2]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[0.2 * math.sqrt(2)]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[-0.2]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[-0.2 * math.sqrt(2)]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[-0.2]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # (A)(b) diabatic gradient rescaling - Here no change in P since for
        # Tully A at R=0, V11 == V22, but always hop allowed.
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "gradient"
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"

        # (A)(b)(1) centroid rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"

        # From state 1 to 2
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[1.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[-1.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[-1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # From state 2 to 1
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[1.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # (A)(b)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"

        # From state 1 to 2
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[1.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[-1.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[-1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # From state 2 to 1, always hop
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[1.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # (A)(c) adiabatic gradient rescaling
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "gradient"
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"

        # (A)(c)(1) centroid rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"

        # From state 1 to 2
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)
        P = np.array([[1.]])

        # First try using Tully model A at R = 0, however A_kj = 0 here which
        # shouldn't be allowed.
        with self.assertRaises(AssertionError):
            nuclei_classical.electrons._momentum_rescaling(R, P, 1)

        # So use Tully model C instead. Here hop is not allowed for
        # -p_hop < P < p_hop where p_hop = math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))
        param["TullyModel"]["model_type"] = "model_C"
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertFalse(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                        1))
        np.testing.assert_allclose(P, [[0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[0.]], atol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[math.sqrt(16 * math.sqrt(0.01 + 3.6e-07))]],
                                   rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy_centroid.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]],
                                   rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy_centroid, E_ref)

        # (A)(c)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"
        # First try with Tully model A
        param["TullyModel"]["model_type"] = "model_A"

        # From state 1 to 2
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_classical = nuclei.Nuclei(param, 0.0)
        P = np.array([[1.]])

        # A_kj = 0 here which shouldn't be allowed.
        with self.assertRaises(AssertionError):
            nuclei_classical.electrons._momentum_rescaling(R, P, 1)

        # So use Tully model C instead.
        param["TullyModel"]["model_type"] = "model_C"
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertFalse(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                        1))
        np.testing.assert_allclose(P, [[0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       1))
        np.testing.assert_allclose(P, [[0.]], atol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_classical = nuclei.Nuclei(param, 0.0)

        P = np.array([[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[math.sqrt(16 * math.sqrt(0.01 + 3.6e-07))]],
                                   rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        P = np.array([[0.]])
        E_ref = nuclei_classical.energy.copy()
        self.assertTrue(nuclei_classical.electrons._momentum_rescaling(R, P,
                                                                       0))
        np.testing.assert_allclose(P, [[math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))]],
                                   rtol=1e-7)
        self.assertAlmostEqual(nuclei_classical.energy, E_ref)

        # (B) 2 bead case
        param = self.param_rpmd
        param["TullyModel"]["model_type"] = "model_A"

        # (B)(a) NAC rescaling
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "nac"
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"

        # (B)(a)(1) centroid rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"
        R = np.array([[-1., 1.]])

        # From state 1 to 2, not always hop allowed due to insufficient energy.
        # For -0.2 < P_centroid < 0.2, hops are not allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[-0.2, 0.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertFalse(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[-0.2, 0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        P = np.array([[0.1, 0.3]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[-0.1, 0.1]], atol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0.1, 0.3]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[-0.1 + 0.2 * math.sqrt(2),
                                        0.1 + 0.2 * math.sqrt(2)]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        P = np.array([[-1., 1.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[-1.2, 0.8]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # (B)(a)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"
        R = np.array([[0., -1.0e5]])

        # From state 1 to 2, not always hop allowed due to insufficient energy.
        # For -math.sqrt(0.12) < P[0] < math.sqrt(0.12), hops are not allowed,
        # P[1] (value for second bead) doesn't matter.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertFalse(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        P = np.array([[math.sqrt(0.12), 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], atol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[math.sqrt(0.12), 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[math.sqrt(0.24), 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        P = np.array([[-math.sqrt(0.12), 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[-math.sqrt(0.24), 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        # (B)(b) diabatic gradient rescaling
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "gradient"
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"

        # (B)(b)(1) centroid rescaling - Here no change in P since for
        # Tully A at R_centroid = 0, V11 == V22, but always hop allowed.
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"
        R = np.array([[-1., 1.]])

        # From state 1 to 2
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 2.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 2.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # From state 2 to 1
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[-0.2, 0.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[-0.2, 0.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # (B)(b)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"
        R = np.array([[0., -1.0e5]])

        # From state 1 to 2, not always hop allowed due to insufficient energy.
        # For -math.sqrt(0.08) < P[0] < math.sqrt(0.08), hops are not allowed,
        # P[1] (value for second bead) doesn't matter.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertFalse(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        P = np.array([[math.sqrt(0.08), 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], atol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[-math.sqrt(0.08), 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        # (B)(c) adiabatic gradient rescaling - use Tully model C
        param["SurfaceHoppingElectrons"]["rescaling_type"] = "gradient"
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"
        param["TullyModel"]["model_type"] = "model_C"

        # (B)(c)(1) centroid rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "centroid"
        R = np.array([[-1., 1.]])
        p_ref = math.sqrt(8 * math.sqrt(0.01 + 3.6e-07))

        # From state 1 to 2. Here hop is not allowed for
        # -p_ref < P_centroid < p_ref
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[-1., 1.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertFalse(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[-1., 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        P = np.array([[2*p_ref - 1., 2*p_ref + 1.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[math.sqrt(3) * p_ref - 1,
                                        math.sqrt(3) * p_ref + 1]], atol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[p_ref - 1, p_ref + 1]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[math.sqrt(2) * p_ref - 1,
                                        math.sqrt(2) * p_ref + 1]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        P = np.array([[-1., 1.]])
        E_ref = nuclei_rpmd.energy_centroid.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[p_ref - 1, p_ref + 1]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy_centroid, E_ref)

        # (B)(c)(2) bead rescaling
        param["SurfaceHoppingElectrons"]["rpsh_rescaling"] = "bead"
        R = np.array([[0., -1.0e5]])

        # From state 1 to 2, hop is not allowed for
        # -p_hop < P[0] < p_hop where p_hop = math.sqrt(8. * (0.0006 + math.sqrt(0.01 + 3.6e-07)))
        # P[1] (value for second bead) doesn't really matter though.
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertFalse(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        P = np.array([[math.sqrt(8. * (0.0006 + math.sqrt(0.01 + 3.6e-07))),
                       1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 1))
        np.testing.assert_allclose(P, [[0., 1.]], atol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        # From state 2 to 1, always hops are allowed.
        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        nuclei_rpmd = nuclei.Nuclei(param, 0.0)

        P = np.array([[0., 1.]])
        E_ref = nuclei_rpmd.energy.copy()
        self.assertTrue(nuclei_rpmd.electrons._momentum_rescaling(R, P, 0))
        np.testing.assert_allclose(P, [[2 * math.sqrt(0.0012 + 2 * math.sqrt(0.01+3.6e-07)),
                                        1.]], rtol=1e-7)
        self.assertAlmostEqual(nuclei_rpmd.energy, E_ref)

        return

    def test_get_population(self):
        ### 1 bead case
        param = self.param_classical

        # Population in the same basis
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertEqual(pop, 1.0)

        # Population in different basis using Tully model A;
        # should give the same result for all rpsh types for 1 bead case.
        param["TullyModel"]["model_type"] = "model_A"

        # First in adiabatic basis.
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"
        param["SurfaceHoppingElectrons"]["rpsh_type"] = "bead"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        # This is done just to reset all pes quantities to required position value.
        # The transformation matrix here is U = [[0, 1], [1, 0]] for R=1.0e5.
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "centroid"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "density_matrix"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.0)

        # Then in diabatic basis.
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"
        param["SurfaceHoppingElectrons"]["rpsh_type"] = "bead"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        # This is done just to reset all pes quantities to required position value.
        # The transformation matrix here is U = [[0, 1], [1, 0]] for R=1.0e5.
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "centroid"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "density_matrix"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_classical = sh.SurfaceHoppingElectrons(param,
                                                            param.masses,
                                                            param.coordinates,
                                                            param.momenta)
        sh_electrons_classical.energy(np.array([[1.0e5]]))
        pop = sh_electrons_classical.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 1.0)
        pop = sh_electrons_classical.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.0)

        ### 2 bead case
        param = self.param_rpmd

        # Population in the same basis
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertEqual(pop, 1.0)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertEqual(pop, 0.0)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertEqual(pop, 0.0)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertEqual(pop, 1.0)

        # Population in different basis using Tully model A;
        # should give different results for different rpsh types in this case.
        param["TullyModel"]["model_type"] = "model_A"

        # First in adiabatic basis.
        param["SurfaceHoppingElectrons"]["basis"] = "adiabatic"
        param["SurfaceHoppingElectrons"]["rpsh_type"] = "bead"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        # This is done just to reset all pes quantities to required position value.
        # The transformation matrix for R=0 is U = 1/sqrt(2) * [[-1, 1], [1, 1]],
        # and at R=-1e5 it is U = [[1, 0], [0, 1]].
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.125 *  (3 - 2 * math.sqrt(2)))
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.125)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5,  0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.125)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.125 * (3 + 2 * math.sqrt(2)))

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "centroid"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 1.0e5]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.5)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.5)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 1.0e5]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.5)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.5)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "density_matrix"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.75)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.25)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "diabatic")
        self.assertAlmostEqual(pop, 0.25)
        pop = sh_electrons_rpmd.get_population(1, "diabatic")
        self.assertAlmostEqual(pop, 0.75)

        # Then in diabatic basis.
        param["SurfaceHoppingElectrons"]["basis"] = "diabatic"
        param["SurfaceHoppingElectrons"]["rpsh_type"] = "bead"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        # This is done just to reset all pes quantities to required position value.
        # The transformation matrix for R=0 is U = 1/sqrt(2) * [[1, -1], [1, 1]],
        # and at R=-1e5 it is U = [[1, 0], [0, 1]].
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.125 * (3 - 2 * math.sqrt(2)))
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.125)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5,  0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.125)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.125 * (3 + 2 * math.sqrt(2)))

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "centroid"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 1.0e5]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.5)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.5)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 1.0e5]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.5)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.5)

        param["SurfaceHoppingElectrons"]["rpsh_type"] = "density_matrix"
        param["SurfaceHoppingElectrons"]["initial_state"] = 0
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.75)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.25)

        param["SurfaceHoppingElectrons"]["initial_state"] = 1
        sh_electrons_rpmd = sh.SurfaceHoppingElectrons(param,
                                                       param.masses,
                                                       param.coordinates,
                                                       param.momenta)
        sh_electrons_rpmd.energy(np.array([[-1.0e5, 0.0]]))
        pop = sh_electrons_rpmd.get_population(0, "adiabatic")
        self.assertAlmostEqual(pop, 0.25)
        pop = sh_electrons_rpmd.get_population(1, "adiabatic")
        self.assertAlmostEqual(pop, 0.75)

        # TODO: add more tests with 3 state test using morse diabatic after
        #       correcting phase factor issue in 3 state basis transformation.
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(SurfaceHoppingTest)
    unittest.TextTestRunner().run(suite)
