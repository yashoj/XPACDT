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

import math
import numpy as np
import random
import unittest

import XPACDT.System.SurfaceHoppingElectrons as sh
import XPACDT.Input.Inputfile as infile


class SurfaceHoppingTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        
        parameters = infile.Inputfile("FilesForTesting/SystemTests/input_SH_classical.in")
        self.sh_electrons_classical = sh.SurfaceHoppingElectrons(parameters, parameters.n_beads,
                                                                 parameters.masses, parameters.coordinates, parameters.momenta)

        parameters = infile.Inputfile("FilesForTesting/SystemTests/input_SH_rpmd.in")
        self.sh_electrons_rpmd = sh.SurfaceHoppingElectrons(parameters, parameters.n_beads,
                                                            parameters.masses, parameters.coordinates, parameters.momenta)

    def test_creation(self):
        self.assertEqual(self.sh_electrons_classical.name, 'SurfaceHoppingElectrons')
        self.assertEqual(self.sh_electrons_classical.basis, 'adiabatic')
        self.assertEqual(self.sh_electrons_classical.current_state, 0)
        self.assertAlmostEqual(self.sh_electrons_classical.timestep, 0.01)
        self.assertEqual(self.sh_electrons_classical.rpsh_type, 'bead')
        self.assertEqual(self.sh_electrons_classical.rpsh_rescaling, 'bead')
        self.assertEqual(self.sh_electrons_classical.rescaling_type, 'nac')
        self.assertEqual(self.sh_electrons_classical.evolution_picture, 'schroedinger')
        self.assertEqual(self.sh_electrons_classical.ode_solver, 'rk4')
        return

    def test_energy(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # Adiabatic basis
        R = np.array([[0.]])

        self.sh_electrons_classical.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=False),
                                   [-math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=True),
                                   -math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        self.sh_electrons_classical.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=False),
                                   [math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=True),
                                   math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_classical.basis = 'diabatic'
        self.sh_electrons_classical.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=False),
                                   [0.0006], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=True),
                                   0.0006, rtol=1e-7)

        self.sh_electrons_classical.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=False),
                                   [-0.0006], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.energy(R, centroid=True),
                                   -0.0006, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        # Adiabatic basis
        R = np.array([[-1.0e05, 1.0e05]])

        self.sh_electrons_rpmd.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=False),
                                   [-0.0006, -math.sqrt(0.04 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=True),
                                   -math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        self.sh_electrons_rpmd.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=False),
                                   [0.0006, math.sqrt(0.04 + 3.6e-07)], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=True),
                                   math.sqrt(0.01 + 3.6e-07), rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'
        self.sh_electrons_rpmd.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=False),
                                   [0.0006, 0.0006], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=True),
                                   0.0006, rtol=1e-7)

        self.sh_electrons_rpmd.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=False),
                                   [-0.0006, -0.0006], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.energy(R, centroid=True),
                                   -0.0006, rtol=1e-7)

        return

    def test_gradient(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # Adiabatic basis
        R = np.array([[0.]])

        self.sh_electrons_classical.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=False),
                                   [[-0.009 / math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=True),
                                   [-0.009 / math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)

        self.sh_electrons_classical.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.009 / math.sqrt(0.01 + 3.6e-07)]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=True),
                                   [0.009 / math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_classical.basis = 'diabatic'
        self.sh_electrons_classical.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        self.sh_electrons_classical.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=False),
                                   [[0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_classical.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        # Adiabatic basis
        R = np.array([[-1.0e05, 1.0e05]])

        self.sh_electrons_rpmd.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=True),
                                   [-0.009 / math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)

        self.sh_electrons_rpmd.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.009 / math.sqrt(0.01 + 3.6e-07)], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'
        self.sh_electrons_rpmd.current_state = 0
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        self.sh_electrons_rpmd.current_state = 1
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=False),
                                   [[0.0, 0.0]], rtol=1e-7)
        np.testing.assert_allclose(self.sh_electrons_rpmd.gradient(R, centroid=True),
                                   [0.0], rtol=1e-7)

        return

    def test_get_velocity(self):
        # 1 bead, 1 dof test
        P = np.array([[2.0]])
        np.testing.assert_allclose(self.sh_electrons_classical._get_velocity(P),
                                   [[1.0]], rtol=1e-7)
        p_centroid = np.array([2.0])
        np.testing.assert_allclose(self.sh_electrons_classical._get_velocity(p_centroid),
                                   [1.0], rtol=1e-7)

        # 2 beads, 1 dof test
        P = np.array([[2.0, 1.0]])
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_velocity(P),
                                   [[1.0, 0.5]], rtol=1e-7)

        # TODO: add test for more than 1 dof once higher dimensional potentials are available
        return

    def test_get_modified_V(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])

        # Adiabatic basis
        V_ref = np.array([[[-math.sqrt(0.01 + 3.6e-07), 0.],
                           [0., math.sqrt(0.01 + 3.6e-07)]]])
        self.sh_electrons_classical.basis = 'adiabatic'
        self.sh_electrons_classical.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        # Diabatic basis
        V_ref = np.array([[[0.0006, 0.1], [0.1, -0.0006]]])
        self.sh_electrons_classical.basis = 'diabatic'
        self.sh_electrons_classical.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_classical._get_modified_V(R),
                                   V_ref, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])

        # Adibatic basis
        self.sh_electrons_rpmd.basis = 'adiabatic'
        self.sh_electrons_rpmd.rpsh_type = 'bead'
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[-V_ad_ref, 0.],
                                     [0., V_ad_ref]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[-math.sqrt(0.01 + 3.6e-07), 0.],
                                     [0., math.sqrt(0.01 + 3.6e-07)]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[-0.0006, 0.], [0., 0.0006]],
                                    [[-math.sqrt(0.04 + 3.6e-07), 0.],
                                     [0., math.sqrt(0.04 + 3.6e-07)]]], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'
        self.sh_electrons_rpmd.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.1],
                                     [0.1, -0.0006]]], rtol=1e-7)
        
        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.1],
                                     [0.1, -0.0006]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_modified_V(R),
                                   [[[0.0006, 0.0], [0.0, -0.0006]],
                                    [[0.0006, 0.2], [0.2, -0.0006]]], rtol=1e-7)

        return

    def test_get_kinetic_coupling_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        P = np.array([[4.]])
        nac_ref = -2.7e-05/(0.01 + 3.6e-07)

        # Adiabatic basis
        D_ref = np.array([[[0., 2.0 * nac_ref], [-2.0 * nac_ref, 0.]]])
        self.sh_electrons_classical.basis = 'adiabatic'
        self.sh_electrons_classical.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P),
                                   D_ref, rtol=1e-7)

        # Diabatic basis - should just return None for all cases.
        self.sh_electrons_classical.basis = 'diabatic'
        self.sh_electrons_classical.rpsh_type = 'bead'
        self.assertIsNone(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        self.sh_electrons_classical.rpsh_type = 'centroid'
        self.assertIsNone(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        self.assertIsNone(self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P))

        ### test for 2 state potential, 1 dof, 2 beads
        # Adibatic basis
        R = np.array([[-1.0e05, 1.0e05]])
        P = np.array([[2., 6.]])

        self.sh_electrons_rpmd.basis = 'adiabatic'
        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., 2.0 * nac_ref],
                                     [-2.0 * nac_ref, 0.]]], rtol=1e-7)

        R = np.array([[0., 1.0e05]])
        P = np.array([[4., 2.]])
        self.sh_electrons_rpmd.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., nac_ref],
                                     [-nac_ref, 0.]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P),
                                   [[[0., 2.0 * nac_ref], [-2.0 * nac_ref, 0.]],
                                    [[0., 0.], [0., 0.]]], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'
        self.assertIsNone(self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P))

        # TODO: add test for more than 1 dof once higher dimensional potentials are available

        return

    def test_get_H_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        R = np.array([[0.]])
        P = np.array([[4.]])
        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        D_ref = -2. * 2.7e-05/(0.01 + 3.6e-07)

        with self.assertRaises(AssertionError):
            self.sh_electrons_classical._get_H_matrix(R)

        # Adiabatic basis
        self.sh_electrons_classical.basis = 'adiabatic'
        H_ref_schroedinger = np.array([[[-V_ad_ref + 0.0j, -1j * D_ref],
                                        [1j * D_ref, V_ad_ref + 0.0j]]])
        H_ref_interaction = np.array([[[0.0 + 0.0j, -1j * D_ref],
                                       [1j * D_ref, 0.0 + 0.0j]]])


        self.sh_electrons_classical.rpsh_type = 'bead'
        D = self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        D = self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        D = self.sh_electrons_classical._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R, D),
                                   H_ref_interaction, rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_classical.basis = 'diabatic'
        H_ref_schroedinger = np.array([[[0.0006 + 0.0j, 0.1 + 0.0j],
                                        [0.1 + 0.0j, -0.0006 + 0.0j]]])
        H_ref_interaction = np.array([[[0.0 + 0.0j, 0.1 + 0.0j],
                                       [0.1 + 0.0j, 0.0 + 0.0j]]])

        self.sh_electrons_classical.rpsh_type = 'bead'
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_schroedinger, rtol=1e-7)
        self.sh_electrons_classical.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_classical._get_H_matrix(R),
                                   H_ref_interaction, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])
        P = np.array([[2., 6.]])

        # Adibatic basis
        self.sh_electrons_rpmd.basis = 'adiabatic'

        self.sh_electrons_rpmd.rpsh_type = 'bead'
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        D = self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-V_ad_ref + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, V_ad_ref + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        D_ref = -2. * 2.7e-05/(0.01 + 3.6e-07)
        D = self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-V_ad_ref + 0.0j, -1j * D_ref],
                                     [1j * D_ref, V_ad_ref + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, -1j * D_ref],
                                     [1j * D_ref, 0.0 + 0.0j]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        D = self.sh_electrons_rpmd._get_kinetic_coupling_matrix(R, P)
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[-0.0006 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0006 + 0.0j]],
                                    [[-math.sqrt(0.04 + 3.6e-07) + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, math.sqrt(0.04 + 3.6e-07) + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R, D),
                                   [[[0.0 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0 + 0.0j]],
                                    [[0.0 + 0.0j, 0. + 0.0j],
                                     [0. + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'

        self.sh_electrons_rpmd.rpsh_type = 'bead'
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)
        
        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.1 + 0.0j],
                                     [0.1 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        self.sh_electrons_rpmd.evolution_picture = 'schroedinger'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0006 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, -0.0006 + 0.0j]],
                                    [[0.0006 + 0.0j, 0.2 + 0.0j],
                                     [0.2 + 0.0j, -0.0006 + 0.0j]]], rtol=1e-7)
        self.sh_electrons_rpmd.evolution_picture = 'interaction'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_H_matrix(R),
                                   [[[0.0 + 0.0j, 0.0 + 0.0j],
                                     [0.0 + 0.0j, 0.0 + 0.0j]],
                                    [[0.0 + 0.0j, 0.2 + 0.0j],
                                     [0.2 + 0.0j, 0.0 + 0.0j]]], rtol=1e-7)
        return

    def test_get_diff_diag_V_matrix(self):
        ### test for 2 state potential, 1 dof, 1 bead
        # All rpsh_type should give the same result for 1 bead case.
        
        R = np.array([[0.]])
        
        self.sh_electrons_classical.evolution_picture = 'schroedinger'
        with self.assertRaises(AssertionError):
            self.sh_electrons_classical._get_diff_diag_V_matrix(R)
            
        self.sh_electrons_classical.evolution_picture = 'interaction'

        # Adiabatic basis
        self.sh_electrons_classical.basis = 'adiabatic'
        V_ad_ref = math.sqrt(0.01 + 3.6e-07)
        diff_ref = np.array([[[0., 2. * V_ad_ref],
                              [-2. * V_ad_ref, 0.]]])

        self.sh_electrons_classical.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   diff_ref, rtol=1e-7)
        
        # Diabatic basis
        V_ref = np.array([[[0., -0.0012], [0.0012, 0.]]])
        self.sh_electrons_classical.basis = 'diabatic'
        self.sh_electrons_classical.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        self.sh_electrons_classical.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_classical._get_diff_diag_V_matrix(R),
                                   V_ref, rtol=1e-7)

        ### test for 2 state potential, 1 dof, 2 beads
        R = np.array([[-1.0e05, 1.0e05]])
        self.sh_electrons_rpmd.evolution_picture = 'interaction'

        # Adibatic basis
        self.sh_electrons_rpmd.basis = 'adiabatic'
        self.sh_electrons_rpmd.rpsh_type = 'bead'
        V_ad_ref = (math.sqrt(0.04 + 3.6e-07) + 0.0006) * 0.5
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 2. * V_ad_ref],
                                     [-2. * V_ad_ref, 0.]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 2. * math.sqrt(0.01 + 3.6e-07)],
                                     [-2. * math.sqrt(0.01 + 3.6e-07), 0.]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., 0.0012], [-0.0012, 0.]],
                                    [[0., 2. * math.sqrt(0.04 + 3.6e-07)],
                                     [-2. * math.sqrt(0.04 + 3.6e-07), 0.]]], rtol=1e-7)

        # Diabatic basis
        self.sh_electrons_rpmd.basis = 'diabatic'
        self.sh_electrons_rpmd.rpsh_type = 'bead'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)
        
        self.sh_electrons_rpmd.rpsh_type = 'centroid'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)

        self.sh_electrons_rpmd.rpsh_type = 'density_matrix'
        np.testing.assert_allclose(self.sh_electrons_rpmd._get_diff_diag_V_matrix(R),
                                   [[[0., -0.0012], [0.0012, 0.]],
                                    [[0., -0.0012], [0.0012, 0.]]], rtol=1e-7)

        return

    def test_step(self):
        return

    def test_get_b_jk(self):
        return

    def test_surface_hopping(self):
        return

    def test_momentum_rescaling(self):
        return

    def test_linear_interpolation_1d(self):
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(SurfaceHoppingTest)
    unittest.TextTestRunner().run(suite)
