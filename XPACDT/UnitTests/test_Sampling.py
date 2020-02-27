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
import shutil
import unittest

import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile
import XPACDT.Sampling.Sampling as sampling


class SamplingTest(unittest.TestCase):

    def setUp(self):
        self.parameters_position_shift = infile.Inputfile("FilesForTesting/SamplingTest/input_position_shift.in")
        self.system_position = xSystem.System(self.parameters_position_shift)

        self.parameters_momentum_shift = infile.Inputfile("FilesForTesting/SamplingTest/input_momentum_shift.in")
        self.system_momentum = xSystem.System(self.parameters_momentum_shift)

        self.parameters_position_shift_xyz = infile.Inputfile("FilesForTesting/SamplingTest/input_position_shift_xyz.in")
        self.system_position_xyz = xSystem.System(self.parameters_position_shift_xyz)

        self.parameters_momentum_shift_xyz = infile.Inputfile("FilesForTesting/SamplingTest/input_momentum_shift_xyz.in")
        self.system_momentum_xyz = xSystem.System(self.parameters_momentum_shift_xyz)

    @unittest.skip("Please implement a test here.")
    def test_sample(self):
        raise NotImplementedError("Please implement a test here.")

    def test_shifts(self):
        coordinate_ref = np.array([[2.0], [1.0], [4.0], [-2.0]])
        momenta_ref = np.array([[-1.0], [0.1], [2.0], [1.25]])
        shift_ref = np.array([[1.0], [1.0], [-1.0], [-1.0]])

        sampled_systems = sampling.sample(self.system_position, self.parameters_position_shift, True)

        self.assertEqual(len(sampled_systems), 1000)
        for s in sampled_systems:
            self.assertEqual(s.nuclei.n_dof, 4)
            np.testing.assert_allclose(s.nuclei.positions, coordinate_ref + shift_ref,
                                       rtol=1e-7)
            np.testing.assert_allclose(s.nuclei.momenta, momenta_ref,
                                       rtol=1e-7)
        os.rmdir('test')

        sampled_systems = sampling.sample(self.system_momentum, self.parameters_momentum_shift, True)

        self.assertEqual(len(sampled_systems), 1000)
        for s in sampled_systems:
            self.assertEqual(s.nuclei.n_dof, 4)
            np.testing.assert_allclose(s.nuclei.positions, coordinate_ref,
                                       rtol=1e-7)
            np.testing.assert_allclose(s.nuclei.momenta, momenta_ref + shift_ref,
                                       rtol=1e-7)
        os.rmdir('test')

        coordinate_ref = np.array([[0.0], [0.0], [0.0], [2.0], [0.0], [0.0]])
        momenta_ref = np.array([[-0.1], [0.0], [1.0], [0.0], [0.0], [1.0]])
        shift_ref = np.array([[2.0], [-1.0], [0.0], [-1.0], [2.0], [0.0]])

        sampled_systems = sampling.sample(self.system_position_xyz, self.parameters_position_shift_xyz, True)

        self.assertEqual(len(sampled_systems), 1000)
        for s in sampled_systems:
            self.assertEqual(s.nuclei.n_dof, 6)
            np.testing.assert_allclose(s.nuclei.positions, coordinate_ref + shift_ref,
                                       rtol=1e-7)
            np.testing.assert_allclose(s.nuclei.momenta, momenta_ref,
                                       rtol=1e-7)
        os.rmdir('test')

        sampled_systems = sampling.sample(self.system_momentum_xyz, self.parameters_momentum_shift_xyz, True)

        self.assertEqual(len(sampled_systems), 1000)
        for s in sampled_systems:
            self.assertEqual(s.nuclei.n_dof, 6)
            np.testing.assert_allclose(s.nuclei.positions, coordinate_ref,
                                       rtol=1e-7)
            np.testing.assert_allclose(s.nuclei.momenta, momenta_ref + shift_ref,
                                       rtol=1e-7)

        os.rmdir('test')

    def tearDown(self):
        if os.path.isdir('test'):
            shutil.rmtree('test')

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(SamplingTest)
    unittest.TextTestRunner().run(suite)
