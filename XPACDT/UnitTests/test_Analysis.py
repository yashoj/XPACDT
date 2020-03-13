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

import copy
import collections
import numpy as np
import random
import os
import shutil
import unittest
import warnings

import XPACDT.Tools.Analysis as analysis
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile
import XPACDT.Tools.Operations as op


class AnalysisTest(unittest.TestCase):

    def setUp(self):
        self.parameters = infile.Inputfile("FilesForTesting/SamplingTest/input_fixed.in")
        self.system = xSystem.System(self.parameters)

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        self.systems = []
        # Add 4 systems containing only log of current state to the empty list
        for i in range(4):
            shape = self.system.nuclei.positions.shape
            self.systems.append(copy.deepcopy(self.system))
            self.systems[-1].nuclei.positions = np.random.randn(*shape)
            self.systems[-1].nuclei.momenta = np.random.randn(*shape)
            self.systems[-1].do_log(init=True)

        # Add another nuclei to the log of the last system.
        self.systems[-1].nuclei.time = 2.0
        self.systems[-1].nuclei.positions = np.random.randn(*shape)
        self.systems[-1].nuclei.momenta = np.random.randn(*shape)
        self.systems[-1].do_log()

    @unittest.skip("Please implement as an integrated test.")
    def test_do_analysis(self):
        raise NotImplementedError("Please implement a test here.")

    def test_check_command(self):
        with self.assertRaises(ValueError):
            command = {'name': 'test', 'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1,2',
                       'step': '', 'format': 'time', 'results': [],
                       'all_operations': {'op0': op.Operations('+pos -1 0,1'),
                                          'op': op.Operations('+mom -1 0,1,2')}}
            analysis.check_command(command, self.systems[3])

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Generate warning
            command = {'name': 'test', 'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1',
                       'step': '', 'format': 'time', 'results': [],
                       'all_operations': {'op0': op.Operations('+pos -1 0,1'),
                                          'op': op.Operations('+mom -1 0,1')}}
            analysis.check_command(command, self.systems[3])
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)

    def test_apply_command(self):
        with self.assertRaises(ValueError):
            command = {'name': 'test', 'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1,2',
                       'step': '', 'format': 'time', 'results': [],
                       'all_operations': {'op0': op.Operations('+pos -1 0,1'),
                                          'op': op.Operations('+mom -1 0,1,2')}}
            analysis.apply_command(command, self.systems[3], [])

        command = {'op': '+pos -1 0 +mom -1 0', 'step': '', 'results': [],
                   'all_operations': {'op': op.Operations('+pos -1 0 +mom -1 0')}}
        analysis.apply_command(command, self.systems[0], [])
        results_ref = np.array([[0.49671415 * -0.23415337]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)

        command = {'op': '+pos -1 0 +mom -1 0', 'step': '1', 'results': [],
                   'all_operations': {'op': op.Operations('+pos -1 0 +mom -1 0')}}
        analysis.apply_command(command, self.systems[0], [1])
        results_ref = np.array([])
        np.testing.assert_array_almost_equal(command['results'], results_ref)

        command = {'op0': '+pos -1 0 ', 'op': '+mom -1 0', 'step': '', 'results': [],
                   'all_operations': {'op0': op.Operations('+pos -1 0'),
                                      'op': op.Operations('+mom -1 0')}}
        analysis.apply_command(command, self.systems[3], [])
        results_ref = np.array([[-0.54438272*-0.60063869], [-0.54438272*0.2088636]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)

        command = {'op0': '+pos -1 0 ', 'op': '+mom -1 0', 'step': '1', 'results': [],
                   'all_operations': {'op0': op.Operations('+pos -1 0'),
                                      'op': op.Operations('+mom -1 0')}}
        analysis.apply_command(command, self.systems[3], [1])
        results_ref = np.array([[-0.54438272*0.2088636]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)

        command2d = {'op': '+pos -1 0', 'step': '', '2d': None, '2op': '+mom -1 0', 'results': [],
                     'all_operations': {'op': op.Operations('+pos -1 0'),
                                        '2op': op.Operations('+mom -1 0')}}
        analysis.apply_command(command2d, self.systems[3], [])
        results_ref = np.array([[-0.54438272], [-0.01349722], [-0.60063869], [0.2088636]])
        np.testing.assert_array_almost_equal(command2d['results'], results_ref)

        return

    def test_output_data(self):
        header = "This is a stupid header\nFor my super test!"
        output_file = "test.dat"
        form = 'time'
        times = np.array([0.0, 2.0])
        bins = None
        results = np.array([[-5.0], [2.0]])

        compare_text = "# This is a stupid header\n# For my super test!\n 0.00000000e+00 -5.00000000e+00\n 2.00000000e+00  2.00000000e+00\n"

        analysis.output_data(header, output_file, form, times, bins, results)
        text = ''
        with open(output_file, 'r') as infile:
            for line in infile:
                text += line

        os.remove("test.dat")
        self.assertEqual(text, compare_text)

        header = "This is a stupid header\nFor my super test!\n"
        output_file = "test.dat"
        form = 'value'
        times = np.array([0.0, 2.0])
        bins = [np.array([-2.0, -1.0, 0.0, 1.0, 2.0])]
        results = [np.array([[-5.0, 2.0, 3.0, 2.0, 1.5], [5.0, -2.0, -3.0, -2.0, -1.5]])]

        compare_text = "# This is a stupid header\n" + \
        "# For my super test!\n" + \
        "#  0.00000000e+00 \t  0.00000000e+00 \t 2.00000000e+00 \t  2.00000000e+00 \t \n" + \
        "# \n-2.00000000e+00 -5.00000000e+00  5.00000000e+00\n" + \
        "-1.00000000e+00  2.00000000e+00 -2.00000000e+00\n" + \
        " 0.00000000e+00  3.00000000e+00 -3.00000000e+00\n" + \
        " 1.00000000e+00  2.00000000e+00 -2.00000000e+00\n" + \
        " 2.00000000e+00  1.50000000e+00 -1.50000000e+00\n"

        analysis.output_data(header, output_file, form, times, bins, results)
        text = ''
        with open(output_file, 'r') as infile:
            for line in infile:
                text += line

        os.remove("test.dat")
        os.remove("test.plt")
        self.assertEqual(text, compare_text)

        header = "# This is a stupid header\n# For my super test!\n"
        output_file = "test.dat"
        form = '2d'
        times = np.array([0.0, 2.0])
        bins = [np.array([-2.0, -1.0, 0.0, 1.0, 2.0])]
        results = [np.array([[-5.0, 2.0, 3.0, 2.0, 1.5], [5.0, -2.0, -3.0, -2.0, -1.5]])]

        compare_text = "# This is a stupid header\n" + \
        "# For my super test!\n" + \
        " 0.00000000e+00 -2.00000000e+00 -5.00000000e+00 \n" + \
        " 0.00000000e+00 -1.00000000e+00  2.00000000e+00 \n" + \
        " 0.00000000e+00  0.00000000e+00  3.00000000e+00 \n" + \
        " 0.00000000e+00  1.00000000e+00  2.00000000e+00 \n" + \
        " 0.00000000e+00  2.00000000e+00  1.50000000e+00 \n" + \
        "\n" + \
        " 2.00000000e+00 -2.00000000e+00  5.00000000e+00 \n" + \
        " 2.00000000e+00 -1.00000000e+00 -2.00000000e+00 \n" + \
        " 2.00000000e+00  0.00000000e+00 -3.00000000e+00 \n" + \
        " 2.00000000e+00  1.00000000e+00 -2.00000000e+00 \n" + \
        " 2.00000000e+00  2.00000000e+00 -1.50000000e+00 \n\n"

        analysis.output_data(header, output_file, form, times, bins, results)
        text = ''
        with open(output_file, 'r') as infile:
            for line in infile:
                text += line

        os.remove("test.dat")
        os.remove("test.plt")
        self.assertEqual(text, compare_text)

    def test_get_step_list(self):
        command = {}
        steps = analysis._get_step_list(command)
        self.assertEqual(steps, [])

        command = {'step': 'index 1 2 3'}
        steps = analysis._get_step_list(command)
        self.assertEqual(steps, [1, 2, 3])

        # Checking if 'last' option works; self.systems[0] should have only one
        # log nuclei and self.systems[3] should have two.
        command = {'step': 'last'}
        steps = analysis._get_step_list(command, self.systems[0])
        self.assertEqual(steps, [0])

        steps = analysis._get_step_list(command, self.systems[3])
        self.assertEqual(steps, [1])

        command = {'name': 'test', 'step': 'random'}
        with self.assertRaises(ValueError):
            steps = analysis._get_step_list(command)

        return

    def test_use_time(self):
        for i in range(0, 100):
            self.assertTrue(analysis._use_time(i, []))
        for i in range(0, 100):
            if i in [1, 5, 6, 10]:
                self.assertTrue(analysis._use_time(i, [1, 5, 6, 10]))
            else:
                self.assertFalse(analysis._use_time(i, [1, 5, 6, 10]))

    def test_get_directory_list(self):
        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        dir_list_ref2 = ["./trj_0", "./trj_2"]
        for d in dir_list_ref:
            os.mkdir(d)

        for d in dir_list_ref2:
            open(d + "/pickle.dat", 'a').close()

        dir_list = analysis.get_directory_list()
        self.assertSequenceEqual(dir_list, dir_list_ref)

        dir_list = analysis.get_directory_list(file_name='pickle.dat')
        self.assertSequenceEqual(dir_list, dir_list_ref2)

        for d in dir_list_ref:
            shutil.rmtree(d)

    def test_get_systems(self):
        with self.assertRaises(RuntimeError):
            analysis.get_systems(None, None, None)

        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        dir_list_ref2 = ["./trj_0", "./trj_2"]
        for d in dir_list_ref:
            os.mkdir(d)

        for d in dir_list_ref2:
            open(d + "/pickle.dat", 'a').close()

        sys = analysis.get_systems(dir_list_ref, 'pickle.dat', None)
        self.assertTrue(isinstance(sys, collections.Iterable))

        for d in dir_list_ref:
            shutil.rmtree(d)

    def tearDown(self):
        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        for d in dir_list_ref:
            if os.path.isdir(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(AnalysisTest)
    unittest.TextTestRunner().run(suite)
