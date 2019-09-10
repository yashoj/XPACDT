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


class AnalysisTest(unittest.TestCase):

    def setUp(self):
        self.parameters = infile.Inputfile("FilesForTesting/SamplingTest/input_fixed.in")
        self.system = xSystem.System(self.parameters)

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        self.systems = []
        for i in range(4):
            shape = self.system.nuclei.positions.shape
            self.systems.append(copy.deepcopy(self.system))
            self.systems[-1].nuclei.positions = np.random.randn(*shape)
            self.systems[-1].nuclei.momenta = np.random.randn(*shape)
            self.systems[-1].do_log(init=True)

        self.systems[-1].nuclei.time = 2.0
        self.systems[-1].nuclei.positions = np.random.randn(*shape)
        self.systems[-1].nuclei.momenta = np.random.randn(*shape)
        self.systems[-1].do_log()

#    def test_do_analysis(self):
#        # TODO: Implement a more integrated test here!
#        raise NotImplementedError("Please implement a test here!!")
#        pass
#

    def test_check_command(self):
        with self.assertRaises(ValueError):
            command = {'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1,2', 'step': '', 'format': 'time', 'results': []}
            analysis.check_command(command, self.systems[3])

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Generate warning
            command = {'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1', 'step': '', 'format': 'time', 'results': []}
            analysis.check_command(command, self.systems[3])
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)

    def test_apply_command(self):
        with self.assertRaises(ValueError):
            command = {'op0': '+pos -1 0,1 ', 'op': '+mom -1 0,1,2', 'step': '', 'format': 'time', 'results': []}
            analysis.apply_command(command, self.systems[3])

        command = {'op': '+pos -1 0 +mom -1 0', 'step': '', 'results': []}
        analysis.apply_command(command, self.systems[0])
        results_ref = np.array([[0.49671415 * -0.23415337]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)
        times_ref = np.array([0.0])
        np.testing.assert_array_almost_equal(command['times'], times_ref)

        command = {'op': '+pos -1 0 +mom -1 0', 'step': '1', 'results': []}
        analysis.apply_command(command, self.systems[0])
        results_ref = np.array([])
        np.testing.assert_array_almost_equal(command['results'], results_ref)
        times_ref = np.array([])
        np.testing.assert_array_almost_equal(command['times'], times_ref)

        command = {'op0': '+pos -1 0 ', 'op': '+mom -1 0', 'step': '', 'results': []}
        analysis.apply_command(command, self.systems[3])
        results_ref = np.array([[-0.54438272*-0.60063869], [-0.54438272*0.2088636]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)
        times_ref = np.array([0.0, 2.0])
        np.testing.assert_array_almost_equal(command['times'], times_ref)

        command = {'op0': '+pos -1 0 ', 'op': '+mom -1 0', 'step': '1', 'results': []}
        analysis.apply_command(command, self.systems[3])
        results_ref = np.array([[-0.54438272*0.2088636]])
        np.testing.assert_array_almost_equal(command['results'], results_ref)
        times_ref = np.array([2.0])
        np.testing.assert_array_almost_equal(command['times'], times_ref)

        command2d = {'op': '+pos -1 0', 'step': '', '2d': None, '2op': '+mom -1 0', 'results': []}
        analysis.apply_command(command2d, self.systems[3])
        results_ref = np.array([[-0.54438272], [-0.01349722], [-0.60063869], [0.2088636]])
        np.testing.assert_array_almost_equal(command2d['results'], results_ref)
        times_ref = np.array([0.0, 2.0])
        np.testing.assert_array_almost_equal(command2d['times'], times_ref)

        pass

    def test_apply_operation(self):
        with self.assertRaises(RuntimeError):
            operation = "+notImplemented"
            value = analysis.apply_operation(operation, self.systems[0].nuclei)

        with self.assertRaises(RuntimeError):
            operation = "No Operation given here!"
            value = analysis.apply_operation(operation, self.systems[0].nuclei)

        operation = "+id"
        value_ref = 1.0
        value = analysis.apply_operation(operation, self.systems[0].nuclei)
        np.testing.assert_equal(value, value_ref)

        operation = "+identity"
        value_ref = 1.0
        value = analysis.apply_operation(operation, self.systems[0].nuclei)
        np.testing.assert_equal(value, value_ref)

        operation = "+pos -1 1,2"
        value_ref = np.array([-0.1382643, 0.64768854])
        value = analysis.apply_operation(operation, self.systems[0].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

        operation = "+position -1 0"
        value_ref = np.array([0.49671415])
        value = analysis.apply_operation(operation, self.systems[0].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

        operation = "+mom -1 1,2"
        value_ref = np.array([-1.91328024, -1.72491783])
        value = analysis.apply_operation(operation, self.systems[1].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

        operation = "+momentum -1 0"
        value_ref = np.array([0.24196227])
        value = analysis.apply_operation(operation, self.systems[1].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

        operation = "+vel -1 1,2"
        value_ref = np.array([-0.2257763 / 2.0, 0.0675282 / 12.0])
        value = analysis.apply_operation(operation, self.systems[2].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

        operation = "+velocity -1 0"
        value_ref = np.array([1.46564877])
        value = analysis.apply_operation(operation, self.systems[2].nuclei)
        np.testing.assert_array_almost_equal(value, value_ref)

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
        self.assertEqual(text, compare_text)

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

# TODO: make sure these are always removed!
        for d in dir_list_ref:
            shutil.rmtree(d)

    def test_get_systems(self):
        with self.assertRaises(RuntimeError):
            analysis.get_systems(None, None, None)

        # TODO generate a list of systems and store them to be read by pickle etc.
        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        dir_list_ref2 = ["./trj_0", "./trj_2"]
        for d in dir_list_ref:
            os.mkdir(d)

        for d in dir_list_ref2:
            open(d + "/pickle.dat", 'a').close()

        sys = analysis.get_systems(dir_list_ref, 'pickle.dat', None)
        self.assertTrue(isinstance(sys, collections.Iterable))

# TODO: make sure these are always removed!
        for d in dir_list_ref:
            shutil.rmtree(d)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(AnalysisTest)
    unittest.TextTestRunner().run(suite)
