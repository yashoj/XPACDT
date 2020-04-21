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

import collections
import copy
import os
import numpy as np
import shutil
import random
import unittest

import XPACDT.Tools.Xtools as xtools
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile


class XtoolsTest(unittest.TestCase):

    def setUp(self):
        parameters = infile.Inputfile("FilesForTesting/OperationsTest/input_classicalNuclei.in")
        self.system = xSystem.System(parameters)

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

    def test_get_directory_list(self):
        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        dir_list_ref2 = ["./trj_0", "./trj_2"]
        for d in dir_list_ref:
            os.mkdir(d)

        for d in dir_list_ref2:
            open(d + "/pickle.dat", 'a').close()

        dir_list = xtools.get_directory_list()
        self.assertSequenceEqual(dir_list, dir_list_ref)

        dir_list = xtools.get_directory_list(file_name='pickle.dat')
        self.assertSequenceEqual(dir_list, dir_list_ref2)

        for d in dir_list_ref:
            shutil.rmtree(d)

    def test_get_systems(self):
        with self.assertRaises(RuntimeError):
            xtools.get_systems(None, None, None)

        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        dir_list_ref2 = ["./trj_0", "./trj_2"]
        for d in dir_list_ref:
            os.mkdir(d)

        for d in dir_list_ref2:
            open(d + "/pickle.dat", 'a').close()

        sys = xtools.get_systems(dir_list_ref, 'pickle.dat', None)
        self.assertTrue(isinstance(sys, collections.Iterable))

        for d in dir_list_ref:
            shutil.rmtree(d)

    def tearDown(self):
        dir_list_ref = ["./trj_" + str(i) for i in range(len(self.systems))]
        for d in dir_list_ref:
            if os.path.isdir(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(XtoolsTest)
    unittest.TextTestRunner().run(suite)
