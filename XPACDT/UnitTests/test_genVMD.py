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

import os
import unittest

import XPACDT.Bin.genVMD as gVMD
import XPACDT.Input.Inputfile as infile
import XPACDT.System.System as xsys


class xpacdtTest(unittest.TestCase):

    def test_gen_XYZ(self):
        parameters_anharmonic_4_nb = infile.Inputfile("FilesForTesting/SystemTests/anharmonic_4.in")
        system = xsys.System(parameters_anharmonic_4_nb)
        system.step(1, True)

        gVMD.gen_XYZ(system, ".")

        with open("FilesForTesting/BinTest/centroids_reference.txt", "r") as myfile:
            centroids_reference = myfile.readlines()

        with open("./centroids.xyz", "r") as myfile:
            centroids = myfile.readlines()

        self.assertEqual(centroids_reference, centroids)

        with open("FilesForTesting/BinTest/beads_reference.txt", "r") as myfile:
            beads_reference = myfile.readlines()

        with open("./beads.xyz", "r") as myfile:
            beads = myfile.readlines()

        self.assertEqual(beads_reference, beads)

        os.remove("centroids.xyz")
        os.remove("beads.xyz")

    def test_gen_VMD(self):
        gVMD.gen_VMD(".")

        with open("FilesForTesting/BinTest/vmd_reference.txt", "r") as myfile:
            vmd_reference = myfile.readlines()

        with open("./movie.vmd", "r") as myfile:
            vmd = myfile.readlines()

        self.assertEqual(vmd_reference, vmd)

        with open("FilesForTesting/BinTest/bash_vmd_reference.txt", "r") as myfile:
            bash_reference = myfile.readlines()

        with open("./run.sh", "r") as myfile:
            bash = myfile.readlines()

        self.assertEqual(bash_reference, bash)

        os.remove("movie.vmd")
        os.remove("run.sh")

    @unittest.skip("How to test here?")
    def test_run_VMD(self):
        raise NotImplementedError("Please implement a test here!!")
        pass

    @unittest.skip("How to test here?")
    def test_start(self):
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(xpacdtTest)
    unittest.TextTestRunner().run(suite)
