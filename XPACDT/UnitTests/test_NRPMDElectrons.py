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
import random
import unittest

import XPACDT.System.NRPMDElectrons as nrpmd
import XPACDT.Input.Inputfile as infile


class NRPMDElectronsTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.parameters = infile.Inputfile("FilesForTesting/SystemTest/input_NRPMD_classical.in")
        # self.system = xSystem.System(self.parameters0)

    @unittest.skip("Please implement a test here.")
    def test_step(self):
        return

    @unittest.skip("Please implement a test here.")
    def test_energy(self):
        return

    @unittest.skip("Please implement a test here.")
    def test_gradient(self):
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NRPMDElectronsTest)
    unittest.TextTestRunner().run(suite)
