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

import XPACDT.Sampling.QuasiclassicalSampling as qcs
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile

# TODO: How to properly test? Use some seeding for random number generator
# and obtain actual values to compare to?
# TODO: Use other PES once available!


class QuasiclassicalSamplingTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.parameters0 = infile.Inputfile("FilesForTesting/SamplingTest/input_quasiclassical_0.in")
        self.system0 = xSystem.System(self.parameters0)

        self.parameters2 = infile.Inputfile("FilesForTesting/SamplingTest/input_quasiclassical_2.in")
        self.system2 = xSystem.System(self.parameters2)

    def test_do_Quasiclassical_sampling(self):
        samples = qcs.do_Quasiclassical_sampling(self.system0, self.parameters0)

        self.assertEqual(len(samples), 1000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 1)
            self.assertAlmostEqual(0.5, s.nuclei.energy)

        samples = qcs.do_Quasiclassical_sampling(self.system2, self.parameters2)

        self.assertEqual(len(samples), 1000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 1)
            self.assertAlmostEqual(2.5, s.nuclei.energy)


if __name__ == "__main__":
    suite = unittest.TestLoader().\
        loadTestsFromTestCase(QuasiclassicalSamplingTest)
    unittest.TextTestRunner().run(suite)
