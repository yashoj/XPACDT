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
import random
import scipy.stats
import unittest

import XPACDT.System.System as xSystem
import XPACDT.Sampling.ThermostattedSampling as thermo
import XPACDT.Input.Inputfile as infile
import XPACDT.Tools.Units as units


class ThermostattedSamplingTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.parameters = infile.Inputfile("FilesForTesting/SamplingTest/input_Thermo.in")
        self.system = xSystem.System(self.parameters)

    def test_do_Thermostatted_sampling(self):
        samples = thermo.do_Thermostatted_sampling(self.system, self.parameters,
                                                   int(self.parameters.get("sampling").get('samples')))

        energies = [s.nuclei.energy for s in samples]
        statistics = scipy.stats.bayes_mvs(energies, alpha=0.9)
        mean_min, mean_max = statistics[0][1]
        dev_min, dev_max = statistics[2][1]
        mean_reference = 1.0 / (315777*units.boltzmann)

        self.assertTrue(mean_min < mean_reference < mean_max)
        self.assertTrue(dev_min < mean_reference < dev_max)
        self.assertEqual(len(samples), 2000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 1)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ThermostattedSamplingTest)
    unittest.TextTestRunner().run(suite)
