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

import molmod.constants as const
import numpy as np
import random
import scipy.stats
import unittest

import XPACDT.Dynamics.System as xSystem
import XPACDT.Dynamics.WignerSampling as wigner
import XPACDT.Input.Inputfile as infile


class WignerSamplingTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.parameters0 = infile.Inputfile(**{'filename': "FilesForTesting/SamplingTest/input_Wigner_0.in"})
        self.system0 = xSystem.System(self.parameters0)

        self.parameters2 = infile.Inputfile(**{'filename': "FilesForTesting/SamplingTest/input_Wigner_300K.in"})
        self.system2 = xSystem.System(self.parameters2)

    def test_do_Wigner_sampling(self):
        samples = wigner.do_Wigner_sampling(self.system0, self.parameters0)
        energies = [s.nuclei.energy for s in samples]
        statistics = scipy.stats.bayes_mvs(energies)
        mean_min, mean_max = statistics[0][1]
        dev_min, dev_max = statistics[2][1]

        self.assertTrue(mean_min < 0.5 < mean_max)
        self.assertTrue(dev_min < 0.5 < dev_max)
        self.assertEqual(len(samples), 100000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 1)

        samples = wigner.do_Wigner_sampling(self.system2, self.parameters2)
        energies = [s.nuclei.energy for s in samples]
        statistics = scipy.stats.bayes_mvs(energies)
        mean_min, mean_max = statistics[0][1]
        dev_min, dev_max = statistics[2][1]
        mean_reference = 1.0/(np.exp(1.0 / (315777*const.boltzmann))-1.0)+0.5

        self.assertTrue(mean_min < mean_reference < mean_max)
        self.assertTrue(dev_min < mean_reference < dev_max)
        self.assertEqual(len(samples), 100000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 1)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(WignerSamplingTest)
    unittest.TextTestRunner().run(suite)
