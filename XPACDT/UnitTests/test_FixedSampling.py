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
import unittest

import XPACDT.Sampling.FixedSampling as fixed
import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile


class FixedSamplingTest(unittest.TestCase):

    def setUp(self):
        self.parameters = infile.Inputfile("FilesForTesting/SamplingTest/input_fixed.in")
        self.system = xSystem.System(self.parameters)

    def test_do_Fixed_sampling(self):
        coordinate_ref = np.array([[2.0], [1.0], [4.0], [-2.0]])
        momenta_ref = np.array([[-1.0], [0.1], [2.0], [1.25]])
        samples = fixed.do_Fixed_sampling(self.system, self.parameters,
                                          int(self.parameters.get("sampling").get('samples')))

        self.assertEqual(len(samples), 1000)
        for s in samples:
            self.assertEqual(s.nuclei.n_dof, 4)
            np.testing.assert_allclose(s.nuclei.positions, coordinate_ref,
                                       rtol=1e-7)
            np.testing.assert_allclose(s.nuclei.momenta, momenta_ref,
                                       rtol=1e-7)
            # TODO: more attributes to be checked?

    def test_shifts(self):
        raise NotImplementedError("Please implement a test here!!")
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(FixedSamplingTest)
    unittest.TextTestRunner().run(suite)
