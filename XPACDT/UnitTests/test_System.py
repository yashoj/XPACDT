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
import unittest

import XPACDT.System.System as xSystem
import XPACDT.Input.Inputfile as infile


class SystemTest(unittest.TestCase):

    def setUp(self):
        self.parameters = infile.Inputfile("FilesForTesting/SamplingTest/input_fixed.in")
        self.system = xSystem.System(self.parameters)
        self.nuclei = copy.deepcopy(self.system.nuclei)

    def test_do_log(self):
        # initial setup check
        self.assertEqual(len(self.system.log), 1)
        self.assertEqual(self.system.log[0].time, 0.0)
        self.assertEqual(self.system.log[0], self.nuclei)

        # do one log
        self.system.do_log()
        self.assertEqual(len(self.system.log), 2)
        self.assertEqual(self.system.log[1].time, 0.0)
        self.assertEqual(self.system.log[1], self.nuclei)

        return

    def test_reset(self):
        self.system.log.append(copy.deepcopy(self.system.log[0]))
        self.system.log[-1].time = 1.0
        self.system.log[-1].positions[0, 0] = 10.0

        self.system.reset()
        self.assertEqual(len(self.system.log), 1)
        self.assertEqual(self.system.log[0].time, 0.0)

        self.nuclei_ref = copy.deepcopy(self.nuclei)
        self.nuclei_ref.positions[0, 0] = 10.0
        self.assertEqual(self.system.nuclei, self.nuclei)
        self.assertNotEqual(self.system.nuclei, self.nuclei_ref)

        return

    def test_step(self):
        # Step equal to single nuclear timestep

        # Set up dummy propagator
        self.system.nuclei.propagator = DummyProp(1.0)
        self.system.step(1.0)

        # check correct advance
        self.nuclei_ref = copy.deepcopy(self.nuclei)
        self.nuclei_ref.positions *= 2.0
        self.nuclei_ref.momenta *= 2.0
        self.assertEqual(self.system.nuclei.time, 1.0)
        self.assertEqual(len(self.system.log), 2)
        self.assertEqual(self.system.nuclei, self.nuclei_ref)

        # check correct logging
        self.assertEqual(self.system.log[0].time, 0.0)
        self.assertEqual(self.system.log[0], self.nuclei)

        self.assertEqual(self.system.log[1].time, 1.0)
        self.assertEqual(self.system.log[1], self.nuclei_ref)

        ###############
        # Step multiple of 10 nuclear timesteps

        # reset system
        self.system = xSystem.System(self.parameters)
        self.system.nuclei.propagator = DummyProp(0.1)
        self.system.step(1.0)

        # check correct advance
        self.nuclei_ref = copy.deepcopy(self.nuclei)
        for i in range(10):
            self.nuclei_ref.positions *= 2.0
            self.nuclei_ref.momenta *= 2.0
        self.assertAlmostEqual(self.system.nuclei.time, 1.0)
        self.assertEqual(len(self.system.log), 2)
        self.assertEqual(self.system.nuclei, self.nuclei_ref)

        ###############
        # Step not multiple of nuclear timestep; should give error

        # reset system
        self.system = xSystem.System(self.parameters)
        self.system.nuclei.propagator = DummyProp(0.3)
        with self.assertRaises(RuntimeError):
            self.system.step(1.0)

        return


class DummyProp(object):
    def __init__(self, timestep):
        self.timestep = float(timestep)
        pass

    def propagate(self, R, P, time_propagation, time):
        return (2.0 * R), (2.0 * P)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(SystemTest)
    unittest.TextTestRunner().run(suite)
