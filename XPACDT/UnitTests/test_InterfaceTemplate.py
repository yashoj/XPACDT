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

import XPACDT.Interfaces.InterfaceTemplate as IT


class InterfaceTemplateTest(unittest.TestCase):

    def setUp(self):
        # todo create input file here.
        self.interface = IT.Interface("dummyTemplate")
        return

    def test_changed(self):
        self.assertEqual(self.interface.name, "dummyTemplate")

        # test the given parameters
        with self.assertRaises(AssertionError):
            self.interface._changed([0.0], None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([0.0]), None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[[0.0]]]), None)

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), [0.0])

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), np.array([0.0]))

        with self.assertRaises(AssertionError):
            self.interface._changed(np.array([[0.0]]), np.array([[[0.0]]]))

        # initial storage
        R = np.random.rand(12).reshape(4, 3)
        self.assertTrue(self.interface._changed(R, None))
        self.assertTrue(np.alltrue(self.interface._old_R == R))

        # not changed
        self.assertFalse(self.interface._changed(R, None))

        # changed
        R = np.random.rand(12).reshape(4, 3)
        self.assertTrue(self.interface._changed(R, None))
        self.assertTrue(np.alltrue(self.interface._old_R == R))

        return

#    def test_minimize(self):
#        # TODO: howto exactly test
#        raise NotImplementedError("Please implement a test here!!")
#        pass
#
#    def test_plot1d(self):
#        # TODO: howto exactly test
#        raise NotImplementedError("Please implement a test here!!")
#        pass
#
#    def test_plot2d(self):
#        # TODO: howto exactly test
#        raise NotImplementedError("Please implement a test here!!")
#        pass
#
#    def test_get_Hessian(self):
#        # TODO: howto exactly test
#        raise NotImplementedError("Please implement a test here!!")
#        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(InterfaceTemplateTest)
    unittest.TextTestRunner().run(suite)
