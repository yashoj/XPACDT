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

import XPACDT.Tools.MathematicalTools as mtools


class MathematicalToolsTest(unittest.TestCase):

    def test_linear_interpolation_1d(self):
        # For floats
        self.assertAlmostEqual(mtools.linear_interpolation_1d(0.5, 0., 2.4), 1.2)
        self.assertAlmostEqual(mtools.linear_interpolation_1d(0.25, 0., 2.4), 0.6)

        # For 1d array
        y1 = np.array([0., 1.])
        y2 = np.array([2.4, 3.4])
        np.testing.assert_allclose(mtools.linear_interpolation_1d(0.5, y1, y2),
                                   np.array([1.2, 2.2]), rtol=1e-7)
        np.testing.assert_allclose(mtools.linear_interpolation_1d(0.25, y1, y2),
                                   np.array([0.6, 1.6]), rtol=1e-7)

        # For 2d array
        y1 = np.array([[0., 1.], [-1., -3.4]])
        y2 = np.array([[2.4, 3.4], [-3.4, -1.0]])
        np.testing.assert_allclose(mtools.linear_interpolation_1d(0.5, y1, y2),
                                   np.array([[1.2, 2.2], [-2.2, -2.2]]), rtol=1e-7)
        np.testing.assert_allclose(mtools.linear_interpolation_1d(0.25, y1, y2),
                                   np.array([[0.6, 1.6], [-1.6, -2.8]]), rtol=1e-7)
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(MathematicalToolsTest)
    unittest.TextTestRunner().run(suite)
