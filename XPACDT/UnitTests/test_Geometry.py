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

import XPACDT.Tools.Geometry as geom


class GeometryTest(unittest.TestCase):

    def test_angle(self):
        # Same direction -> angle should be 0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])

        phi_reference = 0.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Orthogonal -> angle should be \pi / 2
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        phi_reference = np.pi / 2.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Opposite direction -> angle should be \pi
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])

        phi_reference = np.pi
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Mid of first quadrant -> angle should be \pi / 4
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])

        phi_reference = np.pi / 4.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Mid of second quadrant -> angle should be 3 \pi / 4
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 1.0, 0.0])

        phi_reference = 3.0 * np.pi / 4.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Mid of third quadrant -> angle should be 3 \pi / 4
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, -1.0, 0.0])

        phi_reference = 3.0 * np.pi / 4.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)

        # Mid of fourth quadrant -> angle should be \pi / 4
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, -1.0, 0.0])

        phi_reference = np.pi / 4.0
        phi = geom.angle(a, b)

        self.assertAlmostEqual(phi, phi_reference)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(GeometryTest)
    unittest.TextTestRunner().run(suite)
