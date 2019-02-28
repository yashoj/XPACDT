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
#  CDTK is free software: you can redistribute it and/or modify
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

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo


class RingPolymerTransformationsTest(unittest.TestCase):

    def test_forward_backward(self):
        # 1 to 12 dimensions
        for k in range(1, 12):
            # several different bead numbers
            for n in [1, 8, 16, 64, 256]:
                x = np.random.rand(n*k).reshape(k, n)
                nm = RPtrafo.to_RingPolymer_normalModes(x)
                xt = RPtrafo.from_RingPolymer_normalModes(nm)
                np.testing.assert_allclose(x, xt, rtol=1e-7)

                nm = np.random.rand(n*k).reshape(k, n)
                x = RPtrafo.from_RingPolymer_normalModes(nm)
                nmt = RPtrafo.to_RingPolymer_normalModes(x)
                np.testing.assert_allclose(nm, nmt, rtol=1e-7)
        return

    def test_to_RingPolymer_normalModes(self):
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        nm = RPtrafo.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

    # TODO: more elaborate multi-dimensional tests
        return

    def test_from_RingPolymer_normalModes(self):
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        x = RPtrafo.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

    # TODO: more elaborate multi-dimensional tests
        return

    def test_1d_to_nm(self):
        x = np.random.rand(3)
        with self.assertRaises(AssertionError):
            nm = RPtrafo._1d_to_nm(x, 3)

        x = np.random.rand(1)
        nm = RPtrafo._1d_to_nm(x, 1)
        self.assertSequenceEqual(nm, x)

        x = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        nm_ref = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124])

        nm = RPtrafo._1d_to_nm(x, 8)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        # TODO: 16 bead test
        return

    def test_1d_from_nm(self):
        nm = np.random.rand(3)
        with self.assertRaises(AssertionError):
            x = RPtrafo._1d_from_nm(nm, 3)

        nm = np.random.rand(1)
        x = RPtrafo._1d_from_nm(nm, 1)
        self.assertSequenceEqual(x, nm)

        nm = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                       -0.34681862, -0.07448673, 0.05603399, 0.4119124])
        x_ref = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                          -2.07919478, -1.70089421, -1.72930132, -1.42050583])

        x = RPtrafo._1d_from_nm(nm, 8)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        # TODO: 16 bead test
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RingPolymerTransformationsTest)
    unittest.TextTestRunner().run(suite)
