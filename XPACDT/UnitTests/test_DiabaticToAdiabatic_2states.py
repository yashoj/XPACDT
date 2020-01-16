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
import math
import unittest

import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad


class DiabaticToAdiabatic2statesTest(unittest.TestCase):

    def test_get_adiabatic_energy(self):
        # 1 bead/centroid test
        V = np.array([[2., 0.], [0., 3.]])
        V_ad_ref = np.array([2., 3.])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        V = np.array([[0., 1.], [1., 0.]])
        V_ad_ref = np.array([-1., 1.])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        V = np.array([[2., 1.], [1., 3.]])
        V_ad_ref = np.array([(5. - math.sqrt(5.)) / 2., (5. + math.sqrt(5.)) / 2.])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        # 3 bead test
        V = np.array([[[2., 0., 2.], [0., 1., 1.]], [[0., 1., 1.], [3., 0., 3.]]])
        V_ad_ref = np.array([[2., -1., (5. - math.sqrt(5.)) / 2.],
                             [3., 1., (5. + math.sqrt(5.)) / 2.]])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

    def test_get_adiabatic_gradient(self):
        # 1 bead/centroid, 1 dof test
        V = np.array([[2., 0.], [0., 3.]])
        dV = np.array([[[0.2], [0.]], [[0.], [0.3]]])
        dV_ad = dia2ad.get_adiabatic_gradient(V, dV)
        dV_ad_ref = np.array([[0.2], [0.3]])
        np.testing.assert_allclose(dV_ad, dV_ad_ref, rtol=1e-7)

        V = np.array([[2., 1.], [1., 3.]])
        dV = np.array([[[0.4], [0.1]], [[0.1], [0.6]]])
        dV_ad = dia2ad.get_adiabatic_gradient(V, dV)
        dV_ad_ref = np.array([[0.5 - 0.3/np.sqrt(5)], [0.5 + 0.3/np.sqrt(5)]])
        np.testing.assert_allclose(dV_ad, dV_ad_ref, rtol=1e-7)

        # 2 bead, 1 dof test
        V = np.array([[[2., 2.], [0., 1.]], [[0., 1.], [3., 3.]]])
        dV = np.array([[[[0.2, 0.4]], [[0., 0.1]]], [[[0., 0.1]], [[0.3, 0.6]]]])
        dV_ad = dia2ad.get_adiabatic_gradient(V, dV)
        dV_ad_ref = np.array([[[0.2, 0.5 - 0.3/np.sqrt(5)]],
                              [[0.3, 0.5 + 0.3/np.sqrt(5)]]])
        np.testing.assert_allclose(dV_ad, dV_ad_ref, rtol=1e-7)

        # 1 bead/centroid, 2 dof test
        V = np.array([[2., 1.], [1., 3.]])
        dV = np.array([[[0.2, 0.4], [0., 0.1]], [[0., 0.1], [0.3, 0.6]]])
        dV_ad = dia2ad.get_adiabatic_gradient(V, dV)
        dV_ad_ref = np.array([[0.25 - 0.05/np.sqrt(5), 0.5 - 0.3/np.sqrt(5)],
                              [0.25 + 0.05/np.sqrt(5), 0.5 + 0.3/np.sqrt(5)]])
        np.testing.assert_allclose(dV_ad, dV_ad_ref, rtol=1e-7)

    def test_get_NAC(self):
        # 1 bead/centroid, 1 dof test
        V = np.array([[2., 0.], [0., 3.]])
        dV = np.array([[[0.2], [0.]], [[0.], [0.3]]])
        nac = dia2ad.get_NAC(V, dV)
        nac_ref = np.array([[[0.], [0.0]], [[0.0], [0.]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        V = np.array([[2., 1.], [1., 3.]])
        dV = np.array([[[0.4], [0.1]], [[0.1], [0.6]]])
        nac = dia2ad.get_NAC(V, dV)
        nac_ref = np.array([[[0.], [-0.02]], [[0.02], [0.]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        # 2 bead, 1 dof test
        V = np.array([[[2., 2.], [0., 1.]], [[0., 1.], [3., 3.]]])
        dV = np.array([[[[0.2, 0.4]], [[0., 0.1]]], [[[0., 0.1]], [[0.3, 0.6]]]])
        nac = dia2ad.get_NAC(V, dV)
        nac_ref = np.array([[[[0., 0.]], [[0., -0.02]]],
                            [[[0., 0.02]], [[0., 0.]]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        # 1 bead/centroid, 2 dof test
        V = np.array([[2., 1.], [1., 3.]])
        dV = np.array([[[0.2, 0.4], [0., 0.1]], [[0., 0.1], [0.3, 0.6]]])
        nac = dia2ad.get_NAC(V, dV)
        nac_ref = np.array([[[0., 0.], [-0.02, -0.02]],
                            [[0.02, 0.02], [0., 0.]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

    def test_get_transformation_matrix(self):
        # 1 bead/centroid test
        V = np.array([[2., 0.], [0., 3.]])
        U_ref = np.array([[0., -1.], [1., 0.]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, atol=1e-8)

        V = np.array([[1./np.sqrt(3), 0.5], [0.5, 0.]])
        U_ref = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, rtol=1e-7)

        # 2 bead test
        V = np.array([[[2., 1./np.sqrt(3)], [0., 0.5]], [[0., 0.5], [3., 0.]]])
        U_ref = np.array([[[0., np.sqrt(3)/2], [-1., -0.5]],
                          [[1., 0.5], [0., np.sqrt(3)/2]]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, atol=1e-8)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(DiabaticToAdiabatic2statesTest)
    unittest.TextTestRunner().run(suite)
