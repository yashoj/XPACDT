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
import math
import unittest

import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad
import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad2S
import XPACDT.Interfaces.MorseDiabatic as morsedia
import XPACDT.Input.Inputfile as infile


class DiabaticToAdiabaticNstatesTest(unittest.TestCase):

    def test_get_adiabatic_energy(self):
        # 1 bead/centroid, 2 states test
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

        # 3 bead, 2 states test
        V = np.array([[[2., 0., 2.], [0., 1., 1.]], [[0., 1., 1.], [3., 0., 3.]]])
        V_ad_ref = np.array([[2., -1., (5. - math.sqrt(5.)) / 2.],
                             [3., 1., (5. + math.sqrt(5.)) / 2.]])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        # 1 bead, 3 states test
        V = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        V_ad_ref = np.array([1., 2., 3.])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        V = np.array([[1., 0., 0.], [0., 2., 1.], [0., 1., 3.]])
        V_ad_ref = np.array([1., (5. - math.sqrt(5.)) / 2., (5. + math.sqrt(5.)) / 2.])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

        # 2 bead, 3 states test
        V = np.array([[[1., 1.], [0., 0.], [0., 0.]],
                      [[0., 0.], [2., 2.], [0., 1.]],
                      [[0., 0.], [0., 1.], [3., 3.]]])
        V_ad_ref = np.array([[1., 1.], [2., (5. - math.sqrt(5.)) / 2.],
                             [3., (5. + math.sqrt(5.)) / 2.]])
        V_ad = dia2ad.get_adiabatic_energy(V)
        np.testing.assert_allclose(V_ad, V_ad_ref, rtol=1e-7)

    def test_get_adiabatic_gradient(self):
        # TODO: Compare to actual calculated values rather than just comparing
        #       to 2 state module value. Also add 3 state tests.

        # 1 bead, 1 dof, 2 states test
        pes = morsedia.MorseDiabatic(infile.Inputfile("FilesForTesting/InterfaceTests/input_Morse1_1-2states.in"))

        R = np.array([[3.4]])
        pes._calculate_diabatic_all(R)
        dV_ad = dia2ad.get_adiabatic_gradient(R, pes._get_diabatic_energy_matrix,
                                              pes.DERIVATIVE_STEPSIZE)
        dV_ad_centroid = dia2ad.get_adiabatic_gradient(R[:, 0], pes._get_diabatic_energy_matrix,
                                                       pes.DERIVATIVE_STEPSIZE)

        dV_ad_ref = dia2ad2S.get_adiabatic_gradient(pes._diabatic_energy,
                                                    pes._diabatic_gradient)
        dV_ad_centroid_ref = dia2ad2S.get_adiabatic_gradient(pes._diabatic_energy_centroid,
                                                             pes._diabatic_gradient_centroid)
        np.testing.assert_allclose(dV_ad, dV_ad_ref, rtol=1e-7)
        np.testing.assert_allclose(dV_ad_centroid, dV_ad_centroid_ref, rtol=1e-7)

        # 2 bead, 1 dof, 2 states test
        pes =  morsedia.MorseDiabatic(infile.Inputfile("FilesForTesting/InterfaceTests/input_Morse1_2-2states.in"))

        R = np.array([[3.4, 4.8]])
        pes._calculate_diabatic_all(R)
        dV_ad = dia2ad.get_adiabatic_gradient(R, pes._get_diabatic_energy_matrix,
                                              pes.DERIVATIVE_STEPSIZE)

        dV_ad_ref = dia2ad2S.get_adiabatic_gradient(pes._diabatic_energy,
                                                    pes._diabatic_gradient)
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
        # TODO: After fixing random pahse error in getting eigenvectors,
        #       uncomment the first reference values.
        # nac_ref = np.array([[[0.], [-0.02]], [[0.02], [0.]]])
        nac_ref = np.array([[[0.], [0.02]], [[-0.02], [0.]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        # 2 bead, 1 dof test
        V = np.array([[[2., 2.], [0., 1.]], [[0., 1.], [3., 3.]]])
        dV = np.array([[[[0.2, 0.4]], [[0., 0.1]]], [[[0., 0.1]], [[0.3, 0.6]]]])
        nac = dia2ad.get_NAC(V, dV)
        # nac_ref = np.array([[[[0., 0.]], [[0., -0.02]]],
        #                     [[[0., 0.02]], [[0., 0.]]]])
        nac_ref = np.array([[[[0., 0.]], [[0., 0.02]]],
                            [[[0., -0.02]], [[0., 0.]]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        # 1 bead/centroid, 2 dof test
        V = np.array([[2., 1.], [1., 3.]])
        dV = np.array([[[0.2, 0.4], [0., 0.1]], [[0., 0.1], [0.3, 0.6]]])
        nac = dia2ad.get_NAC(V, dV)
        # nac_ref = np.array([[[0., 0.], [-0.02, -0.02]],
        #                    [[0.02, 0.02], [0., 0.]]])
        nac_ref = np.array([[[0., 0.], [0.02, 0.02]],
                            [[-0.02, -0.02], [0., 0.]]])
        np.testing.assert_allclose(nac, nac_ref, rtol=1e-7)

        # TODO: Add 3 or more state test.

    def test_get_transformation_matrix(self):
        # 1 bead/centroid test
        V = np.array([[2., 0.], [0., 3.]])
        # TODO: After fixing random pahse error in getting eigenvectors,
        #       uncomment the first reference values.
        # U_ref = np.array([[0., -1.], [1., 0.]])
        U_ref = np.array([[1., 0.], [0., 1.]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, atol=1e-8)

        V = np.array([[1./np.sqrt(3), 0.5], [0.5, 0.]])
        # U_ref = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
        U_ref = np.array([[0.5, -np.sqrt(3)/2], [-np.sqrt(3)/2, -0.5]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, rtol=1e-7)

        # 2 bead test
        V = np.array([[[2., 1./np.sqrt(3)], [0., 0.5]], [[0., 0.5], [3., 0.]]])
        # U_ref = np.array([[[0., np.sqrt(3)/2], [-1., -0.5]],
        #                  [[1., 0.5], [0., np.sqrt(3)/2]]])
        U_ref = np.array([[[1., 0.5], [0., -np.sqrt(3)/2]],
                          [[0., -np.sqrt(3)/2], [1., -0.5]]])
        U = dia2ad.get_transformation_matrix(V)
        np.testing.assert_allclose(U, U_ref, atol=1e-8)

        # TODO: Add 3 or more state test.


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(DiabaticToAdiabaticNstatesTest)
    unittest.TextTestRunner().run(suite)
