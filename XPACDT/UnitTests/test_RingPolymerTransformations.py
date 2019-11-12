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

import math
import numpy as np
from scipy import stats
import unittest

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo


class RingPolymerTransformationsTest(unittest.TestCase):

    def test_forward_backward_using_matrix(self):
        # 1 to 12 dimensions
        for k in range(1, 12):
            # several different bead numbers
            for n in [1, 8, 16, 64, 256]:
                nb_list = [n for i in range(k)]
                RPtransform = RPtrafo.RingPolymerTransformations(nb_list,
                                                                 'matrix')
                x = np.random.rand(n*k).reshape(k, n)
                nm = RPtransform.to_RingPolymer_normalModes(x)
                xt = RPtransform.from_RingPolymer_normalModes(nm)
                np.testing.assert_allclose(x, xt, rtol=1e-7)

                nm = np.random.rand(n*k).reshape(k, n)
                x = RPtransform.from_RingPolymer_normalModes(nm)
                nmt = RPtransform.to_RingPolymer_normalModes(x)
                np.testing.assert_allclose(nm, nmt, rtol=1e-7)
        return

    def test_forward_backward_using_fft(self):
        # 1 to 12 dimensions
        for k in range(1, 12):
            # several different bead numbers
            for n in [1, 8, 16, 64, 256]:
                nb_list = [n for i in range(k)]
                RPtransform = RPtrafo.RingPolymerTransformations(nb_list,
                                                                 'fft')
                x = np.random.rand(n*k).reshape(k, n)
                nm = RPtransform.to_RingPolymer_normalModes(x)
                xt = RPtransform.from_RingPolymer_normalModes(nm)
                np.testing.assert_allclose(x, xt, rtol=1e-7)

                nm = np.random.rand(n*k).reshape(k, n)
                x = RPtransform.from_RingPolymer_normalModes(nm)
                nmt = RPtransform.to_RingPolymer_normalModes(x)
                np.testing.assert_allclose(nm, nmt, rtol=1e-7)
        return

    def test_to_RingPolymer_normalModes_using_matrix(self):
        # 1 dimensional test
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124]])

        # n = x.shape[1]
        n = [8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'matrix')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        # 2 dimensional test
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583],
                      [0.5488135,   0.71518937,  0.60276338,  0.54488318,
                       0.4236548,   0.64589411,  0.43758721,  0.891773]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                            -0.34681862, -0.07448673, 0.05603399, 0.4119124],
                           [1.70078929,  0.209723,  -0.03394114,  -0.08456429,
                            -0.2775114,  0.18073258,  0.03778635, 0.01555642]])
        # n = x.shape[1]
        n = [8, 8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'matrix')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-6)

        return

    def test_to_RingPolymer_normalModes_using_fft(self):
        # 1 dimensional test
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124]])

        # n = x.shape[1]
        n = [8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'fft')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        # 2 dimensional test
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583],
                      [0.5488135,   0.71518937,  0.60276338,  0.54488318,
                       0.4236548,   0.64589411,  0.43758721,  0.891773]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                            -0.34681862, -0.07448673, 0.05603399, 0.4119124],
                           [1.70078929,  0.209723,  -0.03394114,  -0.08456429,
                            -0.2775114,  0.18073258,  0.03778635, 0.01555642]])
        # n = x.shape[1]
        n = [8, 8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'fft')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-6)

        return

    def test_from_RingPolymer_normalModes_using_matrix(self):
        # 1 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        n = [8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'matrix')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        # 2 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124],
                       [1.70078929,  0.209723,  -0.03394114,  -0.08456429,
                        -0.2775114,  0.18073258,  0.03778635, 0.01555642]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583],
                          [0.5488135,   0.71518937,  0.60276338,  0.54488318,
                           0.4236548,   0.64589411,  0.43758721,  0.891773]])
        n = [8, 8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'matrix')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-6)

        return

    def test_from_RingPolymer_normalModes_using_fft(self):
        # 1 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        n = [8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'fft')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        # 2 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124],
                       [1.70078929,  0.209723,  -0.03394114,  -0.08456429,
                        -0.2775114,  0.18073258,  0.03778635, 0.01555642]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583],
                          [0.5488135,   0.71518937,  0.60276338,  0.54488318,
                           0.4236548,   0.64589411,  0.43758721,  0.891773]])
        n = [8, 8]
        RPtransform = RPtrafo.RingPolymerTransformations(n, 'fft')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-6)

        return

    def test_1d_to_nm_using_matrix(self):
        x = np.random.rand(1)
        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        nm = RPtransform._1d_to_nm_using_matrix(x, n)
        self.assertSequenceEqual(nm, x)

        x = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        nm_ref = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124])

        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        nm = RPtransform._1d_to_nm_using_matrix(x, n)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        return

    def test_1d_to_nm_using_fft(self):
        x = np.random.rand(1)
        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        nm = RPtransform._1d_to_nm_using_fft(x, n)
        self.assertSequenceEqual(nm, x)

        x = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        nm_ref = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124])

        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        nm = RPtransform._1d_to_nm_using_fft(x, n)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        return

    def test_1d_from_nm_using_matrix(self):
        nm = np.random.rand(1)
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        x = RPtransform._1d_from_nm_using_matrix(nm, n)
        self.assertSequenceEqual(x, nm)

        nm = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                       -0.34681862, -0.07448673, 0.05603399, 0.4119124])
        x_ref = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                          -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        x = RPtransform._1d_from_nm_using_matrix(nm, n)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        return

    def test_1d_from_nm_using_fft(self):
        nm = np.random.rand(1)
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        x = RPtransform._1d_from_nm_using_fft(nm, n)
        self.assertSequenceEqual(x, nm)

        nm = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                       -0.34681862, -0.07448673, 0.05603399, 0.4119124])
        x_ref = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                          -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        x = RPtransform._1d_from_nm_using_fft(nm, n)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        return

    def test_nm_transformation_matrix(self):
        RPtransform = RPtrafo.RingPolymerTransformations([1], 'matrix')
        C_ref_1 = np.array([[1]])
        C_inv_ref_1 = np.array([[1]])
        np.testing.assert_allclose(RPtransform.C_matrices[1], C_ref_1,
                                   rtol=1e-7)
        np.testing.assert_allclose(RPtransform.C_inv_matrices[1], C_inv_ref_1,
                                   rtol=1e-7)

        RPtransform = RPtrafo.RingPolymerTransformations([2], 'matrix')
        C_ref_2 = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
        C_inv_ref_2 = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                                [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
        np.testing.assert_allclose(RPtransform.C_matrices[2], C_ref_2,
                                   rtol=1e-7)
        np.testing.assert_allclose(RPtransform.C_inv_matrices[2], C_inv_ref_2,
                                   rtol=1e-7)

        RPtransform = RPtrafo.RingPolymerTransformations([4], 'matrix')
        C_ref_4 = np.array([[1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],
                            [1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0), 0.0],
                            [1.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0, -1.0 / 2.0],
                            [0.0, -1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)]])
        C_inv_ref_4 = np.array([[1.0 / 2.0, 1.0 / np.sqrt(2.0), 1.0 / 2.0, 0.],
                                [1.0 / 2.0, 0.0, -1.0 / 2.0, -1.0 / np.sqrt(2.0)],
                                [1.0 / 2.0, -1.0 / np.sqrt(2.0), 1.0 / 2.0, 0.],
                                [1.0 / 2.0, 0.0, -1.0 / 2.0, 1.0 / np.sqrt(2.0)]])
        np.testing.assert_allclose(RPtransform.C_matrices[4], C_ref_4,
                                   atol=1e-8)
        np.testing.assert_allclose(RPtransform.C_inv_matrices[4], C_inv_ref_4,
                                   atol=1e-8)

        # Test if only one key-value pair for repeating number of beads
        RPtransform = RPtrafo.RingPolymerTransformations([1, 1, 1], 'matrix')
        self.assertTrue(list(RPtransform.C_matrices.keys()) == [1])
        self.assertTrue(list(RPtransform.C_inv_matrices.keys()) == [1])
        np.testing.assert_allclose(RPtransform.C_matrices[1], C_ref_1,
                                   rtol=1e-7)
        np.testing.assert_allclose(RPtransform.C_inv_matrices[1], C_inv_ref_1,
                                   rtol=1e-7)

        # Test for different number of beads for each dof
        RPtransform = RPtrafo.RingPolymerTransformations([1, 2, 4], 'matrix')
        self.assertTrue(list(RPtransform.C_matrices.keys()) == [1, 2, 4])
        self.assertTrue(list(RPtransform.C_inv_matrices.keys()) == [1, 2, 4])
        np.testing.assert_allclose(RPtransform.C_matrices[1], C_ref_1,
                                   rtol=1e-7)
        np.testing.assert_allclose(RPtransform.C_matrices[2], C_ref_2,
                                   rtol=1e-7)
        np.testing.assert_allclose(RPtransform.C_matrices[4], C_ref_4,
                                   atol=1e-8)

        # Test for None if 'fft' used
        RPtransform = RPtrafo.RingPolymerTransformations([1], 'fft')
        self.assertTrue(RPtransform.C_matrices is None)
        self.assertTrue(RPtransform.C_inv_matrices is None)

        return

    def test_sample_free_rp_momenta(self):
        mass = 1
        beta = 1
        centroid = 1.0

        # Checking if centroid value is fine
        n_list = [1, 4, 16, 64, 512]

        for n in n_list:
            RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
            p_rp = RPtransform.sample_free_rp_momenta(n, mass, beta, centroid)
            p_rp_centroid = np.mean(p_rp)
            np.testing.assert_allclose(p_rp_centroid, centroid, rtol=1e-7)

        # Test single sampling using seeded (known) random numbers
        nb = 4
        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')
        np.random.seed(0)
        p_rp = RP_nm_transform.sample_free_rp_momenta(nb, mass, beta, centroid)

        p_ref = np.array([3.89490396, -0.78430174, -1.09458954, 1.98398732])
        np.testing.assert_allclose(p_rp, p_ref, rtol=1e-7)

        # This is more of an integrated test to check sampling!!!
        # Test for proper distribution of momenta
        nb = 4
        samples = 10000
        mean_ref = 0.0
        std_ref = 2.0
        mean_centroid_ref = 2.0
        std_centroid_ref = 0.0

        np.random.seed(0)
        p_nm_arr = []
        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')

        for i in range(samples):
            p_rp = RP_nm_transform.sample_free_rp_momenta(nb, mass, beta,
                                                          centroid)
            p_nm_arr.append(RP_nm_transform.to_RingPolymer_normalModes(
                                p_rp.reshape(1, -1)).flatten())

        p_nm_arr = np.array(p_nm_arr)

        for i in range(nb):
            mean, var, std = stats.bayes_mvs(p_nm_arr[:, i], alpha=0.95)
            mean_min, mean_max = mean[1]
            std_min, std_max = std[1]
            if (i == 0):
                np.testing.assert_allclose(mean[0], mean_centroid_ref,
                                           rtol=1e-7)
                np.testing.assert_allclose(std[0], std_centroid_ref, atol=1e-8)
            else:
                self.assertTrue(mean_min < mean_ref < mean_max)
                self.assertTrue(std_min < std_ref < std_max)
        return

    def test_sample_free_rp_coord(self):
        mass = 1
        beta = 1
        centroid = 1.0

        # Checking if centroid value is fine
        n_list = [1, 4, 16, 64, 512]

        for n in n_list:
            RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
            x_rp = RPtransform.sample_free_rp_coord(n, mass, beta, centroid)
            x_rp_centroid = np.mean(x_rp)
            np.testing.assert_allclose(x_rp_centroid, centroid, rtol=1e-7)

        # Test single sampling using seeded (known) random numbers
        nb = 4
        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')
        np.random.seed(0)
        x_rp = RP_nm_transform.sample_free_rp_coord(nb, mass, beta, centroid)

        x_ref = np.array([1.49103274, 0.70529585, 0.60900656, 1.19466484])
        np.testing.assert_allclose(x_rp, x_ref, rtol=1e-7)

        # This is more of an integrated test to check sampling!!!
        # Test for proper distribution of positions
        nb = 4
        samples = 10000
        mean_ref = 0.0
        std_ref = [(1. / (4.*math.sin(i*math.pi/4.))) for i in range(1, nb)]
        std_ref.insert(0, 0.0)  # For centroid
        mean_centroid_ref = 2.0

        np.random.seed(0)
        x_nm_arr = []
        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')

        for i in range(samples):
            x_rp = RP_nm_transform.sample_free_rp_coord(nb, mass, beta,
                                                        centroid)
            x_nm_arr.append(RP_nm_transform.to_RingPolymer_normalModes(
                                x_rp.reshape(1, -1)).flatten())

        x_nm_arr = np.array(x_nm_arr)

        for i in range(nb):
            mean, var, std = stats.bayes_mvs(x_nm_arr[:, i], alpha=0.95)
            mean_min, mean_max = mean[1]
            std_min, std_max = std[1]
            if (i == 0):
                np.testing.assert_allclose(mean[0], mean_centroid_ref,
                                           rtol=1e-7)
                np.testing.assert_allclose(std[0], std_ref[i], atol=1e-8)
            else:
                self.assertTrue(mean_min < mean_ref < mean_max)
                self.assertTrue(std_min < std_ref[i] < std_max)
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RingPolymerTransformationsTest)
    unittest.TextTestRunner().run(suite)
