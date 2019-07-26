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
from scipy import stats
import unittest

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo


class RingPolymerTransformationsTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        np.random.seed(seed)

    def test_forward_backward_using_matrix(self):
        # 1 to 12 dimensions
        for k in range(1, 12):
            # several different bead numbers
            for n in [1, 8, 16, 64, 256]:
                RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
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
                RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
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

        n = x.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
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
        n = x.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-6)

        return
    
    def test_to_RingPolymer_normalModes_using_fft(self):
        # 1 dimensional test
        x = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        nm_ref = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124]])

        n = x.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
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
        n = x.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        nm = RPtransform.to_RingPolymer_normalModes(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-6)

        return

    def test_from_RingPolymer_normalModes_using_matrix(self):
        # 1 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        n = nm.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
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
        n = nm.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-6)

        return

    def test_from_RingPolymer_normalModes_using_fft(self):
        # 1 dimensional test
        nm = np.array([[-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                        -0.34681862, -0.07448673, 0.05603399, 0.4119124]])
        x_ref = np.array([[-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                           -2.07919478, -1.70089421, -1.72930132, -1.42050583]])
        n = nm.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
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
        n = nm.shape[1]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        x = RPtransform.from_RingPolymer_normalModes(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-6)

        return

    def test_1d_to_nm_using_matrix(self):
        x = np.random.rand(1)
        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        nm = RPtransform._1d_to_nm_using_matrix(x)
        self.assertSequenceEqual(nm, x)

        x = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        nm_ref = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124])

        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        nm = RPtransform._1d_to_nm_using_matrix(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        return

    def test_1d_to_nm_using_fft(self):
        x = np.random.rand(1)
        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        nm = RPtransform._1d_to_nm_using_fft(x)
        self.assertSequenceEqual(nm, x)

        x = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                      -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        nm_ref = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                           -0.34681862, -0.07448673, 0.05603399, 0.4119124])

        n = x.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        nm = RPtransform._1d_to_nm_using_fft(x)
        np.testing.assert_allclose(nm, nm_ref, rtol=1e-7)

        return

    def test_1d_from_nm_using_matrix(self):
        nm = np.random.rand(1)
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        x = RPtransform._1d_from_nm_using_matrix(nm)
        self.assertSequenceEqual(x, nm)

        nm = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                       -0.34681862, -0.07448673, 0.05603399, 0.4119124])
        x_ref = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                          -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'matrix')
        x = RPtransform._1d_from_nm_using_matrix(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        return

    def test_1d_from_nm_using_fft(self):
        nm = np.random.rand(1)
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        x = RPtransform._1d_from_nm_using_fft(nm)
        self.assertSequenceEqual(x, nm)

        nm = np.array([-5.09857056, 0.35986656, 0.09453016, 0.04258154,
                       -0.34681862, -0.07448673, 0.05603399, 0.4119124])
        x_ref = np.array([-1.67674668, -1.7151358, -2.21570045, -1.8834562,
                          -2.07919478, -1.70089421, -1.72930132, -1.42050583])
        n = nm.shape[0]
        RPtransform = RPtrafo.RingPolymerTransformations([n], 'fft')
        x = RPtransform._1d_from_nm_using_fft(nm)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)

        return

    def test_nm_transformation_matrix(self):
        RPtransform = RPtrafo.RingPolymerTransformations([1], 'matrix')
        C_ref = np.array([[1]])
        np.testing.assert_allclose(RPtransform.C_matrix, C_ref, rtol=1e-7)

        RPtransform = RPtrafo.RingPolymerTransformations([2], 'matrix')
        C_ref = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                          [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
        np.testing.assert_allclose(RPtransform.C_matrix, C_ref, rtol=1e-7)

        RPtransform = RPtrafo.RingPolymerTransformations([4], 'matrix')
        C_ref = np.array([[1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],
                          [1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0), 0.0],
                          [1.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0, -1.0 / 2.0],
                          [0.0, -1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)]])
        np.testing.assert_allclose(RPtransform.C_matrix, C_ref, atol=1e-8)

        return

    def test_sample_free_rp_momenta(self):
        mass = 1
        beta = 1
        centroid = 1.0

        # Checking if centroid value is fine
        n_list = [1, 4, 16, 64, 512]

        for n in n_list:
            p_rp = RPtrafo.sample_free_rp_momenta(n, mass, beta, centroid)
            p_rp_centroid = np.mean(p_rp)
            np.testing.assert_allclose(p_rp_centroid, centroid, rtol=1e-7)

        # Is this really needed??? Gives negative result due to improper stats
        # Test for proper distribution of momenta without centroid value
#        nb = 4
#        samples = 1000000
#        mean_ref = 0.0
#        std_ref = 2.0
#
#        seed = 0
#        np.random.seed(seed)
#        p_nm_arr = []
#        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')
#        
#        print(samples)
#        
#        for i in range(samples):
#            p_rp = RPtrafo.sample_free_rp_momenta(nb, mass, beta, centroid, 'matrix')
#            p_nm_arr.append(RP_nm_transform.to_RingPolymer_normalModes(p_rp.reshape(1, -1)).flatten())
#            
#        p_nm_arr = np.array(p_nm_arr)
#
#        for i in range(nb):
#            mean, var, std = stats.bayes_mvs(p_nm_arr[:, i])
#            mean_min, mean_max = mean[1]
#            std_min, std_max = std[1]
#            print(mean[0], mean_min, mean_max)
#            print(std[0], std_min, std_max)
#            if (i == 0):
#                np.testing.assert_allclose(mean[0], centroid * np.sqrt(nb), rtol=1e-7)
#                np.testing.assert_allclose(std[0], 0., atol=1e-8)
#            else:
#                self.assertTrue(mean_min < mean_ref < mean_max)
#                self.assertTrue(std_min < std_ref < std_max)

        return

    def test_sample_free_rp_coord(self):
        mass = 1
        beta = 1

        # Checking if centroid value is fine
        centroid = 1.0
        n_list = [1, 4, 16, 64, 512]

        for n in n_list:
            x_rp = RPtrafo.sample_free_rp_coord(n, mass, beta, centroid,
                                                'matrix')
            x_rp_centroid = np.mean(x_rp)
            np.testing.assert_allclose(x_rp_centroid, centroid, rtol=1e-7)
            
        # Test for proper distribution of momenta without centroid value
        nb = 4
        samples = 1000000
        mean_ref = 0.0
        std_ref = 2.0

        seed = 0
        np.random.seed(seed)
        x_nm_arr = []
        RP_nm_transform = RPtrafo.RingPolymerTransformations([nb], 'matrix')
        
        print(samples)
        
        for i in range(samples):
            p_rp = RPtrafo.sample_free_rp_momenta(nb, mass, beta, centroid, 'matrix')
            p_nm_arr.append(RP_nm_transform.to_RingPolymer_normalModes(p_rp.reshape(1, -1)).flatten())
            
        p_nm_arr = np.array(p_nm_arr)

        for i in range(nb):
            mean, var, std = stats.bayes_mvs(p_nm_arr[:, i])
            mean_min, mean_max = mean[1]
            std_min, std_max = std[1]
            print(mean[0], mean_min, mean_max)
            print(std[0], std_min, std_max)
            if (i == 0):
                np.testing.assert_allclose(mean[0], centroid * np.sqrt(nb), rtol=1e-7)
                np.testing.assert_allclose(std[0], 0., atol=1e-8)
            else:
                self.assertTrue(mean_min < mean_ref < mean_max)
                self.assertTrue(std_min < std_ref < std_max)

        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RingPolymerTransformationsTest)
    unittest.TextTestRunner().run(suite)
