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

import XPACDT.Interfaces.MorseDiabatic as morsedia


class MorseDiabaticTest(unittest.TestCase):

    def setUp(self):
        self.pes_model1_1_nb = morsedia.MorseDiabatic(1, **{'n_states': '3',
                                                            'model_type': 'model_1'})
        self.pes_model1_2_nb = morsedia.MorseDiabatic(2, **{'n_states': '3',
                                                            'model_type': 'model_1'})

        self.pes_model2_1_nb = morsedia.MorseDiabatic(1, **{'n_states': '3',
                                                            'model_type': 'model_2'})
        self.pes_model2_2_nb = morsedia.MorseDiabatic(2, **{'n_states': '3',
                                                            'model_type': 'model_2'})

        self.pes_model3_1_nb = morsedia.MorseDiabatic(1, **{'n_states': '3',
                                                            'model_type': 'model_3'})
        self.pes_model3_2_nb = morsedia.MorseDiabatic(2, **{'n_states': '3',
                                                            'model_type': 'model_3'})

        return

    def test_creation(self):
        with self.assertRaises(AssertionError):
            pes = morsedia.MorseDiabatic(1)

        self.assertEqual(self.pes_model1_1_nb.name, 'MorseDiabatic')
        self.assertEqual(self.pes_model1_1_nb.model_type, 'model_1')

        return

    def test_calculate_diabatic_all(self):
        # Test correct diabatic potential values and gradients for all states

        # Model 1: 1 bead
        self.pes_model1_1_nb._calculate_diabatic_all(np.array([[3.4]]))
        np.testing.assert_allclose(self.pes_model1_1_nb._diabatic_energy,
                                   [[[0.01003810], [0.002], [0.0]],
                                    [[0.002], [0.01075110], [4.80346956e-17]],
                                    [[0.0], [4.80346956e-17], [0.06459543]]], rtol=1e-6)
        np.testing.assert_allclose(self.pes_model1_1_nb._diabatic_gradient,
                                   [[[[-0.02018348]], [[0.0]], [[0.0]]],
                                    [[[0.0]], [[-0.00298130]], [[2.15195436e-15]]],
                                    [[[0.0]], [[2.15195436e-15]], [[-0.09341003]]]], rtol=1e-6)
        np.testing.assert_allclose(self.pes_model1_1_nb._diabatic_energy_centroid,
                                   [[0.01003810, 0.002, 0.0],
                                    [0.002, 0.01075110, 4.80346956e-17],
                                    [0.0, 4.80346956e-17, 0.06459543]], rtol=1e-6)
        np.testing.assert_allclose(self.pes_model1_1_nb._diabatic_gradient_centroid,
                                   [[[-0.02018348], [0.0], [0.0]],
                                    [[0.0], [-0.00298130], [2.15195436e-15]],
                                    [[0.0], [2.15195436e-15], [-0.09341003]]], rtol=1e-6)

        # Model 1: 2 beads
        self.pes_model1_2_nb._calculate_diabatic_all(np.array([[3.4, 4.8]]))
        np.testing.assert_allclose(self.pes_model1_2_nb._diabatic_energy,
                                   [[[0.01003810, 5.78199600e-05], [0.002, 4.80346956e-17], [0.0, 0.0]],
                                    [[0.002, 4.80346956e-17], [0.01075110, 1.05813044e-02], [4.80346956e-17, 0.002]],
                                    [[0.0, 0.0], [4.80346956e-17, 0.002], [0.06459543, 1.01876301e-02]]], rtol=1e-6)
        np.testing.assert_allclose(self.pes_model1_2_nb._diabatic_gradient,
                                   [[[[-0.02018348, -6.16596643e-04]], [[0.0, -2.15195436e-15]], [[0.0, 0.0]]],
                                    [[[0.0, -2.15195436e-15]], [[-0.00298130, 1.13227443e-03]], [[2.15195436e-15, 0.0]]],
                                    [[[0.0, 0.0]], [[2.15195436e-15, 0.0]], [[-0.09341003, -1.00516610e-02]]]], rtol=1e-6)
        np.testing.assert_allclose(self.pes_model1_2_nb._diabatic_energy_centroid,
                                   [[1.89603200e-03, 7.87338081e-07, 0.0],
                                    [7.87338081e-07, 1.00135655e-02, 7.87338081e-07],
                                    [0.0, 7.87338081e-07, 2.38370694e-02]], rtol=1e-6) 
        np.testing.assert_allclose(self.pes_model1_2_nb._diabatic_gradient_centroid,
                                   [[[-5.56530645e-03], [-1.76363730e-05], [ 0.00000000e+00]],
                                    [[-1.76363730e-05], [ 2.63251665e-04], [ 1.76363730e-05]],
                                    [[ 0.00000000e+00], [ 1.76363730e-05], [-3.26978665e-02]]], rtol=1e-6)

        # Model 2: 1 bead
        self.pes_model2_1_nb._calculate_diabatic_all(np.array([[3.34]]))
        np.testing.assert_allclose(self.pes_model2_1_nb._diabatic_energy,
                                   [[[0.02533433], [0.00018874], [0.005]],
                                    [[0.00018874], [0.01091281], [0.0]],
                                    [[0.005], [0.0], [0.02295054]]], rtol=1e-4)
        np.testing.assert_allclose(self.pes_model2_1_nb._diabatic_gradient,
                                   [[[[-0.06219724]], [[ 0.00386548]], [[0.0]]],
                                    [[[ 0.00386548]], [[-0.00314728]], [[0.0]]],
                                    [[[0.0]], [[0.0]], [[-0.00770342]]]] , rtol=1e-6)
        np.testing.assert_allclose(self.pes_model2_1_nb._diabatic_energy_centroid,
                                   [[0.02533433, 0.00018874, 0.005],
                                    [0.00018874, 0.01091281, 0.0],
                                    [0.005, 0.0, 0.02295054]] , rtol=1e-4)
        np.testing.assert_allclose(self.pes_model2_1_nb._diabatic_gradient_centroid,
                                   [[[-0.06219724], [ 0.00386548], [0.0]],
                                    [[ 0.00386548], [-0.00314728], [0.0]],
                                    [[0.0], [0.0], [-0.00770342]]] , rtol=1e-6)

        # Model 2: 2 beads
        self.pes_model2_2_nb._calculate_diabatic_all(np.array([[3.34, 3.66]]))
        np.testing.assert_allclose(self.pes_model2_2_nb._diabatic_energy,
                                   [[[0.02533433, 0.01055122], [0.00018874, 0.005], [0.005, 0.00018874]],
                                    [[0.00018874, 0.005], [0.01091281, 0.01021223], [0.0, 0.0]],
                                    [[0.005, 0.00018874], [0.0, 0.0], [0.02295054, 0.02114463]]], rtol=1e-4)
        np.testing.assert_allclose(self.pes_model2_2_nb._diabatic_gradient,
                                   [[[[-0.06219724, -0.03260126]], [[ 0.00386548, 0.0]], [[0.0, -0.00386548]]],
                                    [[[ 0.00386548, 0.0]], [[-0.00314728, -0.00133524]], [[0.0, 0.0]]],
                                    [[[0.0, -0.00386548]], [[0.0, 0.0]], [[-0.00770342, -0.00389701]]]] , rtol=1e-6)
        np.testing.assert_allclose(self.pes_model2_2_nb._diabatic_energy_centroid,
                                   [[0.0167643,  0.00220392, 0.00220392],
                                    [0.00220392, 0.01049019, 0.0],
                                    [0.00220392, 0.0, 0.02189603]] , rtol=1e-6)
        np.testing.assert_allclose(self.pes_model2_2_nb._diabatic_gradient_centroid,
                                   [[[-0.04559765], [ 0.02256815], [-0.02256815]],
                                    [[ 0.02256815], [-0.00216338], [ 0.0]],
                                    [[-0.02256815], [ 0.0], [-0.00556531]]] , rtol=1e-5)

        # Model 3: 1 bead
        self.pes_model3_1_nb._calculate_diabatic_all(np.array([[3.4]]))
        np.testing.assert_allclose(self.pes_model3_1_nb._diabatic_energy,
                                   [[[2.14715220e-02], [0.005], [2.77466793e-37]],
                                    [[0.005], [2.18065165e-02], [0.0]],
                                    [[2.77466793e-37], [0.0], [7.85954291e-02]]] , rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_1_nb._diabatic_gradient,
                                   [[[[-5.51720403e-03]], [[0.0]], [[ 2.78798633e-35]]],
                                    [[[0.0]], [[-5.54973253e-02]], [[0.0]]],
                                    [[[ 2.78798633e-35]], [[ 0.0]],[[-9.34100326e-02]]]] , rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_1_nb._diabatic_energy_centroid,
                                   [[2.14715220e-02, 0.005, 2.77466793e-37],
                                    [0.005, 2.18065165e-02, 0.0],
                                    [2.77466793e-37, 0.0, 7.85954291e-02]] , rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_1_nb._diabatic_gradient_centroid,
                                   [[[-5.51720403e-03], [0.0], [ 2.78798633e-35]],
                                    [[0.0], [-5.54973253e-02], [0.0]],
                                    [[ 2.78798633e-35], [ 0.0],[-9.34100326e-02]]], rtol=1e-7)

        # Model 3: 2 beads
        self.pes_model3_2_nb._calculate_diabatic_all(np.array([[3.4, 4.97]]))
        np.testing.assert_allclose(self.pes_model3_2_nb._diabatic_energy,
                                   [[[2.14715220e-02, 2.20683724e-02], [0.005, 2.77466793e-37], [2.77466793e-37, 0.005]],
                                    [[0.005, 2.77466793e-37], [2.18065165e-02, 1.38595960e-03], [0.0, 0.0]],
                                    [[2.77466793e-37, 0.005], [0.0, 0.0], [7.85954291e-02, 2.27261164e-02]]] , rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_2_nb._diabatic_gradient,
                                   [[[[-5.51720403e-03, 3.49070457e-03]], [[0.0, -2.78798633e-35]], [[ 2.78798633e-35, 0.0]]],
                                    [[[0.0, -2.78798633e-35]], [[-5.54973253e-02, 5.04262499e-03]], [[0.0, 0.0]]],
                                    [[[ 2.78798633e-35, 0.0]], [[0.0, 0.0]],[[-9.34100326e-02, -7.26166711e-03]]]] , rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_2_nb._diabatic_energy_centroid,
                                   [[2.01017545e-02, 1.36467821e-11, 1.36467821e-11],
                                    [1.36467821e-11, 1.03256241e-03, 0.0],
                                    [1.36467821e-11, 0.0, 3.52356076e-02]], rtol=1e-7)
        np.testing.assert_allclose(self.pes_model3_2_nb._diabatic_gradient_centroid,
                                   [[[ 1.05984926e-03], [-6.85614332e-10], [ 6.85614332e-10]],
                                    [[-6.85614332e-10], [-7.25000490e-03], [ 0.0]],
                                    [[ 6.85614332e-10], [ 0.0], [-2.85951767e-02]]], rtol=1e-7)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(MorseDiabaticTest)
    unittest.TextTestRunner().run(suite)
