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
import unittest
from scipy import stats

import XPACDT.Dynamics.MassiveAndersenThermostat as ma
import XPACDT.Dynamics.VelocityVerletPropagator as vv
import XPACDT.System.AdiabaticElectrons as adiabatic
import XPACDT.Input.Inputfile as infile


class MassiveAndersenTest(unittest.TestCase):

    def test_apply(self):
        # Check if apply function works in redrawing samples or doing nothing;
        # resampling is checked by comparing to obtained seed random numbers

        # 1 dof, 1 bead, beta = 1.0
        np.random.seed(0)
        input_params = {'thermostat': {'temperature': '315775.130734', 'time': '1.0 au'}}
        mass = np.array([1.])
        thermostat = ma.MassiveAndersen(input_params, mass)
        x = np.array([[1.0]])
        p = np.array([[0.5]])

        # Apply without change
        x_ref = np.array([[1.0]])
        p_ref = np.array([[0.5]])

        thermostat.apply(x, p, 1, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        thermostat.apply(x, p, 1, 0.5)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        thermostat.apply(x, p, 0, 0.5)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        # Apply with change
        x_ref = np.array([[1.0]])
        p_ref = np.array([[1.76405235]])

        thermostat.apply(x, p, 0, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        ###############

        # 1 dof, 4 beads, beta = 1.0
        np.random.seed(0)
        input_params = {'thermostat': {'temperature': '315775.130734', 'time': '1.0 au'}}
        mass = np.array([1.])
        thermostat = ma.MassiveAndersen(input_params, mass)
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        p = np.array([[0.5, 0., -0.5, 1.]])

        # Apply without change
        x_ref = np.array([[1.0, 2.0, 3.0, 4.0]])
        p_ref = np.array([[0.5, 0., -0.5, 1.]])

        thermostat.apply(x, p, 1, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        # Apply with change
        x_ref = np.array([[1.0, 2.0, 3.0, 4.0]])
        p_ref = np.array([[3.52810469, 0.80031442, 1.95747597, 4.4817864]])

        thermostat.apply(x, p, 0, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        ###############

        # 2 dof, 4 beads, beta = 1.0
        np.random.seed(0)
        input_params = {'thermostat': {'temperature': '315775.130734', 'time': '1.0 au'}}
        mass = np.array([1., 1.])
        thermostat = ma.MassiveAndersen(input_params, mass)
        x = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]])
        p = np.array([[0.5, 0., -0.5, 1.], [1.5, 1., -1.5, 2.]])

        # Apply without change
        x_ref = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]])
        p_ref = np.array([[0.5, 0., -0.5, 1.], [1.5, 1., -1.5, 2.]])

        thermostat.apply(x, p, 1, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

        # Apply with change
        x_ref = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]])
        p_ref = np.array([[3.52810469, 0.80031442, 1.95747597, 4.4817864],
                          [3.73511598, -1.95455576, 1.90017684, -0.30271442]])

        thermostat.apply(x, p, 0, 1.0)
        np.testing.assert_allclose(x, x_ref, rtol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7)

    def test_with_velocity_verlet_step(self):
        # Check if attaching thermostat doesn't affect anything in '_step'
        # function in velocity verlet

        # 1 dof, 4 beads, beta = 8.0
        pes1D_harmonic = adiabatic.AdiabaticElectrons(infile.Inputfile("FilesForTesting/SystemTests/harmonic_4.in"))
        mass = np.array([2.])
        input_params = {'thermostat': {'method': 'MassiveAndersen',
                                       'temperature': '39471.891342', 'time': '1.0 au'}}

        propagator = vv.VelocityVerlet(pes1D_harmonic, mass, [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})
        propagator.attach_thermostat(input_params, mass)
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])
        # Reference values from test of velocity verlet as there should be no
        # changes
        p_ref = np.array([[0.09494187, -0.58829502,  0.53812469,  0.25122846]])
        r_ref = np.array([[0.51738778,  0.95774543,  0.05227955, -0.48741276]])
        rt, pt = propagator._step(r, p, 1.0)

        self.assertIsInstance(propagator.thermostat, ma.MassiveAndersen)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

    def test_with_velocity_verlet_propagate(self):
        # Check if attaching thermostat resamples properly in 'propagate'
        # function in velocity verlet

        # 1 dof, 4 beads, beta = 8.0
        np.random.seed(0)
        pes1D_harmonic = adiabatic.AdiabaticElectrons(infile.Inputfile("FilesForTesting/SystemTests/harmonic_4.in"))
        mass = np.array([2.])
        input_params = {'thermostat': {'method': 'MassiveAndersen',
                                       'temperature': '39471.891342', 'time': '1.0 au'}}

        propagator = vv.VelocityVerlet(pes1D_harmonic, mass, [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})
        propagator.attach_thermostat(input_params, mass)
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])
        rt, pt = propagator.propagate(r, p, 0.2, 0.8)

        # Get p reference values from thermostat using same seed since p values
        # are reset after propagation
        np.random.seed(0)
        thermostat = ma.MassiveAndersen(input_params, mass)
        thermostat.apply(r, p, 0, 1.0)

        p_ref = p.copy()
        # r reference values are unchanged from test of velocity verlet
        r_ref = np.array([[0.51738778, 0.95774543, 0.05227955, -0.48741276]])

        self.assertIsInstance(propagator.thermostat, ma.MassiveAndersen)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

    def test_distribution(self):
        seed = 0
        np.random.seed(seed)

        # This is more of an integrated test to check sampling!!!
        # Test for proper distribution of momenta
        # 1 dof, beta = 1.0
        input_params = {'thermostat': {'temperature': '315775.130734', 'time': '1.0 au'}}
        mass = np.array([1.])
        samples = 10000
        nb = 4
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        p = np.array([[0.5, 0., -0.5, 1.]])

        p_mean_ref = 0.0
        p_std_ref = 2.0
        x_mean_ref = np.array([[1.0, 2.0, 3.0, 4.0]])
        x_std_ref = np.array([[0., 0., 0., 0.]])

        p_arr = []
        x_arr = []
        thermostat = ma.MassiveAndersen(input_params, mass)

        for i in range(samples):
            thermostat.apply(x, p, 0, 1.0)
            p_arr.append(p.copy())
            x_arr.append(x.copy())

        p_arr = np.array(p_arr)
        x_arr = np.array(x_arr)

        np.testing.assert_allclose(np.mean(x_arr, axis=0), x_mean_ref,
                                   rtol=1e-7)
        np.testing.assert_allclose(np.std(x_arr, axis=0), x_std_ref, rtol=1e-7)

        for i in range(nb):
            mean, _, std = stats.bayes_mvs(p_arr[:, 0, i], alpha=0.95)
            mean_min, mean_max = mean[1]
            std_min, std_max = std[1]
            self.assertTrue(mean_min < p_mean_ref < mean_max)
            self.assertTrue(std_min < p_std_ref < std_max)

    def test_generation(self):
        # test temperature consistency check
        input_params = {'thermostat': {'temperature': '315775.130734'},
                        'sampling': {'temperature': '1.0', 'time': '1.0 au'}}
        mass = np.array([1.])
        with self.assertRaises(RuntimeError):
            ma.MassiveAndersen(input_params, mass)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(MassiveAndersenTest)
    unittest.TextTestRunner().run(suite)
