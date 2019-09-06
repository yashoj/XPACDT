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

import XPACDT.Dynamics.VelocityVerlet as vv
import XPACDT.System.AdiabaticElectrons as adiabatic


class VelocityVerletTest(unittest.TestCase):

    def setUp(self):
        self.pes1D_harmonic_classical = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5"}}, [1])
        self.pes1D_shifted_harmonic_classical = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5", 'x0': '1.0'}}, [1])
        self.pes1D_anharmonic_classical = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5 0.1 0.01"}}, [1])
        self.pes1D_quartic_classical = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.0 0.0 0.25"}}, [1])

        self.pes1D_harmonic_4_nb = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5"}}, [4])
        self.pes1D_shifted_harmonic_4_nb = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5", 'x0': '1.0'}}, [4])
        self.pes1D_anharmonic_4_nb = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.5 0.1 0.01"}}, [4])
        self.pes1D_quartic_4_nb = adiabatic.AdiabaticElectrons(
                {'system': {'Interface': 'OneDPolynomial'},
                 'OneDPolynomial': {'a': "0.0 0.0 0.0 0.0 0.25"}}, [4])

        # TODO: also multi-D potential for more testing
        return

    def test_propagate(self):
        # Classical
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25]])
        r = np.array([[0.5]])

        # Single time step
        p_ref = np.array([[0.148]])
        r_ref = np.array([[0.52]])
        rt, pt = propagator.propagate(r, p, 0.2)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

        # Two time steps
        p_ref = np.array([[0.04304]])
        r_ref = np.array([[0.5296]])
        rt, pt = propagator.propagate(r, p, 0.4)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

        ###############

        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        # Single time step
        p_ref = np.array([[0.09494187, -0.58829502,  0.53812469,  0.25122846]])
        r_ref = np.array([[0.51738778,  0.95774543,  0.05227955, -0.48741276]])
        rt, pt = propagator.propagate(r, p, 0.2)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)
        
        # Two time steps
        p_ref = np.array([[-0.06724218, -0.89683408,  0.55144043, 0.49871582]])
        r_ref = np.array([[0.51881322,  0.88294329,  0.10715463, -0.44971114]])
        rt, pt = propagator.propagate(r, p, 0.4)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

        return

    def test_step(self):
        # classical
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.148]])
        r_ref = np.array([[0.52]])
        rt, pt = propagator._step(r, p)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.09494187, -0.58829502,  0.53812469,  0.25122846]])
        r_ref = np.array([[0.51738778,  0.95774543,  0.05227955, -0.48741276]])
        rt, pt = propagator._step(r, p)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)
        return

    def test_verlet_step(self):
        # classical
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.25]])
        r_ref = np.array([[0.525]])
        rt, pt = propagator._verlet_step(r, p)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-7)

        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.19643, -0.39327, 0.54360, 0.15324]])
        r_ref = np.array([[0.522379, 0.96772, 0.052288, -0.492388]])
        rt, pt = propagator._verlet_step(r, p)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-4)
        np.testing.assert_allclose(rt, r_ref, rtol=1e-4)

        return

    def test_velocity_step(self):
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_classical, np.array([2.0]), [1],
                                       **{'timestep': '0.2 au'})

        ###############
        # classical
        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.2]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        propagator.beta = 8.0
        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_4_nb, np.array([2.0]), [4],
                                       **{'timestep': '0.2 au'})
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.2, -0.35, 0.5, 0.05]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        ###############

        propagator = vv.VelocityVerlet(self.pes1D_shifted_harmonic_classical,
                                       np.array([2.0]), [1],
                                       **{'timestep': '0.2 au'})

        # classical
        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.3]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        propagator.beta = 8.0
        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_shifted_harmonic_4_nb,
                                       np.array([2.0]), [4],
                                       **{'timestep': '0.2 au'})
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.3, -0.25, 0.6, 0.15]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        ###############

        propagator = vv.VelocityVerlet(self.pes1D_anharmonic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})
        # classical
        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.192]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_anharmonic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.192, -0.384, 0.5, 0.043]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        ###############

        propagator = vv.VelocityVerlet(self.pes1D_quartic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        # classical
        p = np.array([[0.25]])
        r = np.array([[0.5]])

        p_ref = np.array([[0.2375]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        # 4 beads
        propagator = vv.VelocityVerlet(self.pes1D_quartic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})
        p = np.array([[0.25, -0.25, 0.5, 0.0]])
        r = np.array([[0.5, 1.0, 0.0, -0.5]])

        p_ref = np.array([[0.2375, -0.35, 0.5, 0.0125]])
        pt = propagator._velocity_step(p, r)
        np.testing.assert_allclose(pt, p_ref, rtol=1e-7)

        return

    def test_set_propagation_matrix(self):
        propagator = vv.VelocityVerlet(self.pes1D_harmonic_classical, np.array([2.0]), [1],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        classical_ref = np.array([[[[1.0, 0.0], [0.1, 1.0]]]])
        propagator._set_propagation_matrix()
        classical_pm = propagator.propagation_matrix
        np.testing.assert_allclose(classical_pm, classical_ref, rtol=1e-7)

        propagator = vv.VelocityVerlet(self.pes1D_harmonic_4_nb, np.array([2.0]), [4],
                                       **{'beta': 8.0, 'timestep': '0.2 au'})

        four_beads_ref = np.array([[[[1.0, 0.0], [0.1, 1.0]],
                                   [[0.9900, -0.19933], [0.099667, 0.9900]],
                                   [[0.9801, -0.3973], [0.09933, 0.9801]],
                                   [[0.9900, -0.19933], [0.099667, 0.9900]]]])
        propagator._set_propagation_matrix()
        four_beads_pm = propagator.propagation_matrix
        np.testing.assert_allclose(four_beads_pm, four_beads_ref, rtol=1e-4)

        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(VelocityVerletTest)
    unittest.TextTestRunner().run(suite)
