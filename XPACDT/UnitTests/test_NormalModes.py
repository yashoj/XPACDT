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

import XPACDT.Tools.NormalModes as nm
import XPACDT.Tools.Units as units


class NormalModesTest(unittest.TestCase):

    def test_get_normal_modes(self):
        hessian = np.loadtxt("FilesForTesting/NormalModes/H2O/hess")
        to_cm = units.nm_to_cm # centimeter / lightspeed / 2.0 / np.pi
        mh = units.atom_mass('H')
        mo = units.atom_mass('O')
        mass = np.array([mh]*3 + [mo]*3 + [mh]*3)

        # References obtained with Molcas
        omega_ref = np.loadtxt("FilesForTesting/NormalModes/H2O/freq")
        cartesian_ref = np.loadtxt("FilesForTesting/NormalModes/H2O/modes")

        omega, mode_masses, vec, cartesian = nm.get_normal_modes(hessian, mass)

        np.testing.assert_allclose(omega*to_cm, omega_ref, atol=1e0)
        for c, c_ref in zip(cartesian.T, cartesian_ref.T):
            self.assertTrue(
                    (abs(c-c_ref) < 2e-4).all() or (abs(c+c_ref) < 2e-4).all())

        return

    def test_transform_to_cartesian(self):
        hessian = np.loadtxt("FilesForTesting/NormalModes/H2O/hess")
        mh = units.atom_mass('H')
        mo = units.atom_mass('O')
        mass = np.array([mh]*3 + [mo]*3 + [mh]*3)

        omega, mode_masses, vec, cartesian = nm.get_normal_modes(hessian, mass)

        x0 = np.loadtxt("FilesForTesting/NormalModes/H2O/x0")

        # TODO: put ref to file?
        x_ref = np.array([
                [1.4326, 0.2294, 0.0000+0.5774, 0.0000, 1.4309, 0.0000+0.5774, -1.4326, 0.2294, 0.0000+0.5774],
                [1.4326-0.0001, 0.2294-0.5774, 0.0000, 0.0000-0.0001, 1.4309-0.5774, 0.0000, -1.4326-0.0001, 0.2294-0.5774, 0.0000],
                [1.4326+0.5774, 0.2294-0.0001, 0.0000, 0.0000+0.5774, 1.4309-0.0001, 0.0000, -1.4326+0.5774, 0.2294-0.0001, 0.0000],
                [1.4326, 0.2294, 0.0000+0.7043, 0.0000, 1.4309, 0.0000-0.0888, -1.4326, 0.2294, 0.0000+0.7043],
                [1.4326+0.4218, 0.2294+0.5663, 0.0000, 0.0000-0.0531, 1.4309, 0.0000, -1.4326+0.4218, 0.2294-0.5663, 0.0000],
                [1.4326, 0.2294, 0.0000+0.7071, 0.0000, 1.4309, 0.0000, -1.4326, 0.2294, 0.0000-0.7071],
                [1.4326+0.4480, 0.2294+0.5449, 0.0000, 0.0000, 1.4309-0.0687, 0.0000, -1.4326-0.4480, 0.2294+0.5449, 0.0000],
                [1.4326+0.5703, 0.2294-0.4164, 0.0000, 0.0000, 1.4309+0.0525, 0.0000, -1.4326-0.5703, 0.2294-0.4164, 0.0000],
                [1.4326-0.5405, 0.2294+0.4533, 0.0000, 0.0000+0.0681, 1.4309, 0.0000, -1.4326-0.5405, 0.2294-0.4533, 0.0000]])
        p_ref = np.array([
                [0.0000, 0.0000, 0.0000+0.5774, 0.0000, 0.0000, 0.0000+0.5774, -0.0000, 0.0000, 0.0000+0.5774],
                [0.0000+0.0001, 0.0000-0.5774, 0.0000, 0.0000-0.0001, 0.0000-0.5774, 0.0000, -0.0000-0.0001, 0.0000-0.5774, 0.0000],
                [0.0000+0.5774, 0.0000-0.0001, 0.0000, 0.0000+0.5774, 0.0000-0.0001, 0.0000, -0.0000+0.5774, 0.0000-0.0001, 0.0000],
                [0.0000, 0.0000, 0.0000+0.7043, 0.0000, 0.0000, 0.0000-0.0888, -0.0000, 0.0000, 0.0000+0.7043],
                [0.0000+0.4218, 0.0000+0.5663, 0.0000, 0.0000-0.0531, 0.0000, 0.0000, -0.0000+0.4218, 0.0000-0.5663, 0.0000],
                [0.0000, 0.0000, 0.0000+0.7071, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000-0.7071],
                [0.0000+0.4480, 0.0000+0.5449, 0.0000, 0.0000, 0.0000-0.0687, 0.0000, -0.0000-0.4480, 0.0000+0.5449, 0.0000],
                [0.0000+0.5703, 0.0000-0.4164, 0.0000, 0.0000, 0.0000+0.0525, 0.0000, -0.0000-0.5703, 0.0000-0.4164, 0.0000],
                [0.0000-0.5405, 0.0000+0.4533, 0.0000, 0.0000+0.0681, 0.0000, 0.0000, -0.0000-0.5405, 0.0000-0.4533, 0.0000]])
        for i in range(9):
            x_nm = np.zeros(9)
            p_nm = np.zeros(9)
            x_nm[i] = 1.0
            p_nm[i] = 1.0

            x, p = nm.transform_to_cartesian(x_nm, p_nm, x0, cartesian)
            np.testing.assert_allclose(x, x_ref[i], rtol=1e-7, atol=3e-4)
            np.testing.assert_allclose(p, p_ref[i], rtol=1e-7, atol=3e-4)

        x_nm = np.eye(9)
        p_nm = np.eye(9)
        x, p = nm.transform_to_cartesian(x_nm, p_nm, x0, cartesian)

        np.testing.assert_allclose(x, x_ref, rtol=1e-7, atol=3e-4)
        np.testing.assert_allclose(p, p_ref, rtol=1e-7, atol=3e-4)

        return

# TODO: not tested yet due to moving it around
    def test_get_sampling_modes(self):
        
        raise NotImplementedError("Please implement a test here!!")
        pass


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NormalModesTest)
    unittest.TextTestRunner().run(suite)
