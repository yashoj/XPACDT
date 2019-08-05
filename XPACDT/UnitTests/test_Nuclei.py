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

import unittest
import numpy as np

import XPACDT.System.Nuclei as Nuclei
import XPACDT.Input.Inputfile as infile


class NucleiTest(unittest.TestCase):

    def setUp(self):
        self.parameters_classical = infile.Inputfile("FilesForTesting/SystemTests/Classical.in")
        self.parameters_rpmd = infile.Inputfile("FilesForTesting/SystemTests/RPMD.in")

        self.nuclei_classical = Nuclei.Nuclei(4, self.parameters_classical, 0.0)
        self.nuclei_rpmd = Nuclei.Nuclei(4, self.parameters_rpmd, 0.0)
        pass

    def test_propagate(self):
        pass
        # Set up dummy propagator
        self.nuclei_classical.propagator = DummyProp()
        self.nuclei_classical.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_classical.positions,
                                      self.parameters_classical.coordinates*2.0)
        np.testing.assert_array_equal(self.nuclei_classical.momenta,
                                      self.parameters_classical.momenta*2.0)

        self.nuclei_classical.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_classical.positions,
                                      self.parameters_classical.coordinates*4.0)
        np.testing.assert_array_equal(self.nuclei_classical.momenta,
                                      self.parameters_classical.momenta*4.0)

        self.nuclei_rpmd.propagator = DummyProp()
        self.nuclei_rpmd.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_rpmd.positions,
                                      self.parameters_rpmd.coordinates*2.0)
        np.testing.assert_array_equal(self.nuclei_rpmd.momenta,
                                      self.parameters_rpmd.momenta*2.0)

        self.nuclei_rpmd.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_rpmd.positions,
                                      self.parameters_rpmd.coordinates*4.0)
        np.testing.assert_array_equal(self.nuclei_rpmd.momenta,
                                      self.parameters_rpmd.momenta*4.0)

        return

    def test_x_centroid(self):
        classical_centroid = np.array([2.0, 1.0, 4.0, -2.0])
        rpmd_centroid = np.array([2.1, 0.625, 0.475, 0.325])

        np.testing.assert_array_equal(self.nuclei_classical.x_centroid,
                                      classical_centroid)
        np.testing.assert_array_almost_equal(self.nuclei_rpmd.x_centroid,
                                             rpmd_centroid)

        return

    def test_p_centroid(self):
        classical_centroid = np.array([-1.0, 0.1, 2.0, 1.25])
        rpmd_centroid = np.array([0.55, -0.075, 1.85, 0.2125])

        np.testing.assert_array_equal(self.nuclei_classical.p_centroid,
                                      classical_centroid)
        np.testing.assert_array_almost_equal(self.nuclei_rpmd.p_centroid,
                                             rpmd_centroid)
        pass

    def test_parse_dof(self):
        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.parse_dof("m,0,1")

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.parse_dof("m,0,1", quantity='p')

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.parse_dof("m,0,1", quantity='v')

            self.nuclei_classical.parse_dof("m,0,1", beads=True)

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.parse_dof("m,0,1", quantity='p', beads=True)

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.parse_dof("m,0,1", quantity='v', beads=True)

        with self.assertRaises(RuntimeError):
            self.nuclei_classical.parse_dof("0,2", quantity='y')

        values_ref = np.array([2.0, 4.0])
        values = self.nuclei_classical.parse_dof("0,2")
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([0.05, 1.25/2.1])
        values = self.nuclei_classical.parse_dof("1,3", quantity='v')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([-1.0, 2.0, 1.25])
        values = self.nuclei_classical.parse_dof("0,2,3", quantity='p')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[2.0], [4.0]])
        values = self.nuclei_classical.parse_dof("0,2", beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[0.05], [1.25/2.1]])
        values = self.nuclei_classical.parse_dof("1,3", quantity='v', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[-1.0], [2.0], [1.25]])
        values = self.nuclei_classical.parse_dof("0,2,3", quantity='p', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        # RPMD
        values_ref = np.array([8.4/4, 1.9/4])
        values = self.nuclei_rpmd.parse_dof("0,2")
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([-0.3/4/2, 0.85/4/2.1])
        values = self.nuclei_rpmd.parse_dof("1,3", quantity='v')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([2.2/4, 7.4/4, 0.85/4])
        values = self.nuclei_rpmd.parse_dof("0,2,3", quantity='p')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[2.0, 3.0, 1.0, 2.4], [4.0, -2.0, -0.1, 0.0]])
        values = self.nuclei_rpmd.parse_dof("0,2", beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[0.05, -0.05, -0.25, 0.1], [1.25/2.1, -0.5/2.1, 0.1/2.1, 0.0]])
        values = self.nuclei_rpmd.parse_dof("1,3", quantity='v', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[-1.0, 1.2, 2.0, 0.0], [2.0, -0.1, 2.5, 3.0], [1.25, -0.5, 0.1, 0.0]])
        values = self.nuclei_rpmd.parse_dof("0,2,3", quantity='p', beads=True)
        np.testing.assert_array_equal(values_ref, values)

    def test_energy(self):
        raise NotImplementedError("Please implement a test here once "
                                  "the function is implemented!!")
        pass

    def test_kinetic_energy(self):
        raise NotImplementedError("Please implement a test here once "
                                  "the function is implemented!!")
        pass

    def test_spring_energy(self):
        raise NotImplementedError("Please implement a test here once "
                                  "the function is implemented!!")
        pass

    def test_potential_energy(self):
        raise NotImplementedError("Please implement a test here once "
                                  "the function is implemented!!")
        pass


class DummyProp(object):
    def __init__(self):
        pass

    def propagate(self, R, P, time_propagation):
        return 2.0*R, 2.0*P


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NucleiTest)
    unittest.TextTestRunner().run(suite)
