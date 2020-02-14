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

        self.nuclei_classical = Nuclei.Nuclei(self.parameters_classical, 0.0)
        self.nuclei_rpmd = Nuclei.Nuclei(self.parameters_rpmd, 0.0)

        self.parameters_classical_1D = infile.Inputfile("FilesForTesting/SystemTests/Classical_1D.in")
        self.parameters_rpmd_1D = infile.Inputfile("FilesForTesting/SystemTests/RPMD_1D.in")

        self.nuclei_classical_1D = Nuclei.Nuclei(self.parameters_classical_1D, 0.0)
        self.nuclei_rpmd_1D = Nuclei.Nuclei(self.parameters_rpmd_1D, 0.0)
        pass

    def test_propagate(self):
        ### 1 bead case
        # Set up dummy propagator
        self.nuclei_classical.propagator = DummyProp(1.0)
        self.nuclei_classical.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_classical.positions,
                                      self.parameters_classical.coordinates*2.0)
        np.testing.assert_array_equal(self.nuclei_classical.momenta,
                                      self.parameters_classical.momenta*2.0)

        # Reset nuclei
        self.nuclei_classical = Nuclei.Nuclei(self.parameters_classical,
                                              0.0)
        self.nuclei_classical.propagator = DummyProp(1.0)
        self.nuclei_classical.propagate(5.0)
        coordinates_ref = np.copy(self.parameters_classical.coordinates)
        momenta_ref = np.copy(self.parameters_classical.momenta)

        for i in range(5):
            coordinates_ref *= 2.0
            momenta_ref *= 2.0

        np.testing.assert_array_equal(self.nuclei_classical.positions,
                                      coordinates_ref)
        np.testing.assert_array_equal(self.nuclei_classical.momenta,
                                      momenta_ref)

        with self.assertRaises(RuntimeError):
            self.nuclei_classical.propagate(1.5)

        ### 4 bead case
        self.nuclei_rpmd.propagator = DummyProp(1.0)
        self.nuclei_rpmd.propagate(1.0)

        np.testing.assert_array_equal(self.nuclei_rpmd.positions,
                                      self.parameters_rpmd.coordinates*2.0)
        np.testing.assert_array_equal(self.nuclei_rpmd.momenta,
                                      self.parameters_rpmd.momenta*2.0)

        # Reset nuclei
        self.nuclei_rpmd = Nuclei.Nuclei(self.parameters_rpmd, 0.0)
        self.nuclei_rpmd.propagator = DummyProp(1.0)
        self.nuclei_rpmd.propagate(5.0)
        coordinates_ref = np.copy(self.parameters_rpmd.coordinates)
        momenta_ref = np.copy(self.parameters_rpmd.momenta)

        for i in range(5):
            coordinates_ref *= 2.0
            momenta_ref *= 2.0

        np.testing.assert_array_equal(self.nuclei_rpmd.positions,
                                      coordinates_ref)
        np.testing.assert_array_equal(self.nuclei_rpmd.momenta,
                                      momenta_ref)

        with self.assertRaises(RuntimeError):
            self.nuclei_rpmd.propagate(1.5)

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

    def test_get_selected_quantities(self):
        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.get_selected_quantities("m,0,1")

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.get_selected_quantities("m,0,1", quantity='p')

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.get_selected_quantities("m,0,1", quantity='v')

            self.nuclei_classical.get_selected_quantities("m,0,1", beads=True)

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.get_selected_quantities("m,0,1", quantity='p', beads=True)

        with self.assertRaises(NotImplementedError):
            self.nuclei_classical.get_selected_quantities("m,0,1", quantity='v', beads=True)

        with self.assertRaises(RuntimeError):
            self.nuclei_classical.get_selected_quantities("0,2", quantity='y')

        values_ref = np.array([2.0, 4.0])
        values = self.nuclei_classical.get_selected_quantities("0,2")
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([0.05, 1.25/2.1])
        values = self.nuclei_classical.get_selected_quantities("1,3", quantity='v')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([-1.0, 2.0, 1.25])
        values = self.nuclei_classical.get_selected_quantities("0,2,3", quantity='p')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[2.0], [4.0]])
        values = self.nuclei_classical.get_selected_quantities("0,2", beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[0.05], [1.25/2.1]])
        values = self.nuclei_classical.get_selected_quantities("1,3", quantity='v', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[-1.0], [2.0], [1.25]])
        values = self.nuclei_classical.get_selected_quantities("0,2,3", quantity='p', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        # RPMD
        values_ref = np.array([8.4/4, 1.9/4])
        values = self.nuclei_rpmd.get_selected_quantities("0,2")
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([-0.3/4/2, 0.85/4/2.1])
        values = self.nuclei_rpmd.get_selected_quantities("1,3", quantity='v')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([2.2/4, 7.4/4, 0.85/4])
        values = self.nuclei_rpmd.get_selected_quantities("0,2,3", quantity='p')
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[2.0, 3.0, 1.0, 2.4], [4.0, -2.0, -0.1, 0.0]])
        values = self.nuclei_rpmd.get_selected_quantities("0,2", beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[0.05, -0.05, -0.25, 0.1], [1.25/2.1, -0.5/2.1, 0.1/2.1, 0.0]])
        values = self.nuclei_rpmd.get_selected_quantities("1,3", quantity='v', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        values_ref = np.array([[-1.0, 1.2, 2.0, 0.0], [2.0, -0.1, 2.5, 3.0], [1.25, -0.5, 0.1, 0.0]])
        values = self.nuclei_rpmd.get_selected_quantities("0,2,3", quantity='p', beads=True)
        np.testing.assert_array_equal(values_ref, values)

        return

    def test_energy(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.energy, 0)

        self.nuclei_classical_1D.positions = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.energy, 0.5)

        self.nuclei_classical_1D.momenta = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.energy, 1)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.energy, 15.5)

        # TODO: add test for more that 1 dimensions
        return

    def test_kinetic_energy(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.kinetic_energy, 0)

        self.nuclei_classical_1D.momenta = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.kinetic_energy, 0.5)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.kinetic_energy, 7)

        # 1 bead, 4 dof
        KE_ref = 0.5025 + 1./6. + 1.5625/4.2
        self.assertAlmostEqual(self.nuclei_classical.kinetic_energy, KE_ref)

        # 4 beads, 4 dof
        KE_ref = 6.44 * 0.5 + 0.25 * 0.31 + 19.26/24. + 1.8225/4.2
        self.assertAlmostEqual(self.nuclei_rpmd.kinetic_energy, KE_ref)
        return

    def test_spring_energy(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.spring_energy, 0)

        self.nuclei_classical_1D.positions = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.spring_energy, 0)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.spring_energy, 1.5)

        # 1 bead, 4 dof
        self.assertAlmostEqual(self.nuclei_classical.spring_energy, 0)

        # 4 beads, 4 dof
        SE_ref = 0.125 * (7.12 + 24.52 + 667.44 + 52.878)
        self.assertAlmostEqual(self.nuclei_rpmd.spring_energy, SE_ref)
        return

    def test_potential_energy(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.potential_energy, 0)

        self.nuclei_classical_1D.positions = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.potential_energy, 0.5)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.potential_energy, 7)

        # TODO: add test for more that 1 dimensions
        return

    def test_energy_centroid(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.energy_centroid, 0)

        self.nuclei_classical_1D.positions = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.energy_centroid, 0.5)

        self.nuclei_classical_1D.momenta = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.energy_centroid, 1)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.energy_centroid, 2.25)

        # TODO: add test for more that 1 dimensions
        return

    def test_kinetic_energy_centroid(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.kinetic_energy_centroid, 0)

        self.nuclei_classical_1D.momenta = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.kinetic_energy_centroid, 0.5)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.kinetic_energy_centroid, 1.125)

        # 1 bead, 4 dof
        KE_ref = 0.5025 + 1./6. + 1.5625/4.2
        self.assertAlmostEqual(self.nuclei_classical.kinetic_energy_centroid,
                               KE_ref)

        # 4 beads, 4 dof
        KE_ref = 1. / 16. * (2.42 + 0.09 / 4. + 7.4**2 / 24. + 0.85**2 / 4.2)
        self.assertAlmostEqual(self.nuclei_rpmd.kinetic_energy_centroid,
                               KE_ref)

        return

    def test_potential_energy_centroid(self):
        # 1 bead, 1 dof
        self.assertAlmostEqual(self.nuclei_classical_1D.potential_energy_centroid, 0)

        self.nuclei_classical_1D.positions = np.array([[1.]])
        self.assertAlmostEqual(self.nuclei_classical_1D.potential_energy_centroid, 0.5)

        # 4 beads, 1 dof
        self.assertAlmostEqual(self.nuclei_rpmd_1D.potential_energy_centroid,
                               1.125)

        # TODO: add test for more that 1 dimensions
        return

    @unittest.skip("Please implement a test here.")
    def test_init_electrons(self):
        raise NotImplementedError("Please implement a test here!!")

    @unittest.skip("Please implement a test here.")
    def test_attach_nuclei_propagator(self):
        raise NotImplementedError("Please implement a test here!!")

    @unittest.skip("Please implement a test here.")
    def test_getCOM(self):
        raise NotImplementedError("Please implement a test here while"
                                  " implmenting the function!!")

    @unittest.skip("Please implement a test here.")
    def test_print_size(self):
        # Not really clear how to test this. Here for completness sake.
        pass


class DummyProp(object):
    def __init__(self, timestep):
        self.timestep = float(timestep)
        pass

    def propagate(self, R, P, time_propagation):
        return (2.0 * R), (2.0 * P)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NucleiTest)
    unittest.TextTestRunner().run(suite)
