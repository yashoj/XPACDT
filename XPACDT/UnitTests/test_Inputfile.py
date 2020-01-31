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

import XPACDT.Input.Inputfile as infile


class InputfileTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        np.random.seed(seed)

    def test_creation(self):
        with self.assertRaises(FileNotFoundError):
            infile.Inputfile("input.in")

        infile.Inputfile("FilesForTesting/InputfileTest/input_minimal.in")

        return

    def test_parse_file(self):
        with self.assertRaises(KeyError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleKey.in")

        with self.assertRaises(ValueError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleEqual.in")

        input_reference = {"system": {"miep": "muh", "blah": "", "blubb": "",
                                      "dof": "4"},
                           "trajectory": {"blubb": "1.3 fs"},
                           "pes": {"blibb": "1.3 fs", "hot": "",
                                   "temp": "300 K"},
                           "miep": {"": ""}}
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_works.in")
        self.assertDictEqual(input_reference, parameters.store)

        # TODO: add a test for 'commands'

        return

    def test_parse_values(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_minimal.in")

        key_value_reference = {"miep": "kuh"}
        key_value = parameters._parse_values("miep = kuh")
        self.assertDictEqual(key_value_reference, key_value)

        key_value_reference = {"miep": "kuh ist langer string."}
        key_value = parameters._parse_values("miep = kuh ist langer string.")
        self.assertDictEqual(key_value_reference, key_value)

        key_value_reference = {"muh": ""}
        key_value = parameters._parse_values("muh ")
        self.assertDictEqual(key_value_reference, key_value)

        with self.assertRaises(ValueError):
            key_value = parameters._parse_values("mehrere = ist = doof")

        return

    def test_parse_xyz(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-1.in")

        mass_ref = np.array([1837.152646, 1837.152646, 1837.152646,
                             34631.970366, 34631.970366, 34631.970366])

        coordinate_ref = np.array([[1.0], [2.0], [3.0], [2.0], [1.0], [4.0]])
        input_string = "H 1.0 2.0 3.0 \n" \
            + "F 2.0 1.0 4.0 \n"

        parameters._parse_xyz(input_string)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-4)

        # with two beads
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-2.in")
        coordinate_ref = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1],
                                   [2.0, 2.1], [1.0, 1.1], [4.0, 4.1]])
        input_string = "H 1.0 2.0 3.0 \n" \
            + "H 1.1 2.1 3.1 \n" \
            + "F 2.0 1.0 4.0 \n" \
            + "F 2.1 1.1 4.1 \n"

        parameters._parse_xyz(input_string)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-4)

        # with four beads
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-4.in")
        coordinate_ref = np.array([[1.0, 1.1, 1.2, 1.3],
                                   [2.0, 2.1, 2.2, 2.3],
                                   [3.0, 3.1, 3.2, 3.3],
                                   [2.4, 2.5, 2.6, 2.7],
                                   [1.4, 1.5, 1.6, 1.7],
                                   [4.0, 4.1, 4.2, 4.3]])
        input_string = "H 1.0 2.0 3.0 \n" \
            + "H 1.1 2.1 3.1 \n" \
            + "H 1.2 2.2 3.2 \n" \
            + "H 1.3 2.3 3.3 \n" \
            + "F 2.4 1.4 4.0 \n" \
            + "F 2.5 1.5 4.1 \n" \
            + "F 2.6 1.6 4.2 \n" \
            + "F 2.7 1.7 4.3 \n"

        parameters._parse_xyz(input_string)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-4)

        # test unknwon element
        input_string = "J 1.0 2.0 3.0 \n" \
            + "F 2.0 1.0 4.0 \n"

        with self.assertRaises(KeyError):
            parameters._parse_xyz(input_string)

        # test too many/few coordinates given
        input_string = "H 1.0 2.0 \n" \
            + "F 2.0 1.0 4.0 \n"
        with self.assertRaises(ValueError):
            parameters._parse_xyz(input_string)

        input_string = "H 1.0 2.0 3.0\n" \
            + "F 2.0 1.0 4.0 6.0\n"
        with self.assertRaises(ValueError):
            parameters._parse_xyz(input_string)

        pass

    def test_parse_mass_value(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_2-4.in")
        # 4 beads test

        mass_ref = np.array([1837.3624, 34631.9731])
        coordinate_ref = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 5.0]])
        input_string = "1837.3624 1.0 2.0 3.0 4.0 \n" \
            + "34631.9731 2.0 1.0 4.0 5.0 \n"

        parameters._parse_mass_value(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)

        # 1 bead test
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_2-1.in")

        mass_ref = np.array([1837.3624, 34631.9731])
        coordinate_ref = np.array([[1.0], [2.0]])
        input_string = "1837.3624 1.0 \n" \
            + "34631.9731 2.0 \n"

        parameters._parse_mass_value(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)

        input_string = "1837.3624 1.0 2.0 \n" \
            + "34631.9731 2.0 1.0 4.0 \n"
        with self.assertRaises(ValueError):
            parameters._parse_mass_value(input_string)
        pass

    def test_parse_mass_value_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_2-4.in")
        parameters['rpmd'] = {'nm_transform': 'matrix'}

        mass_ref = np.array([1, 3])
        centroid_ref = np.array([1.0, 2.0])
        input_string = "1.0 1.0 \n" \
            + "3.0 2.0 \n"

        parameters._parse_mass_value(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        self.assertTrue(parameters.coordinates.shape == (2, 4))
        np.testing.assert_allclose(np.mean(parameters.coordinates, axis=1),
                                   centroid_ref, rtol=1e-7)
        self.assertTrue(parameters.momenta is None)

        return

    def test_parse_xyz_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-4.in")
        parameters['rpmd'] = {'nm_transform': 'matrix'}

        mass_ref = np.array([1837.152646, 1837.152646, 1837.152646,
                             34631.970366, 34631.970366, 34631.970366])
        centroid_ref = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])
        input_string = "H 1.0 2.0 3.0 \n" \
            + "F 2.0 1.0 4.0 \n"

        parameters._parse_xyz(input_string)
        self.assertTrue(parameters.coordinates.shape == (6, 4))
        np.testing.assert_allclose(np.mean(parameters.coordinates, axis=1),
                                   centroid_ref, rtol=1e-7)
        self.assertTrue(parameters.momenta is None)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-4)

        return

    def test_get_section(self):
        section1_reference = {"miep": "muh", "blah": "", "blubb": "",
                              "dof": "4"}
        section2_reference = {"blubb": "1.3 fs"}
        section3_reference = {"blibb": "1.3 fs", "hot": "", "temp": "300 K"}
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_works.in")
        self.assertDictEqual(section1_reference, parameters.get("system"))
        self.assertDictEqual(section2_reference, parameters.get("trajectory"))
        self.assertDictEqual(section3_reference, parameters.get("pes"))

        self.assertEqual(None, parameters.get("wrong"))

        self.assertTrue("system" in parameters)
        self.assertTrue("pes" in parameters)
        self.assertFalse("wrong" in parameters)

        return

    def test_flatten_shifts(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_shifts.in")
        parameters._flatten_shifts()
        parameters._c_type = 'xpacdt'

        positionShift_ref = np.array([1.0, 2.0, 3.0, 4.0])
        momentumShift_ref = np.array([-1.0, -2.0, -3.0, -4.0])

        np.testing.assert_array_equal(parameters.positionShift, positionShift_ref)
        np.testing.assert_array_equal(parameters.momentumShift, momentumShift_ref)

    @unittest.skip("Implicitly tested in parse modules.")
    def test_format_coordinates(self):
        # Implicity tested in parse modules - not clear how to test separately.
        pass

    @unittest.skip("Please implement a test here.")
    def test_parse_beads(self):
        raise NotImplementedError("Please implement a test here!!")

    @unittest.skip("Please implement a test here.")
    def test_parse_masses(self):
        raise NotImplementedError("Please implement a test here!!")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(InputfileTest)
    unittest.TextTestRunner().run(suite)
