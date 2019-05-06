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

    def test_creation(self):
        with self.assertRaises(FileNotFoundError):
            infile.Inputfile("input.in")

        infile.Inputfile("FilesForTesting/InputfileTest/input_empty.in")

        return

    def test_parse_file(self):
        with self.assertRaises(IOError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleKey.in")

        with self.assertRaises(IOError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleEqual.in")

        input_reference = {"system": {"miep": "muh", "blah": "", "blubb": ""},
                           "trajectory": {"blubb": "1.3 fs"},
                           "pes": {"blibb": "1.3 fs", "hot": "",
                                   "temp": "300 K"}}
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_works.in")
        self.assertDictEqual(input_reference, parameters.store)

        return

    def test_parse_values(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_empty.in")

        key_value_reference = {"miep": "kuh"}
        key_value = parameters._parse_values("miep = kuh")
        self.assertDictEqual(key_value_reference, key_value)

        key_value_reference = {"miep": "kuh ist langer string."}
        key_value = parameters._parse_values("miep = kuh ist langer string.")
        self.assertDictEqual(key_value_reference, key_value)

        key_value_reference = {"muh": ""}
        key_value = parameters._parse_values("muh ")
        self.assertDictEqual(key_value_reference, key_value)

        with self.assertRaises(IOError):
            key_value = parameters._parse_values("mehrere = ist = doof")

        return

    def test_parse_xyz(self):
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_empty.in")

        mass_ref = np.array([1837.362363054474, 34631.97313115233])
        coordinate_ref = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        input_string = "H 1.0 2.0 3.0 \n" \
            + "F 2.0 1.0 4.0 \n"

        parameters._parse_xyz(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)

        parameters._parse_xyz(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)


        # test unknwon element
        input_string = "J 1.0 2.0 3.0 \n" \
            + "F 2.0 1.0 4.0 \n"
        with self.assertRaises(AttributeError):
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
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_empty.in")

        mass_ref = np.array([1837.3624, 34631.9731])
        coordinate_ref = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        input_string = "1837.3624 1.0 2.0 3.0 \n" \
            + "34631.9731 2.0 1.0 4.0 \n"

        parameters._parse_mass_value(input_string)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(parameters.coordinates, coordinate_ref,
                                   rtol=1e-7)

        input_string = "1837.3624 1.0 2.0 \n" \
            + "34631.9731 2.0 1.0 4.0 \n"
        with self.assertRaises(ValueError):
            parameters._parse_mass_value(input_string)
        pass

    def test_get_section(self):
        section1_reference = {"miep": "muh", "blah": "", "blubb": ""}
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


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(InputfileTest)
    unittest.TextTestRunner().run(suite)
