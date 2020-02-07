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

from XPACDT.Input.Error import XPACDTInputError


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
        with self.assertRaises(XPACDTInputError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleKey.in")

        with self.assertRaises(XPACDTInputError):
            infile.Inputfile("FilesForTesting/InputfileTest/input_doubleEqual.in")

        input_reference = {"system": {"miep": "muh", "blah": "", "blubb": "",
                                      "dof": "4"},
                           "trajectory": {"blubb": "1.3 fs"},
                           "pes": {"blibb": "1.3 fs", "hot": "",
                                   "temp": "300 K"},
                           "miep": {"": ""}}
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_works.in")
        # We check section by section as Inputfile add some computed
        # informations no present in the input file to its dict.
        for section in input_reference:
            self.assertDictEqual(input_reference[section], parameters[section])

        # TODO: add a test for 'commands'

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

        with self.assertRaises(XPACDTInputError):
            key_value = parameters._parse_values("mehrere = ist = doof")

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

    def test_mass_value_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_2-4.in")
        # Bypass the "only set key once" rule by setting .store directly
        parameters.store['rpmd'] = {'nm_transform': 'matrix'}

        mass_ref = np.array([1, 3])
        centroid_ref = np.array([1.0, 2.0])
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-7)
        self.assertTrue(parameters.coordinates.shape == (2, 4))
        np.testing.assert_allclose(np.mean(parameters.coordinates, axis=1),
                                   centroid_ref, rtol=1e-7)
        self.assertTrue("momenta" not in parameters)

    def test_xyz_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-4.in")
        # Bypass the "only set key once" rule by setting .store directly
        parameters.store['rpmd'] = {'nm_transform': 'matrix'}

        mass_ref = np.array([1837.152646, 1837.152646, 1837.152646,
                             34631.970366, 34631.970366, 34631.970366])
        centroid_ref = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])
        self.assertTrue(parameters.coordinates.shape == (6, 4))
        np.testing.assert_allclose(np.mean(parameters.coordinates, axis=1),
                                   centroid_ref, rtol=1e-7)
        self.assertTrue("momenta" not in parameters)
        np.testing.assert_allclose(parameters.masses, mass_ref, rtol=1e-4)

    @unittest.skip("Implicitly tested in parse modules.")
    def test_parse_coordinates_string(self):
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
