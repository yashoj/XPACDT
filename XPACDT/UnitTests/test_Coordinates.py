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
from XPACDT.Tools.Coordinates import parse_mass_value, parse_xyz


class CoordinatesTest(unittest.TestCase):
    def test_parse_xyz(self):
        mass_ref = np.array([1837.152646, 1837.152646, 1837.152646,
                             34631.970366, 34631.970366, 34631.970366])

        coordinate_ref = np.array([[1.0], [2.0], [3.0], [2.0], [1.0], [4.0]])
        input_string = ("H 1.0 2.0 3.0 \n"
                        "F 2.0 1.0 4.0 \n")

        atoms, masses, coordinates = parse_xyz(input_string)
        np.testing.assert_allclose(coordinates, coordinate_ref, rtol=1e-7)
        np.testing.assert_allclose(masses, mass_ref, rtol=1e-4)
        self.assertEqual(atoms[0], "H")
        self.assertEqual(atoms[1], "F")

        # test unknwon element
        input_string = ("J 1.0 2.0 3.0 \n"
                        "F 2.0 1.0 4.0 \n")

        with self.assertRaises(KeyError):
            parse_xyz(input_string)

        # test too many/few coordinates given
        input_string = ("H 1.0 2.0 \n"
                        "F 2.0 1.0 4.0 \n")
        with self.assertRaises(ValueError):
            parse_xyz(input_string)

        input_string = ("H 1.0 2.0 3.0\n"
                        "F 2.0 1.0 4.0 6.0\n")
        with self.assertRaises(ValueError):
            parse_xyz(input_string)

    def test_parse_mass_value(self):
        # 4 beads test
        mass_ref = np.array([1837.3624, 34631.9731])
        coordinate_ref = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 5.0]])
        input_string = ("1837.3624 1.0 2.0 3.0 4.0 \n"
                        "34631.9731 2.0 1.0 4.0 5.0 \n")

        masses, coordinates = parse_mass_value(input_string)
        np.testing.assert_allclose(masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(coordinates, coordinate_ref, rtol=1e-7)

        # 1 bead test
        mass_ref = np.array([1837.3624, 34631.9731])
        coordinate_ref = np.array([[1.0], [2.0]])
        input_string = ("1837.3624 1.0 \n"
                        "34631.9731 2.0 \n")

        masses, coordinates = parse_mass_value(input_string)
        np.testing.assert_allclose(masses, mass_ref, rtol=1e-7)
        np.testing.assert_allclose(coordinates, coordinate_ref, rtol=1e-7)

        # Unbalanced number of beads
        # TODO Shouldn't that be allowed for variable number of beads ?
        input_string = ("1837.3624 1.0 2.0 \n"
                        "34631.9731 2.0 1.0 4.0 \n")
        with self.assertRaises(ValueError):
            parse_mass_value(input_string)

    def test_mass_value_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_2-4.in")
        # Bypass the "only set key once" rule by setting .store directly
        parameters.store['rpmd'] = {'nm_transform': 'matrix'}

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

    def test_xyz_free_rp_sampling(self):
        # So far only shape of output and centroid value tested; maybe add
        # test for distribution?
        parameters = infile.Inputfile("FilesForTesting/InputfileTest/input_6-4.in")
        # Bypass the "only set key once" rule by setting .store directly
        parameters.store['rpmd'] = {'nm_transform': 'matrix'}

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