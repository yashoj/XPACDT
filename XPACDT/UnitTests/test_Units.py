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

import XPACDT.Tools.Units as units


class UnitsTest(unittest.TestCase):

#    def setUp(self):
#        # todo create input file here.
#        self.input = infile.Inputfile("input.in")

    def test_atom_mass(self):
        # from 
        amu_to_au = 1.0 / 5.48579909065e-4
        # from http://www.ciaaw.org/atomic-weights.htm
        mass_h_ref = 1.0078250322 * amu_to_au
        mass_f_ref = 18.998403163 * amu_to_au
        mass_o_ref = 15.994914619 * amu_to_au
        mass_d_ref = 2.0141017781 * amu_to_au
        mass_o18_ref = 17.999159613 * amu_to_au

# TODO: change places etc. test; also check NM test
        np.testing.assert_allclose(units.atom_mass('H'), mass_h_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('F'), mass_f_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('O'), mass_o_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('H2'), mass_d_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('O18'), mass_o18_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('1'), mass_h_ref, rtol=1e-3)
        np.testing.assert_allclose(units.atom_mass('8'), mass_o_ref, rtol=1e-3)

    def test_parse_time(self):
        time_string = "1 au"
        time_ref = 1.0

        time = units.parse_time(time_string)
        np.testing.assert_allclose(time, time_ref)

        time_string = "1 fs"
        time_ref = 100.0 / 2.4188843265857

        time = units.parse_time(time_string)
        np.testing.assert_allclose(time, time_ref)

        time_string = "1 as"
        time_ref = 0.1 / 2.4188843265857

        time = units.parse_time(time_string)
        np.testing.assert_allclose(time, time_ref)

        time_string = "1 ps"
        time_ref = 100000.0 / 2.4188843265857

        time = units.parse_time(time_string)
        np.testing.assert_allclose(time, time_ref)

        with self.assertRaises(RuntimeError):
            units.parse_time("1 s")

    def test_constants(self):
        # from https://en.wikipedia.org/wiki/Boltzmann_constant
        boltzmann_ref = 3.1668114e-6
        np.testing.assert_allclose(units.boltzmann, boltzmann_ref, rtol=1e-6)

        # from http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
        nm_to_cm_ref = 219474.63
        np.testing.assert_allclose(units.nm_to_cm, nm_to_cm_ref)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitsTest)
    unittest.TextTestRunner().run(suite)
