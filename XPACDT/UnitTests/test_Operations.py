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

import XPACDT.Tools.Operations as operations
import XPACDT.System.Nuclei as nuclei


class OperationsTest(unittest.TestCase):

    def setUp(self):
        # TODO: set up log
        self.log = {'nuclei': nuclei.Nuclei(None, None, None)}

    def test_position(self):
        pos = operations.position("-x 0".split(), self.log)
        self.assertAlmostEqual(pos, 0.0)

    def test_momentum(self):
        raise NotImplementedError("Please implement a test here!!")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(OperationsTest)
    unittest.TextTestRunner().run(suite)
