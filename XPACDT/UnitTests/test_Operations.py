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
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURmomE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  **************************************************************************

import numpy as np
import unittest

import XPACDT.Tools.Operations as operations
import XPACDT.System.Nuclei as nuclei
import XPACDT.Input.Inputfile as infile


class OperationsTest(unittest.TestCase):

    def setUp(self):
        # Set up nuclei as log
        self.log_classical = nuclei.Nuclei(4, infile.Inputfile("FilesForTesting/OperationsTest/input_classicalNuclei.in"), 0.0)
        self.log_rpmd = nuclei.Nuclei(4, infile.Inputfile("FilesForTesting/OperationsTest/input_rpmdNuclei.in"), 0.0)

    def test_position(self):
        # first single position operations
        pos = operations.position("-1 0".split(), self.log_classical)
        pos_ref = np.array([2.0])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 1".split(), self.log_classical)
        pos_ref = np.array([1.0])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 2".split(), self.log_classical)
        pos_ref = np.array([4.0])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 3".split(), self.log_classical)
        pos_ref = np.array([-2.0])
        np.testing.assert_array_equal(pos, pos_ref)

        with self.assertRaises(IndexError):
            pos = operations.position("-1 4".split(), self.log_classical)

        # test combinations
        pos = operations.position("-1 0,1,2,3".split(), self.log_classical)
        pos_ref = np.array([2.0, 1.0, 4.0, -2.0])
        np.testing.assert_array_equal(pos, pos_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            pos = operations.position("-1 m,0,1,2,3".split(), self.log_classical)
            pos_ref = np.array([np.NaN])
            np.testing.assert_array_equal(pos, pos_ref)

        # test distances
        pos = operations.position("-1 0 -2 1".split(), self.log_classical)
        pos_ref = 1.0
        np.testing.assert_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3".split(), self.log_classical)
        pos_ref = np.sqrt(4.0+9.0)
        np.testing.assert_equal(pos, pos_ref)

        # test projections
        pos = operations.position("-1 0 -p <,1.0".split(), self.log_classical)
        pos_ref = 0.0
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p >,1.0".split(), self.log_classical)
        pos_ref = 1.0
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p 0.0,<,3.0".split(), self.log_classical)
        pos_ref = 1.0
        np.testing.assert_array_equal(pos, pos_ref)


        pos = operations.position("-1 0,1 -p 1.5,<,3.0".split(), self.log_classical)
        pos_ref = [1.0, 0.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p >,3.0".split(), self.log_classical)
        pos_ref = [0.0, 0.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p  <,3.0".split(), self.log_classical)
        pos_ref = [1.0, 1.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3 -p <,4.0".split(), self.log_classical)
        pos_ref = 1.0
        np.testing.assert_equal(pos, pos_ref)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <4.0".split(), self.log_classical)

        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,4.0,>,3.0".split(), self.log_classical)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,>,4.0".split(), self.log_classical)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1,2 -2 2,3".split(), self.log_classical)

        # RPMD stuff - centroid
        # first single position operations
        pos = operations.position("-1 0".split(), self.log_rpmd)
        pos_ref = np.array([1.0])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 1".split(), self.log_rpmd)
        pos_ref = np.array([-0.5])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 2".split(), self.log_rpmd)
        pos_ref = np.array([3.0])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 3".split(), self.log_rpmd)
        pos_ref = np.array([-2.25])
        np.testing.assert_array_equal(pos, pos_ref)

        with self.assertRaises(IndexError):
            pos = operations.position("-1 4".split(), self.log_rpmd)

        # test combinations
        pos = operations.position("-1 0,1,2,3".split(), self.log_rpmd)
        pos_ref = np.array([1.0, -0.5, 3.0, -2.25])
        np.testing.assert_array_equal(pos, pos_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            pos = operations.position("-1 m,0,1,2,3".split(), self.log_rpmd)
            pos_ref = np.array([np.NaN])
            np.testing.assert_array_equal(pos, pos_ref)

        # test distances
        pos = operations.position("-1 0 -2 1".split(), self.log_rpmd)
        pos_ref = 1.5
        np.testing.assert_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3".split(), self.log_rpmd)
        pos_ref = np.sqrt(4.0+3.0625)
        np.testing.assert_equal(pos, pos_ref)

        # test projections
        pos = operations.position("-1 0 -p <,1.0".split(), self.log_rpmd)
        pos_ref = 0.0
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p >,0.5".split(), self.log_rpmd)
        pos_ref = 1.0
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p 0.0,<,3.0".split(), self.log_rpmd)
        pos_ref = 1.0
        np.testing.assert_array_equal(pos, pos_ref)


        pos = operations.position("-1 0,1 -p 0.5,<,3.0".split(), self.log_rpmd)
        pos_ref = [1.0, 0.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p >,3.0".split(), self.log_rpmd)
        pos_ref = [0.0, 0.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p  <,3.0".split(), self.log_rpmd)
        pos_ref = [1.0, 1.0]
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3 -p <,4.0".split(), self.log_rpmd)
        pos_ref = 1.0
        np.testing.assert_equal(pos, pos_ref)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <4.0".split(), self.log_rpmd)

        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,4.0,>,3.0".split(), self.log_rpmd)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,>,4.0".split(), self.log_rpmd)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1,2 -2 2,3".split(), self.log_rpmd)

        # RPMD stuff - beads
        # first single position operations
        pos = operations.position("-1 0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[2.0, 3.0, -1.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 1 -r".split(), self.log_rpmd)
        pos_ref = np.array([[1.0, 2.0, -2.0, -3.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 2 -r".split(), self.log_rpmd)
        pos_ref = np.array([[4.0, 5.0, 2.0, 1.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 3 -r".split(), self.log_rpmd)
        pos_ref = np.array([[-2.0, -3.0, -4.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        with self.assertRaises(IndexError):
            pos = operations.position("-1 4 -r".split(), self.log_rpmd)

        # test combinations
        pos = operations.position("-1 0,1,2,3 -r".split(), self.log_rpmd)
        pos_ref = np.array([[2.0, 3.0, -1.0, 0.0], [1.0, 2.0, -2.0, -3.0], [4.0, 5.0, 2.0, 1.0], [-2.0, -3.0, -4.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            pos = operations.position("-1 m,0,1,2,3 -r".split(), self.log_rpmd)
            pos_ref = np.array([np.NaN])
            np.testing.assert_array_equal(pos, pos_ref)

        # test distances
        pos = operations.position("-1 0 -2 1 -r".split(), self.log_rpmd)
        pos_ref = np.array([1.0, 1.0, 1.0, 3.0])
        np.testing.assert_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3 -r".split(), self.log_rpmd)
        pos_ref = np.sqrt(np.array([4.0+9.0, 4.0+25.0, 9.0+4.0, 1.0+9.0]))
        np.testing.assert_equal(pos, pos_ref)

        # test projections
        pos = operations.position("-1 0 -p <,1.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p >,0.5 -r".split(), self.log_rpmd)
        pos_ref = np.array([[1.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0 -p 0.0,<,3.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)


        pos = operations.position("-1 0,1 -p 0.5,<,3.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p >,3.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -p  <,3.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(pos, pos_ref)

        pos = operations.position("-1 0,1 -2 2,3 -p <,4.0 -r".split(), self.log_rpmd)
        pos_ref = np.array([1.0, 0.0, 1.0, 1.0])
        np.testing.assert_equal(pos, pos_ref)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <4.0 -r".split(), self.log_rpmd)

        with self.assertRaises(RuntimeError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,4.0,>,3.0 -r".split(), self.log_rpmd)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1 -2 2,3 -p <,>,4.0 -r".split(), self.log_rpmd)

        with self.assertRaises(ValueError):
            pos = operations.position("-1 0,1,2 -2 2,3 -r".split(), self.log_rpmd)

    def test_momentum(self):
        # first single momentum operations
        mom = operations.momentum("-1 0".split(), self.log_classical)
        mom_ref = np.array([-1.0])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 1".split(), self.log_classical)
        mom_ref = np.array([0.1])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 2".split(), self.log_classical)
        mom_ref = np.array([2.0])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 3".split(), self.log_classical)
        mom_ref = np.array([1.25])
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(IndexError):
            mom = operations.momentum("-1 4".split(), self.log_classical)

        # test combinations
        mom = operations.momentum("-1 0,1,2,3".split(), self.log_classical)
        mom_ref = np.array([-1.0, 0.1, 2.0, 1.25])
        np.testing.assert_array_equal(mom, mom_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 m,0,1,2,3".split(), self.log_classical)
            mom_ref = np.array([np.NaN])
            np.testing.assert_array_equal(mom, mom_ref)

        # test distances
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0 -2 1".split(), self.log_classical)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3".split(), self.log_classical)

        # test projections
        mom = operations.momentum("-1 0 -p <,1.0".split(), self.log_classical)
        mom_ref = 1.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p >,1.0".split(), self.log_classical)
        mom_ref = 0.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p 0.0,<,3.0".split(), self.log_classical)
        mom_ref = 0.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p 0.0,<,3.0".split(), self.log_classical)
        mom_ref = [0.0, 1.0]
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p >,3.0".split(), self.log_classical)
        mom_ref = [0.0, 0.0]
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p  <,3.0".split(), self.log_classical)
        mom_ref = [1.0, 1.0]
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0".split(), self.log_classical)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <4.0".split(), self.log_classical)

        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0,>,3.0".split(), self.log_classical)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,>,4.0".split(), self.log_classical)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1,2 -2 2,3".split(), self.log_classical)

        # RPMD stuff - centroid
        # first single momentum operations
        mom = operations.momentum("-1 0".split(), self.log_rpmd)
        mom_ref = np.array([-0.2])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 1".split(), self.log_rpmd)
        mom_ref = np.array([0.6])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 2".split(), self.log_rpmd)
        mom_ref = np.array([3.5])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 3".split(), self.log_rpmd)
        mom_ref = np.array([0.3125])
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(IndexError):
            mom = operations.momentum("-1 4".split(), self.log_rpmd)

        # test combinations
        mom = operations.momentum("-1 0,1,2,3".split(), self.log_rpmd)
        mom_ref = np.array([-0.2, 0.6, 3.5, 0.3125])
        np.testing.assert_array_equal(mom, mom_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 m,0,1,2,3".split(), self.log_rpmd)
            mom_ref = np.array([np.NaN])
            np.testing.assert_array_equal(mom, mom_ref)

        # test distances
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0 -2 1".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3".split(), self.log_rpmd)

        # test projections
        mom = operations.momentum("-1 0 -p <,1.0".split(), self.log_rpmd)
        mom_ref = 1.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p >,0.5".split(), self.log_rpmd)
        mom_ref = 0.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p 0.0,<,3.0".split(), self.log_rpmd)
        mom_ref = 0.0
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p 0.5,<,3.0".split(), self.log_rpmd)
        mom_ref = [0.0, 1.0]
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p >,3.0".split(), self.log_rpmd)
        mom_ref = [0.0, 0.0]
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p  <,3.0".split(), self.log_rpmd)
        mom_ref = [1.0, 1.0]
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0".split(), self.log_rpmd)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <4.0".split(), self.log_rpmd)

        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0,>,3.0".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,>,4.0".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1,2 -2 2,3".split(), self.log_rpmd)

        # RPMD stuff - beads
        # first single momentum operations
        mom = operations.momentum("-1 0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[-1.0, 2.0, 0.2, -2.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 1 -r".split(), self.log_rpmd)
        mom_ref = np.array([[0.1, 1.5, 0.5, 0.3]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 2 -r".split(), self.log_rpmd)
        mom_ref = np.array([[2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 3 -r".split(), self.log_rpmd)
        mom_ref = np.array([[1.25, 0.0, -0.5, 0.5]])
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(IndexError):
            mom = operations.momentum("-1 4 -r".split(), self.log_rpmd)

        # test combinations
        mom = operations.momentum("-1 0,1,2,3 -r".split(), self.log_rpmd)
        mom_ref = np.array([[-1.0, 2.0, 0.2, -2.0], [0.1, 1.5, 0.5, 0.3], [2.0, 3.0, 4.0, 5.0], [1.25, 0.0, -0.5, 0.5]])
        np.testing.assert_array_equal(mom, mom_ref)

        # TODO: add test for COM once implemented
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 m,0,1,2,3 -r".split(), self.log_rpmd)
            mom_ref = np.array([np.NaN])
            np.testing.assert_array_equal(mom, mom_ref)

        # test distances
        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0 -2 1 -r".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -r".split(), self.log_rpmd)

        # test projections
        mom = operations.momentum("-1 0 -p <,1.0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[1.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p >,0.5 -r".split(), self.log_rpmd)
        mom_ref = np.array([[0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0 -p 0.0,<,3.0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[0.0, 1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(mom, mom_ref)


        mom = operations.momentum("-1 0,1 -p 0.5,<,3.0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p >,1.0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        mom = operations.momentum("-1 0,1 -p  <,3.0 -r".split(), self.log_rpmd)
        mom_ref = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(mom, mom_ref)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0 -r".split(), self.log_rpmd)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <4.0 -r".split(), self.log_rpmd)

        with self.assertRaises(RuntimeError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,4.0,>,3.0 -r".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1 -2 2,3 -p <,>,4.0 -r".split(), self.log_rpmd)

        with self.assertRaises(NotImplementedError):
            mom = operations.momentum("-1 0,1,2 -2 2,3 -r".split(), self.log_rpmd)

    def test_projection(self):
        # below a value
        values = operations._projection("<,0.0", -1.0)
        values_ref = 1.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("<,0.0", 0.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("<,0.0", 1.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        # above a value
        values = operations._projection(">,0.0", -1.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection(">,0.0", 0.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection(">,0.0", 1.0)
        values_ref = 1.0
        np.testing.assert_equal(values, values_ref)

        # within a range
        values = operations._projection("0.0,<,1.0", -1.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("0.0,<,1.0", 0.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("0.0,<,1.0", 0.3)
        values_ref = 1.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("0.0,<,1.0", 1.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        values = operations._projection("0.0,<,1.0", 3.0)
        values_ref = 0.0
        np.testing.assert_equal(values, values_ref)

        # below a value - 1d array
        values = operations._projection("<,0.0", np.array([-1.0, 0.0, 1.0]))
        values_ref = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(values, values_ref)

        # above a value - 1d array
        values = operations._projection(">,0.0", np.array([-1.0, 0.0, 1.0]))
        values_ref = np.array([0.0, 0.0, 1.0])
        np.testing.assert_equal(values, values_ref)

        # within a range - 1d array
        values = operations._projection("0.0,<,1.0", np.array([-1.0, 0.0, 0.3, 1.0, 2.0]))
        values_ref = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_equal(values, values_ref)

        # below a value - 2d array
        values = operations._projection("<,0.0", np.array([[-1.0], [0.0], [1.0]]))
        values_ref = np.array([[1.0], [0.0], [0.0]])
        np.testing.assert_array_equal(values, values_ref)

        # above a value - 2d array
        values = operations._projection(">,0.0", np.array([[-1.0], [0.0], [1.0]]))
        values_ref = np.array([[0.0], [0.0], [1.0]])
        np.testing.assert_equal(values, values_ref)

        # within a range - 2d array
        values = operations._projection("0.0,<,1.0", np.array([[-1.0, 0.0, 0.2], [0.3, 1.0, 2.0]]))
        values_ref = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        np.testing.assert_equal(values, values_ref)

        # Different parsing errors
        with self.assertRaises(RuntimeError):
            operations._projection("<4.0", None)

        with self.assertRaises(RuntimeError):
            operations._projection("<,4.0,>,3.0", None)

        with self.assertRaises(ValueError):
            operations._projection("<,>,4.0", None)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(OperationsTest)
    unittest.TextTestRunner().run(suite)
