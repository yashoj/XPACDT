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
#  Copyright (C) 2019, 2020
#  Ralph Welsch, DESY, <ralph.welsch@desy.de>
#  Yashoj Shakya, DESY, <yashoj.shakya@desy.de>
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
import shutil
import os

import XPACDT.Bin.compile_xpacdt as cx


class compile_xpacdtTest(unittest.TestCase):

    def setUp(self):
        os.mkdir("test")
        os.mkdir("test/Interfaces")
        os.mkdir("test/Interfaces/blah_mod")
        os.mkdir("test/System")
        os.mkdir("test/Sampling")
        os.mkdir("test/Dynamics")
        os.mknod("test/Interfaces/first.py")
        os.mknod("test/Interfaces/second.py")
        os.mknod("test/Interfaces/third.py")
        os.mknod("test/Interfaces/blah_mod/first.dat")
        os.mknod("test/Interfaces/blah_mod/second.dat")
        os.mknod("test/Interfaces/blah_mod/third.dat")

    def test_get_named_files(self):
        reference = ['Interfaces/third.py', 'Interfaces/first.py',
                     'Interfaces/second.py']
        ret = cx.get_named_files("test/Interfaces", "test")
        self.assertEqual(len(reference), len(ret))
        for s in reference:
            self.assertTrue(s in ret)

        reference = ['Interfaces/third.py', 'Interfaces/second.py']
        ret = cx.get_named_files("test/Interfaces", "test",
                                 exclusion=['first.py'])
        self.assertSetEqual(set(reference), set(ret))
        reference = ['Interfaces/third.py', 'Interfaces/first.py']
        ret = cx.get_named_files("test/Interfaces", "test", contains='ir')
        self.assertEqual(len(reference), len(ret))
        for s in reference:
            self.assertTrue(s in ret)

        reference = ['Interfaces/blah_mod/third.dat',
                     'Interfaces/blah_mod/first.dat',
                     'Interfaces/blah_mod/second.dat']
        ret = cx.get_named_files("test/Interfaces/blah_mod", "test",
                                 suffix='.dat')
        self.assertEqual(len(reference), len(ret))
        for s in reference:
            self.assertTrue(s in ret)

        shutil.rmtree("test")

    def test_discover_hidden_imports(self):
        reference = "--hidden-import='Interfaces.third' " \
            "--hidden-import='Interfaces.first' " \
            "--hidden-import='Interfaces.second' "
        ret = cx.discover_hidden_imports("test/Interfaces", "test").split()
        self.assertEqual(len(reference.split()), len(ret))
        for s in reference.split():
            self.assertTrue(s in ret)

    def test_discover_data_files(self):
        reference = "--add-data 'test/Interfaces/blah_mod/third.dat:Interfaces/blah_mod' " \
            "--add-data 'test/Interfaces/blah_mod/first.dat:Interfaces/blah_mod' " \
            "--add-data 'test/Interfaces/blah_mod/second.dat:Interfaces/blah_mod' "
        ret = cx.discover_data_files("test/Interfaces", "test").split()
        self.assertEqual(len(reference.split()), len(ret))
        for s in reference.split():
            self.assertTrue(s in ret)

    def tearDown(self):
        d = "test"
        if os.path.isdir(d):
            shutil.rmtree(d)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(compile_xpacdtTest)
    unittest.TextTestRunner().run(suite)
