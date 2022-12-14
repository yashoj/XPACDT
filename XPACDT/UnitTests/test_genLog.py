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

import numpy as np
import os
import unittest

import XPACDT.Input.Inputfile as infile
import XPACDT.Bin.genLog as genLog
import XPACDT.System.Nuclei as nuclei


class genLogTest(unittest.TestCase):

    def setUp(self):
        parameters_classical = infile.Inputfile("FilesForTesting/SystemTests/Classical.in")
        parameters_rpmd = infile.Inputfile("FilesForTesting/SystemTests/RPMD.in")

        self.nuclei_classical = nuclei.Nuclei(parameters_classical, None)
        self.nuclei_rpmd = nuclei.Nuclei(parameters_rpmd, None)

    def test_write_R(self):
        time_reference = np.random.rand()
        r_reference = np.random.rand(4, 1)
        rc_reference = np.average(r_reference, axis=1)

        self.nuclei_classical.positions = r_reference
        self.nuclei_classical.time = time_reference
        log = self.nuclei_classical

        outfile = open("R.log", 'w')
        genLog.write_R(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("R.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(rc_reference, written_data[1:], atol=1e-7)

        r_reference = np.random.rand(4, 3)
        rc_reference = np.average(r_reference, axis=1)

        self.nuclei_rpmd.positions = r_reference
        self.nuclei_rpmd.time = time_reference
        log = self.nuclei_rpmd

        outfile = open("R.log", 'w')
        genLog.write_R(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("R.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(rc_reference, written_data[1:], atol=1e-7)

        os.remove("R.log")

        return

    def test_write_P(self):
        time_reference = np.random.rand()
        p_reference = np.random.rand(4, 1)
        pc_reference = np.average(p_reference, axis=1)

        self.nuclei_classical.momenta = p_reference
        self.nuclei_classical.time = time_reference
        log = self.nuclei_classical

        outfile = open("P.log", 'w')
        genLog.write_P(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("P.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(pc_reference, written_data[1:], atol=1e-7)

        p_reference = np.random.rand(4, 3)
        pc_reference = np.average(p_reference, axis=1)

        self.nuclei_rpmd.momenta = p_reference
        self.nuclei_rpmd.time = time_reference
        log = self.nuclei_rpmd

        outfile = open("P.log", 'w')
        genLog.write_P(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("P.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(pc_reference, written_data[1:], atol=1e-7)

        os.remove("P.log")

        return

    def test_write_Rrp(self):
        time_reference = np.random.rand()
        r_reference = np.random.rand(4, 1)

        self.nuclei_classical.positions = r_reference
        self.nuclei_classical.time = time_reference
        log = self.nuclei_classical

        outfile = open("Rrp.log", 'w')
        genLog.write_Rrp(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("Rrp.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(r_reference.flatten(), written_data[1:],
                                   atol=1e-7)

        r_reference = np.random.rand(4, 3)

        self.nuclei_rpmd.positions = r_reference
        self.nuclei_rpmd.time = time_reference
        log = self.nuclei_rpmd

        outfile = open("Rrp.log", 'w')
        genLog.write_Rrp(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("Rrp.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(r_reference.flatten(), written_data[1:], atol=1e-7)

        os.remove("Rrp.log")

        return

    def test_write_Prp(self):
        time_reference = np.random.rand()
        p_reference = np.random.rand(4, 1)

        self.nuclei_classical.momenta = p_reference
        self.nuclei_classical.time = time_reference
        log = self.nuclei_classical

        outfile = open("Prp.log", 'w')
        genLog.write_Prp(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("Prp.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(p_reference.flatten(), written_data[1:], atol=1e-7)

        p_reference = np.random.rand(4, 3)

        self.nuclei_rpmd.momenta = p_reference
        self.nuclei_rpmd.time = time_reference
        log = self.nuclei_rpmd

        outfile = open("Prp.log", 'w')
        genLog.write_Prp(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("Prp.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(p_reference.flatten(), written_data[1:], atol=1e-7)

        os.remove("Prp.log")

        return

    def test_write_electronic_state(self):
        time_reference = np.random.rand()
        state_reference = 1

        # Test for surface hopping electrons
        param_sh_classical = infile.Inputfile("FilesForTesting/SystemTests/input_SH_classical.in")
        param_sh_classical["SurfaceHoppingElectrons"]["initial_state"] = state_reference

        nuclei_sh_classical = nuclei.Nuclei(param_sh_classical, time_reference)
        log = nuclei_sh_classical

        outfile = open("state.log", 'w')
        genLog.write_electronic_state(log, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("state.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(state_reference, written_data[1:], atol=1e-7)

        os.remove("state.log")

        return

    def test_write_energy(self):
        time_reference = np.random.rand()
        energy_reference = np.array([1.125, 1.125, 2.25])

        # Setting up proper 1D nuclei.
        parameters_rpmd_1D = infile.Inputfile("FilesForTesting/SystemTests/RPMD_1D.in")
        log_rpmd_1D = nuclei.Nuclei(parameters_rpmd_1D, time_reference)

        outfile = open("energy.log", 'w')
        genLog.write_energy(log_rpmd_1D, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("energy.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(energy_reference, written_data[1:], atol=1e-7)

        os.remove("energy.log")
        return

    def test_write_energy_rp(self):
        time_reference = np.random.rand()
        energy_reference = np.array([7.0, 1.5, 7.0, 15.5])

        # Setting up proper 1D nuclei.
        parameters_rpmd_1D = infile.Inputfile("FilesForTesting/SystemTests/RPMD_1D.in")
        log_rpmd_1D = nuclei.Nuclei(parameters_rpmd_1D, time_reference)

        outfile = open("energy_rp.log", 'w')
        genLog.write_energy_rp(log_rpmd_1D, outfile, 16, 8)
        outfile.close()

        written_data = np.genfromtxt("energy_rp.log")
        np.testing.assert_allclose(time_reference, written_data[0], atol=1e-7)
        np.testing.assert_allclose(energy_reference, written_data[1:], atol=1e-7)

        os.remove("energy_rp.log")
        return

    @unittest.skip("Please implement a test here.")
    def test_setup_outfiles(self):
        raise NotImplementedError("Please implement a test here.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(genLogTest)
    unittest.TextTestRunner().run(suite)
