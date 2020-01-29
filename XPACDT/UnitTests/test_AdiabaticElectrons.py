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

import XPACDT.System.AdiabaticElectrons as elec
import XPACDT.Input.Inputfile as infile


class AdiabaticElectronsTest(unittest.TestCase):

    def setUp(self):
        parameters_harmonic = infile.Inputfile("FilesForTesting/SystemTests/harmonic.in")
        self.pes1D_harmonic = elec.AdiabaticElectrons(parameters_harmonic)

        parameters_shifted_harmonic = infile.Inputfile("FilesForTesting/SystemTests/harmonic_shifted.in")
        self.pes1D_shifted_harmonic = elec.AdiabaticElectrons(parameters_shifted_harmonic)

        parameters_anharmonic = infile.Inputfile("FilesForTesting/SystemTests/anharmonic.in")
        self.pes1D_anharmonic = elec.AdiabaticElectrons(parameters_anharmonic)

        parameters_quartic = infile.Inputfile("FilesForTesting/SystemTests/quartic.in")
        self.pes1D_quartic = elec.AdiabaticElectrons(parameters_quartic)

        return

    def test_energy(self):
        energy_ref = 0.0
        energy = self.pes1D_harmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0
        energy = self.pes1D_harmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0
        energy = self.pes1D_shifted_harmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_shifted_harmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_anharmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0+0.1*1.0*1.0*1.0+0.01*1.0*1.0*1.0*1.0
        energy = self.pes1D_anharmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_quartic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.25*1.0*1.0*1.0*1.0
        energy = self.pes1D_quartic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

    def test_gradient(self):
        energy_ref = 0.0
        energy = self.pes1D_harmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0
        energy = self.pes1D_harmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0
        energy = self.pes1D_shifted_harmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_shifted_harmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_anharmonic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.5*1.0*1.0+0.1*1.0*1.0*1.0+0.01*1.0*1.0*1.0*1.0
        energy = self.pes1D_anharmonic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.0
        energy = self.pes1D_quartic.energy(np.array([[0.0]]))
        self.assertAlmostEqual(energy, energy_ref)

        energy_ref = 0.25*1.0*1.0*1.0*1.0
        energy = self.pes1D_quartic.energy(np.array([[1.0]]))
        self.assertAlmostEqual(energy, energy_ref)

    def test_current_state(self):
        pass
    
    def test_get_populations(self):
        pass

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(AdiabaticElectronsTest)
    unittest.TextTestRunner().run(suite)
