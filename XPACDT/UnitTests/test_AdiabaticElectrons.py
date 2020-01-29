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


class AdiabaticElectronsTest(unittest.TestCase):

    def setUp(self):
        self.pes1D_harmonic = elec.AdiabaticElectrons(
            {'system': {'Interface': 'OneDPolynomial'},
             'OneDPolynomial': {'a': "0.0 0.0 0.5"}},
            [1])
        self.pes1D_shifted_harmonic = elec.AdiabaticElectrons(
            {'system': {'Interface': 'OneDPolynomial'},
             'OneDPolynomial': {'a': "0.0 0.0 0.5", 'x0': '1.0'}},
            [1])
        self.pes1D_anharmonic = elec.AdiabaticElectrons(
            {'system': {'Interface': 'OneDPolynomial'},
             'OneDPolynomial': {'a': "0.0 0.0 0.5 0.1 0.01"}},
            [1])
        self.pes1D_quartic = elec.AdiabaticElectrons(
            {'system': {'Interface': 'OneDPolynomial'},
             'OneDPolynomial': {'a': "0.0 0.0 0.0 0.0 0.25"}},
            [1])

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


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(AdiabaticElectronsTest)
    unittest.TextTestRunner().run(suite)
