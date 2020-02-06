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
import random
import unittest

import XPACDT.System.NRPMDElectrons as nrpmd
import XPACDT.Input.Inputfile as infile


class NRPMDElectronsTest(unittest.TestCase):

    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        parameters_c = infile.Inputfile("FilesForTesting/SystemTests/input_NRPMD_classical.in")
        parameters_mb = infile.Inputfile("FilesForTesting/SystemTests/input_NRPMD_multibeads.in")
#        print(parameters.n_beads)
#        self.system = xSystem.System(self.parameters0)
        self.electron = nrpmd.NRPMDElectrons(parameters_c, parameters_c.n_beads)  
        self.electron_mb = nrpmd.NRPMDElectrons(parameters_mb, parameters_mb.n_beads)
#       self.pes = tullym.TullyModel(1, **{'model_type': 'model_C'})
#       self.pes.diabatic_energy(np.array([[[0.0006], [0.1]],[[0.1], [-0.0006]]]))

#       Position and momenta for the one bead case of seed 0
#       [[-0.30191837][-1.69078578]]
#       [[-0.95333378][-0.37582371]]

#       Position and momenta for the two beads case of seed 0
#       [[-0.60174275 -0.27828619][ 0.7993503  -1.37453086]]
#       [[-0.79868996 -0.9604982 ][-1.53656731 -1.05388088]]

    def test_step(self):

        R = np.array([[0.0]])
        step_ref = (np.array([[-0.33850079], [-1.77728803]]),
                    np.array([[-0.77959315], [-0.34481733]]))
        step = self.electron.step(R)
        np.testing.assert_allclose(step, step_ref, rtol=1e-7)

        R = np.array([[1.0e5, -1.0e5]])
        step_ref_mb = (np.array([[-0.89549266, -0.27886244],
                                 [0.62565698, -1.37389828]]),
                       np.array([[-0.94121693, -0.96033106],
                                 [-1.38591375, -1.05470541]]))
        step_mb = self.electron_mb.step(R)
        np.testing.assert_allclose(step_mb, step_ref_mb, rtol=1e-7)

        return

    def test_energy(self):

        R = np.array([[0.0]])

        energy_ref_classical = np.array([0.0862764724734855])
        energy = self.electron.energy(R, centroid=False)

        print(energy_ref_classical)
        print("Energy", self.electron.energy(R, centroid=False))
        np.testing.assert_allclose(energy, energy_ref_classical, rtol=1e-7)

        R = np.array([[1.0e5, -1.0e5]])

        energy_ref_multibeads = np.array([0.1486475271, -0.0006])
        energy = self.electron_mb.energy(R, centroid=False)

        print(energy_ref_multibeads)
        print("Energy", self.electron_mb.energy(R, centroid=False))
        np.testing.assert_allclose(energy, energy_ref_multibeads, rtol=1e-7)

        return

    def test_gradient(self):

        R = np.array([[0.0]])
        gradient_ref = np.array([-0.078188825])
        gradient = self.electron.gradient(R, centroid=False)

        print(gradient_ref)
        print("Gradient", gradient)
        np.testing.assert_allclose(gradient, gradient_ref, rtol=1e-7)

        R = np.array([[1.0e5, -1.0e5]])
        gradient_ref_multibeads = np.array([0.000000, 0.00000000])
        gradient = self.electron_mb.gradient(R, centroid=False)

        print(gradient_ref_multibeads)
        print("Gradient", gradient)
        np.testing.assert_allclose(gradient, gradient_ref_multibeads, rtol=1e-7)
        return

    def test_get_population(self):

        R = np.array([[0.0]])
        pop_ref = np.array([0.0, 1.0])
        pop = self.electron.get_population(R, centroid=False)
        np.testing.assert_allclose(pop, pop_ref, rtol=1e-5)

        R = np.array([[1.0e5, -1.0e5]])
        pop_ref_mb = np.array([0.0, 1.0])
        pop_mb = self.electron_mb.get_population(R, centroid=False)
        np.testing.assert_allclose(pop_mb, pop_ref_mb, atol=1e-7)
        return


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(NRPMDElectronsTest)
    unittest.TextTestRunner().run(suite)
