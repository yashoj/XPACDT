#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yashoj
"""

import numpy as np
import unittest

import XPACDT.Dynamics.MassiveAndersen as ma


class MassiveAndersenTest(unittest.TestCase):
    
    def setUp(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        self.parameters0 = infile.Inputfile("FilesForTesting/SamplingTest/input_Wigner_0.in")
        self.system0 = xSystem.System(self.parameters0)


    def test_forward_backward(self):
        


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            MassiveAndersenTest)
    unittest.TextTestRunner().run(suite)
