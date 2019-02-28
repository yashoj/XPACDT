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
#  CDTK is free software: you can redistribute it and/or modify
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

import argparse
import numpy as np

import XPACDT.Dynamics.VelocityVerlet as vv
import XPACDT.Input.Inputfile as infile

import XPACDT.Interfaces.OneDPolynomial as oneDP
import XPACDT.Interfaces.InterfaceTemplate as template


def start():
    """Start any XPACDT calculation."""

    parser = argparse.ArgumentParser()

    iHelp = "Name of the XPACDT input file."
    parser.add_argument("-i", "--input", type=str, dest="InputFile",
                        required=True, help=iHelp)

    args = parser.parse_args()
    print("The inputfile '" + args.InputFile + "' is read! \n")

    parameters = infile.Inputfile(args.InputFile)
    print(parameters.get_section("1dPolynomial"))

    # Example usage for potentials
    pes = oneDP.OneDPolynomial(**parameters.get_section("1dPolynomial"))
    print(isinstance(pes, oneDP.OneDPolynomial))
    print(isinstance(pes, template.Interface))
#    print(pes.name)
#    print(pes.energy(np.array([[0.0]])))
#    print(pes.energy(np.array([[0.0]])))
#    print(pes.energy(np.array([[1.0]])))
#    print(pes.minimize(np.array([0.1])))
#
#    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, False)
#    pes.plot_1D(np.array([0.0]), 0, -1.0, 1.0, 0.5, True)

    # Example usage for propagator - not meant to be used like this later!!
    propagator = vv.VelocityVerlet(0.0001, pes, np.array([1.0]), **{'beta': 8.0})
    r = np.random.rand(4).reshape(1, 4) + 1.0
    p = np.array([[0.0]*4])
    print(r, p)
    outfile = open("/tmp/blah.dat", 'w')
    for i in range(101):
        outfile.write(str(i*0.1) + " ")
        outfile.write(str(np.mean(r[0])) + " ")
        outfile.write(str(p[0, 0]) + " ")
        outfile.write("\n")
        r, p = propagator.propagate(r, p, 0.1)
    outfile.close()
    pass
    return


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
