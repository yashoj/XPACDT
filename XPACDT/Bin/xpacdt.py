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

import XPACDT.Input.Inputfile as infile


def start():
    """Start any XPACDT calculation."""

    parser = argparse.ArgumentParser()

    iHelp = "Name of the XPACDT input file."
    parser.add_argument("-i", "--input", type=str, dest="InputFile",
                        required=True, help=iHelp)

    args = parser.parse_args()
    print("The inputfile '" + args.InputFile + "' is read! \n")

    inputFile = infile.Inputfile(args.InputFile)
    print(inputFile.get_section("quack"))

    return


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
