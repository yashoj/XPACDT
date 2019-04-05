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

""" This module is for generating human readable text files for some of the
quantities calculated in XPACDT. THIS IS NOT REALLY IMPLEMENTED YET!!

TODO: Read from command line what to output!
"""

import pickle


def start():
    """
    Generate a human readable logfile. This is not really functional yet and
    just for some small testing purposes!!
    """

    system = pickle.load(open('pickle.dat', 'rb'))
    R_outfile = open('R.log', 'w')
    P_outfile = open('P.log', 'w')
    for t, nuc in system._log:
        R_outfile.write(str(t) + " " + str(nuc.x_centroid[0]) + " \n")
        P_outfile.write(str(t) + " " + str(nuc.p_centroid[0]) + " \n")
    R_outfile.close()
    P_outfile.close()


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
