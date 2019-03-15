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

# import XPACDT.Dynamics.System as xSystem
# import XPACDT.Input.Inputfile as infile


def propagate(system, parameters):
    """ Propagate the system."""

    # only a basic test right now
    outfile = open("/tmp/blah.dat", 'w')
    for i in range(101):
        outfile.write(str(i*0.1) + " ")
        outfile.write(str(system.nuclei.x_centroid[0]) + " ")
        outfile.write(str(system.nuclei.p_centroid[0]) + " ")
        outfile.write("\n")
        system.step(0.1)
    outfile.close()
