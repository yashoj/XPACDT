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

import pickle

# import XPACDT.Dynamics.System as xSystem
# import XPACDT.Input.Inputfile as infile


def propagate(system, parameters):
    """ Propagate the system."""

    # only a basic test right now
    outfile = open("/tmp/blah.dat", 'w')
    for i in range(11):
        outfile.write(str(i*0.1) + " ")
        outfile.write(str(system.nuclei.x_centroid[0]) + " ")
        outfile.write(str(system.nuclei.p_centroid[0]) + " ")
        outfile.write("\n")
        system.step(0.1)

    pickle.dump(system, open("/tmp/pickle.dat", 'wb'), -1)
    system2 = pickle.load(open("/tmp/pickle.dat", 'rb'))

    for t, nt in system2._log:
        print(t, nt.x_centroid[0])

    for i in range(11):
        outfile.write(str((i+11)*0.1) + " ")
        outfile.write(str(system2.nuclei.x_centroid[0]) + " ")
        outfile.write(str(system2.nuclei.p_centroid[0]) + " ")
        outfile.write("\n")
        system2.step(0.1)
    outfile.close()

    for t, nt in system2._log:
        print(t, nt.x_centroid[0])
