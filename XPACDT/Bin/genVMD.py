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

""" This module is for generating VMD readable output. A file containing the
positions in XYZ format is produced as well as a VMD script for basic
displaying.
"""

import argparse
import numpy as np
import os
import pickle
import sys
import subprocess as sp

import XPACDT.Tools.Units as units


def start():
    """
    Generate a VMD readable script. Currently very basic...
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    i_help = "Name of the XPACDT pickle file to parse for output generation." \
        " The default is 'pickle.dat'."
    parser.add_argument("-i", "--input", type=str, dest="PickleFile",
                        required=False, help=i_help, default='pickle.dat')

    args = parser.parse_args()

    # Get input file and folder
    system = pickle.load(open(args.PickleFile, 'rb'))
    folder = os.path.dirname(os.path.abspath(args.PickleFile))

    # Generate XYZ file
    gen_XYZ(system, folder)

    # Generate VMD Script
    gen_VMD(folder)
    run_VMD(folder)


def gen_XYZ(system, folder):
    """ Output XYZ files for the centroids and the beads.

    Parameters
    ----------
    system : XPACDT.System.System
        The system to produce the VMD movie for.
    folder : string
        The folder in which to work.
    """

    # TODO: Handle non-atom systems...

    # TODO: also add RPMD stuff, Use different format, etc.
    file = open(os.path.join(folder, "centroids.xyz"), 'w')
    symbols = np.array([units.atom_symbol(m) for m in system.log[0].masses[::3]])
    for log_nuclei in system.log:
        centroids = log_nuclei.x_centroid.reshape(-1, 3)
        atoms = np.concatenate((symbols[:, None], centroids), axis=1)

        file.write(str(len(symbols)) + "\n\n")
        for atom in atoms:
            for value in atom:
                file.write(str(value).replace('H', 'Na') + "\t")
            file.write("\n")

    file.close()

    file = open(os.path.join(folder, "beads.xyz"), 'w')
    # TODO make variable!
    n_beads = system.nuclei.n_beads[0]
    for log_nuclei in system.log:
        positions = log_nuclei.positions.reshape(3,3,4).swapaxes(1,2).reshape(-1, 3)
        atoms = np.concatenate((symbols.repeat(n_beads)[:, None], positions), axis=1)

        file.write(str(len(symbols)*n_beads) + "\n\n")
        for atom in atoms:
            for value in atom:
                file.write(str(value).replace('H', 'Na') + "\t")
            file.write("\n")

    file.close()


def gen_VMD(folder):
    """ Generates VMD script.

    Parameters
    ----------
    folder : string
        The folder in which to work.
    """

    file = open(os.path.join(folder, "movie.vmd"), 'w')
    file.write("## Read in centroids:\n")
    file.write("mol new {/home/ralph/results/XPACDT-Testing/tmp/test_pes/muh2/centroids.xyz} type {xyz} first 0 last -1 step 1 waitfor -1\n\n")
    file.write("## Set styles - Big balls + Dynamic bonds\n")
    file.write("mol modstyle 0 0 CPK 1.000000 0.000000 12.000000 12.000000\n")
    file.write("mol addrep 0\n")
    file.write("mol modstyle 1 0 DynamicBonds 2.000000 0.100000 12.000000\n\n")
    file.write("## Read in beads:\n")
    file.write("mol new {/home/ralph/results/XPACDT-Testing/tmp/test_pes/muh2/beads.xyz} type {xyz} first 0 last -1 step 1 waitfor -1\n\n")
    file.write("## Set styles - small balls, transparent\n")
    file.write("mol modstyle 0 1 CPK 0.500000 0.000000 12.000000 12.000000\n")
    file.write("mol modmaterial 0 1 Transparent\n")
    file.write("mol modcolor 0 1 ColorID 0\n\n")
    file.write("## Go to the beginning\n")
    file.write("animate goto 0\n")
    file.write("display resetview\n\n")
    file.write("## Get number of frames\n")
    file.write("set n [molinfo top get numframes]\n\n")
    file.write("## Reset views etc.\n")
    file.write("color Display Background white\n")
    file.write("axes location Off\n")
    file.write("display projection orthographic\n")
    file.write("display depthcue off\n\n")
    file.write("## Iterate over frames\n")
    file.write("for { set i 0 } { $i < $n } { incr i } {\n\n")
    file.write("  ## Go to current frame\n")
    file.write("  animate goto $i\n\n")
    file.write("  ## Output\n")
    file.write("  set name [format \"%05d\" $i]\n")
    file.write("  render TachyonInternal $name.tga convert %s %s.png\n")
    file.write("}\n\n")
    file.write("quit\n")
    file.close()

    file = open(os.path.join(folder, "run.sh"), 'w')
    file.write("#!/bin/bash\n\n")
    file.write("rm movie.gif movie.mp4\n\n")
    file.write("vmd -e movie.vmd \n\n")
    file.write("convert -delay 10 -loop 1 *.png movie.gif\n")
    file.write("ffmpeg -framerate 15 -i %05d.tga.png -s:v 600x750 -c:v libx264 -profile:v high -crf 15 -r 30 -pix_fmt yuv420p movie.mp4 \n\n")
    file.write("rm *tga *png\n")
    file.close()


def run_VMD(folder):
    """ Runs VMD script.

    Parameters
    ----------
    folder : string
        The folder in which to work.
    """
    command = "cd " + folder + "; "
    command += "chmod u+x run.sh; ./run.sh &> run.out"
    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
