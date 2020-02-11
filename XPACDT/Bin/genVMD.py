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
import math
import numpy as np
import os
import pickle
import subprocess as sp

import XPACDT.Tools.Units as units


def start():
    """
    Generate XYZ files from a XPACDT pickle file and a VMD readable script to
    display the XPACDT system. Currently very basic...
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
    """ Output XYZ files for the centroids and the beads. The data is stored
    in "centroids.xyz" and "beads.xyz", respectively.

    Parameters
    ----------
    system : XPACDT.System.System
        The system to produce the VMD movie for.
    folder : string
        The folder in which to work.
    """

    # Currently we assume that any system that has a multiple of three
    # number of degrees of freedom is and 'xyz' type
    # Any other system will be reformatted to fit the xyz type and additional
    # degrees of freedom are set to always be 0, the atom symbol for everything
    # is then assumed to be Ar for visual purposes

    n_dof = system.nuclei.n_dof
    # required number of atoms to save the actual degrees of freedom
    n_atom = math.ceil(n_dof / 3)
    n_dof_required = n_atom * 3

    # TODO make variable!
    n_beads = max(system.nuclei.n_beads)

    if n_dof < n_dof_required:
        symbols = np.array(['Ar'] * n_atom)
    else:
        # Save in input file and actually get from there
        symbols = np.array([units.atom_symbol(m) for m in system.log[0].masses[::3]])

    file_centroid = open(os.path.join(folder, "centroids.xyz"), 'w')
    file_beads = open(os.path.join(folder, "beads.xyz"), 'w')

    extended_centroid = np.zeros(n_dof_required)
    extended_beads = np.zeros((n_dof_required, n_beads))
    for log_nuclei in system.log:
        # Obtain centroid positions and reshape to match n_dof_required and
        # xyz file format
        extended_centroid[:n_dof] = log_nuclei.x_centroid
        centroids = extended_centroid.reshape(-1, 3)
        atoms_c = np.concatenate((symbols[:, None], centroids), axis=1)

        # Obtain bead positions and reshape to match n_dof_required and
        # xyz file format
        extended_beads[:n_dof, :] = log_nuclei.positions
        beads = extended_beads.reshape(-1, 3, n_beads).swapaxes(1, 2).reshape(-1, 3)
        atoms_b = np.concatenate((symbols.repeat(n_beads)[:, None], beads), axis=1)

        width = 16
        prec = 8

        file_centroid.write(str(len(symbols)) + "\n\n")
        for atom in atoms_c:
            for i, value in enumerate(atom):
                if i == 0:
                    file_centroid.write(value + " \t")
                else:
                    file_centroid.write("{: {width}.{prec}f} \t".format(float(value),
                                                                        width=width, prec=prec))
            file_centroid.write("\n")

        file_beads.write(str(len(symbols)*n_beads) + "\n\n")
        for atom in atoms_b:
            for i, value in enumerate(atom):
                if i == 0:
                    file_beads.write(value + " \t")
                else:
                    file_beads.write("{: {width}.{prec}f} \t".format(float(value),
                                                                     width=width, prec=prec))
            file_beads.write("\n")

    file_centroid.close()
    file_beads.close()


def gen_VMD(folder):
    """ Generates VMD script 'movie.vmd' and an associated bash script
    'run.sh'. The VMD script loads the xyz files generated and renders an
    image for each of them. The bash script actually runs the VMD script and
    then converts the generated images to a gif and a mp4.

    Parameters
    ----------
    folder : string
        The folder in which to work.
    """

    # movie.vmd is a vmd script to generate images of the system
    file = open(os.path.join(folder, "movie.vmd"), 'w')
    file.write("## Read in centroids:\n")
    file.write("mol new {" + os.path.join(folder, 'centroids.xyz') +
               "} type {xyz} first 0 last -1 step 1 waitfor -1\n\n")
    file.write("## Set styles for centroid - Big balls + Dynamic bonds\n")
    file.write("mol modstyle 0 0 CPK 1.000000 0.000000 12.000000 12.000000\n")
    file.write("mol addrep 0\n")
    file.write("mol modstyle 1 0 DynamicBonds 2.000000 0.100000 12.000000\n\n")
    file.write("## Read in beads:\n")
    file.write("mol new {" + os.path.join(folder, 'beads.xyz') +
               "} type {xyz} first 0 last -1 step 1 waitfor -1\n\n")
    file.write("## Set styles for beads - small balls, transparent\n")
    file.write("mol modstyle 0 1 CPK 0.500000 0.000000 12.000000 12.000000\n")
    file.write("mol modmaterial 0 1 Transparent\n")
    file.write("## Go to the beginning\n")
    file.write("animate goto 0\n")
    file.write("display resetview\n\n")
    file.write("## Get number of frames\n")
    file.write("set n [molinfo top get numframes]\n\n")
    file.write("## Reset views etc.\n")
    file.write("color Display Background white\n")
    file.write("color Name H silver\n")
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

    # run.sh is a bash script to run the vmd script and convert the images to
    # both a gif and a mp4.
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
    sp.run(command, shell=True, executable="bash")


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
