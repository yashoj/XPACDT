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
quantities calculated in XPACDT. Currently only position and momenta of
the centroid and the ring polymer are outputted. More needs to be added.
"""

import argparse
import pickle
import sys


def start():
    """
    Generate a human readable logfile. Currently only position and momenta of
    the centroid and the ring polymer are outputted. More needs to be added.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    i_help = "Name of the XPACDT pickle file to parse for output generation." \
        " The default is 'pickle.dat'."
    parser.add_argument("-i", "--input", type=str, dest="PickleFile",
                        required=False, help=i_help, default='pickle.dat')

    w_help = "Width for number format in output. Default 16."
    parser.add_argument("-w", "--width", type=int, dest="width",
                        required=False, help=w_help, default=16)

    p_help = "Precision for number format in output. Default 8."
    parser.add_argument("-p", "--precision", type=int, dest="prec",
                        required=False, help=p_help, default=8)

    args = parser.parse_args()

    # Formatting style
    WIDTH = args.width
    PREC = args.prec

    # Get input file
    system = pickle.load(open(args.PickleFile, 'rb'))

    # Currently just position and momenta output.
    # Add more functions for each value
    outfiles = setup_outfiles(args)

    # iterate over log
    for d in system.log:
        # for each required output, write the current line to the file
        for key, outfile in outfiles.items():
            getattr(sys.modules[__name__], "write_" + key)(d, outfile,
                                                           WIDTH, PREC)

    for outfile in outfiles.values():
        outfile.close()


def write_R(log, outfile, width, prec):
    outfile.write("{: {width}.{prec}f} ".format(log.time,
                  width=width, prec=prec))
    for x in log.x_centroid:
        outfile.write("{: {width}.{prec}f} ".format(x,
                      width=width, prec=prec))
    outfile.write(" \n")
    return


def write_P(log, outfile, width, prec):
    outfile.write("{: {width}.{prec}f} ".format(log.time,
                  width=width, prec=prec))
    for p in log.p_centroid:
        outfile.write("{: {width}.{prec}f} ".format(p,
                      width=width, prec=prec))
    outfile.write(" \n")
    return


def write_Rrp(log, outfile, width, prec):
    outfile.write("{: {width}.{prec}f} ".format(log.time,
                  width=width, prec=prec))
    for bead_position in log.positions:
        for x in bead_position:
            outfile.write("{: {width}.{prec}f} ".format(x,
                          width=width, prec=prec))
    outfile.write(" \n")
    return


def write_Prp(log, outfile, width, prec):
    outfile.write("{: {width}.{prec}f} ".format(log.time,
                  width=width, prec=prec))
    for bead_momenta in log.momenta:
        for p in bead_momenta:
            outfile.write("{: {width}.{prec}f} ".format(p,
                          width=width, prec=prec))
    outfile.write(" \n")
    return


def setup_outfiles(args):
    outfiles = {}

    # TODO parse cmd line here to create more files
    outfiles['R'] = open('R.log', 'w')
    outfiles['P'] = open('P.log', 'w')
    outfiles['Rrp'] = open('R_rp.log', 'w')
    outfiles['Prp'] = open('P_rp.log', 'w')

    return outfiles


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
