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
#  Yashoj Shakya, DESY, <yashoj.shakya@desy.de>
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
quantities calculated in XPACDT. Implemented are the following:

Generated always:
- Centroid and ring polymer positions
- Centroid and ring polymer momenta

Optional (through command line arguments):
- Current electronic state (if available)
"""

import argparse
import pickle
import sys


def start():
    """
    Generate a human readable logfile.
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

    s_help = "Generate current electronic state logfile. Default: False."
    parser.add_argument("-s", "--state", action="store_true", dest="state",
                        required=False, help=s_help, default=False)

    e_help = "Generate energy logfile. Default: False."
    parser.add_argument("-e", "--energy", action="store_true", dest="energy",
                        required=False, help=e_help, default=False)

    args = parser.parse_args()

    # Formatting style
    WIDTH = args.width
    PREC = args.prec

    # Get input file
    system = pickle.load(open(args.PickleFile, 'rb'))

    # Generate appropriate output files
    outfiles = setup_outfiles(args)

    # iterate over log
    for logged_nuclei in system.log:
        # for each required output, write the current line to the file
        for key, outfile in outfiles.items():
            getattr(sys.modules[__name__], "write_" + key)(logged_nuclei,
                                                           outfile,
                                                           WIDTH, PREC)

    for outfile in outfiles.values():
        outfile.close()


def write_R(log_nuclei, outfile, width, prec):
    """ Write centroid position to outfile with given precision.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    for x in log_nuclei.x_centroid:
        outfile.write("{: {width}.{prec}f} ".format(x,
                      width=width, prec=prec))
    outfile.write(" \n")
    return


def write_P(log_nuclei, outfile, width, prec):
    """ Write centroid momentum to outfile with given precision.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    for p in log_nuclei.p_centroid:
        outfile.write("{: {width}.{prec}f} ".format(p,
                      width=width, prec=prec))
    outfile.write(" \n")
    return


def write_Rrp(log_nuclei, outfile, width, prec):
    """ Write ring polymer bead positions to outfile with given precision.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    for bead_position in log_nuclei.positions:
        for x in bead_position:
            outfile.write("{: {width}.{prec}f} ".format(x,
                          width=width, prec=prec))
    outfile.write(" \n")
    return


def write_Prp(log_nuclei, outfile, width, prec):
    """ Write ring polymer bead momenta to outfile with given precision.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    for bead_momenta in log_nuclei.momenta:
        for p in bead_momenta:
            outfile.write("{: {width}.{prec}f} ".format(p,
                          width=width, prec=prec))
    outfile.write(" \n")
    return


def write_electronic_state(log_nuclei, outfile, width, prec):
    """ Write current electronic state information to outfile with
    given precision.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    outfile.write("{: {width}d} ".format(log_nuclei.electrons.current_state,
                  width=width))
    outfile.write(" \n")
    return


def write_energy(log_nuclei, outfile, width, prec):
    """ Write centroid energy to outfile with given precision. The columns
    represent time in au, centroid kinetic energy in au, centroid potential
    energy in au and total centroid energy in au.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.kinetic_energy_centroid,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.potential_energy_centroid,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.energy_centroid,
                  width=width, prec=prec))
    outfile.write(" \n")
    return


def write_energy_rp(log_nuclei, outfile, width, prec):
    """ Write ring polymer energy to outfile with given precision. The columns
    represent time in au, ring polymer kinetic energy in au, energy of spring
    terms of the ring polymer, potential energy in au and total ring polymer
    energy in au.

    Paramters
    ---------
    log_nuclei : XPACDT.System.Nuclei
        Current nuclei object (from system.log).
    outfile : file object
        Opened log file to be written to.
    width, prec : integers
        Width and precicion for formatting the output.
    """
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.time,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.kinetic_energy,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.spring_energy,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.potential_energy,
                  width=width, prec=prec))
    outfile.write("{: {width}.{prec}f} ".format(log_nuclei.energy,
                  width=width, prec=prec))
    outfile.write(" \n")
    return


def setup_outfiles(args):
    """Open output files based on command line arguments.

    Parameters
    ----------
    args : ArgumentParser object
        Command line arguments from argpars.parse_args().
    """
    outfiles = {}

    outfiles['R'] = open('R.log', 'w')
    outfiles['P'] = open('P.log', 'w')
    outfiles['Rrp'] = open('R_rp.log', 'w')
    outfiles['Prp'] = open('P_rp.log', 'w')

    # Check command line arguments for additional requests
    if (args.state):
        outfiles['electronic_state'] = open('state.log', 'w')
    if (args.energy):
        outfiles['energy'] = open('energy.log', 'w')
        outfiles['energy_rp'] = open('energy_rp.log', 'w')

    return outfiles


# This is a wrapper for the setup function!
if __name__ == "__main__":
    start()
    exit
