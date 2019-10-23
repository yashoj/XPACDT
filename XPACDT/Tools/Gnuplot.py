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

import os

"""This module implements some basic access to use gnuplot with the data
generated by XPACDT, e.g., writing of basic gnuplot scripts.
"""


def write_gnuplot_file(basepath, basename, setup, command,
                       is2D=False, nLoop=0):
    """Write a basic gnuplot file for plotting data that was generated by
    XPACDT.

    Parameters:
    -----------
    basepath: string
        The path where the gnuplot script should be put.
    basename: string
        The base filename of the generated data file, i.e., BASENAME.dat, which
        will also be used as base name for the gnuplot file, i.e.,
        BASENAME.plt, and the generated eps file, i.e., BASENAME.eps.
    setup: string
        Setup command for gnuplot that are written to the script after the
        general header and before the plotting command. This can be, e.g.,
        setting axis labels, etc.
    command: string
        The part of the gnuplot command following the data file and defining
        which columns to be used, which linestyle, etc. E.g.,
        'using 1:2 w l ls 1'.
    is2D: bool, optional, default False
        Wheter a contour plot (true) of 3d data or a normal plot is requested.
    nLoop: int, optional, default 0
        Wheter the plot should be done nLoop times for different blocks.
    """

    gnuplotfile = open(os.path.join(basepath, basename + '.plt'), "w")
    gnuplotfile.write("# Automatically generated by XPACDT"
                      " for analysis plotting with gnuplot...\n")
    gnuplotfile.write("")
    gnuplotfile.write("set term postscript eps enhanced color"
                      " font \",24\"\n\n")
    gnuplotfile.write("# Green #00C400\n")
    gnuplotfile.write("# Blue #4D99FF\n")
    gnuplotfile.write("# Orange #FF8000\n\n")
    gnuplotfile.write("# solid lines\n")
    gnuplotfile.write("set style line 1 lt 1 dt 1 lc rgb 'black' lw 2\n")
    gnuplotfile.write("set style line 2 lt 1 dt 1 lc rgb '#00C400' lw 2\n")
    gnuplotfile.write("set style line 3 lt 1 dt 1 lc rgb '#4D99FF' lw 2\n")
    gnuplotfile.write("set style line 4 lt 1 dt 1 lc rgb '#FF8000' lw 2\n\n")
    gnuplotfile.write("# long broken: (10,5,10,5) -> bad\n")
    gnuplotfile.write("# short broken: (5,10,5,10) -> ok\n")
    gnuplotfile.write("# dots: (2,10,2,10) -> ok\n")
    gnuplotfile.write("# dash dot: (5,10,2,10) -> bad\n")
    gnuplotfile.write("# dash dot dot: (5,10,2,10,2,10) -> bad\n\n")
    gnuplotfile.write("# short broken\n")
    gnuplotfile.write("set style line 11 pt 1 lt 1 dt (5,10,5,10)"
                      "lc rgb 'black' lw 3\n")
    gnuplotfile.write("set style line 12 pt 1 lt 1 dt (5,10,5,10)"
                      " lc rgb '#00C400' lw 3\n")
    gnuplotfile.write("set style line 13 pt 1 lt 1 dt (5,10,5,10)"
                      " lc rgb '#4D99FF' lw 3\n")
    gnuplotfile.write("set style line 14 pt 1 lt 1 dt (5,10,5,10)"
                      " lc rgb '#FF8000' lw 3\n\n")
    gnuplotfile.write("# dots\n")
    gnuplotfile.write("set style line 21 lt 1 dt (2,10,2,10)"
                      " lc rgb 'black' lw 3\n")
    gnuplotfile.write("set style line 22 lt 1 dt (2,10,2,10)"
                      " lc rgb '#00C400' lw 3\n")
    gnuplotfile.write("set style line 23 lt 1 dt (2,10,2,10)"
                      " lc rgb '#4D99FF' lw 3\n")
    gnuplotfile.write("set style line 24 lt 1 dt (2,10,2,10)"
                      " lc rgb '#FF8000' lw 3\n\n")
    gnuplotfile.write("# long broken\n")
    gnuplotfile.write("set style line 31 lt 1 dt (10,5,10,5)"
                      " lc rgb 'black' lw 3\n")
    gnuplotfile.write("set style line 32 lt 1 dt (10,5,10,5)"
                      " lc rgb '#00C400' lw 3\n")
    gnuplotfile.write("set style line 33 lt 1 dt (10,5,10,5)"
                      " lc rgb '#4D99FF' lw 3\n")
    gnuplotfile.write("set style line 34 lt 1 dt (10,5,10,5)"
                      " lc rgb '#FF8000' lw 3\n\n")
    gnuplotfile.write("# dash dot\n")
    gnuplotfile.write("set style line 41 lt 1 dt (5,10,2,10)"
                      " lc rgb 'black' lw 3\n")
    gnuplotfile.write("set style line 42 lt 1 dt (5,10,2,10)"
                      " lc rgb '#00C400' lw 3\n")
    gnuplotfile.write("set style line 43 lt 1 dt (5,10,2,10)"
                      " lc rgb '#4D99FF' lw 3\n")
    gnuplotfile.write("set style line 44 lt 1 dt (5,10,2,10)"
                      " lc rgb '#FF8000' lw 3\n\n")
    gnuplotfile.write("# dash dot dot\n")
    gnuplotfile.write("set style line 51 lt 1 dt (5,10,2,10,2,10)"
                      " lc rgb 'black' lw 3\n")
    gnuplotfile.write("set style line 52 lt 1 dt (5,10,2,10,2,10)"
                      " lc rgb '#00C400' lw 3\n")
    gnuplotfile.write("set style line 53 lt 1 dt (5,10,2,10,2,10)"
                      " lc rgb '#4D99FF' lw 3\n")
    gnuplotfile.write("set style line 54 lt 1 dt (5,10,2,10,2,10)"
                      " lc rgb '#FF8000' lw 3\n\n")

    gnuplotfile.write(setup)

    if is2D:
        gnuplotfile.write("unset ztics \n")
        gnuplotfile.write("unset key \n")
        gnuplotfile.write("unset title \n")
        gnuplotfile.write("set contour base \n")
        gnuplotfile.write("set view map \n")
        gnuplotfile.write("unset surface \n")
        gnuplotfile.write("set cntrparam levels incr 0.0,0.01,0.5\n")

    if nLoop > 0:
        gnuplotfile.write("do for [a=0:" + str(nLoop) + "] {\n")
        gnuplotfile.write("\toutfile = "
                          "sprintf('" + basename + "_%003.0f.eps',a)\n")
        gnuplotfile.write("\tset output outfile\n")
        gnuplotfile.write(("\tsplot '" if is2D else "\tplot '")
                          + basename + ".dat' index a " + command + "\n")
        gnuplotfile.write("}\n")
        gnuplotfile.write("system \"convert -delay 20 -loop 0 "
                          + basename + "_*.eps " + basename + ".gif\" \n")
    else:
        gnuplotfile.write("set output '" + basename + ".eps'\n")
        gnuplotfile.write(("splot '" if is2D else "plot '")
                          + basename + ".dat' " + command + "\n")
    gnuplotfile.close()
    return
