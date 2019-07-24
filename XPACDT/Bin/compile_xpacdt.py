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

"""Convert XPACDT to an packed executable using pyinstaller.
"""

import subprocess as sp


if __name__ == "__main__":

    # TODO read some stuff from command line

    command = "cd $XPACDTPATH/Bin; "
    command += "pyinstaller --onefile --runtime-tmpdir=\".\" -n genLog.exe genLog.py; "
    command += "pyinstaller --onefile --runtime-tmpdir=\".\" -n xpacdt.exe xpacdt.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    exit

# --hidden-import='git'
# --hidden-import='git'
