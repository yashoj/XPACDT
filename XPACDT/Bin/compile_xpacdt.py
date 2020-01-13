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

import git
import inspect
import os
import subprocess as sp


if __name__ == "__main__":

    # TODO read some stuff from command line ?

    # Get branch and version info.
    # Write to file that will be included in bundle
    current_path = os.path.abspath(inspect.getsourcefile(lambda: 0))
    repo = git.Repo(path=current_path, search_parent_directories=True)
    branch_name = repo.active_branch.name
    hexsha = repo.head.object.hexsha
    version_file = open('.version', 'w')
    version_file.write("Branch: " + branch_name + " \n")
    version_file.write("Commit: " + hexsha + " \n")
    version_file.close()

    # TODO: check for more hidden imports, check for more data files
    command = "cd $XPACDTPATH/Bin; "

    # For genLog.py - not implemented yet
#    command += "pyinstaller --add-data '.version:.' "
#    command += "--onefile "
#    command += "--hidden-import='XPACDT' "
#    command += "--runtime-tmpdir=\".\" -n genLog.exe genLog.py; "

    # For xpacdt.py
    command += "pyinstaller --add-data '.version:.' "
    command += "--add-data 'XPACDT/Interfaces/LWAL_module/fhhfit_1050.dat:XPACDT/Interfaces/LWAL_module' "
    command += "--add-data 'XPACDT/Interfaces/LWAL_module/fhhfit_1078_new.dat:XPACDT/Interfaces/LWAL_module' "
    command += " --onefile"
    command += "--hidden-import='git' "
    command += "--hidden-import='XPACDT.System.AdiabaticElectrons' "
    command += "--hidden-import='XPACDT.Dynamics.MassiveAndersen' "
    command += "--hidden-import='XPACDT.Dynamics.VelocityVerlet' "
    command += "--hidden-import='XPACDT.Sampling.FixedSampling' "
    command += "--hidden-import='XPACDT.Sampling.QuasiclassicalSampling' "
    command += "--hidden-import='XPACDT.Sampling.ThermostattedSampling' "
    command += "--hidden-import='XPACDT.Sampling.WignerSampling' "
    command += "--hidden-import='XPACDT.Interfaces.BKMP2' "
    command += "--hidden-import='XPACDT.Interfaces.EckartBarrier' "
    command += "--hidden-import='XPACDT.Interfaces.OneDPolynomial' "
    command += "--runtime-tmpdir=\".\" -n xpacdt.exe xpacdt.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    exit
