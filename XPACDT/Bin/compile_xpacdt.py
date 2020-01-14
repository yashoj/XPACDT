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

"""Convert XPACDT main programs to stand-alone executable using pyinstaller.
"""

import git
import inspect
import os
import subprocess as sp


if __name__ == "__main__":
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

    current_path = os.path.dirname(current_path)

    # Generate hidden imports and include required data files
    command_base = "cd $XPACDTPATH/Bin; "

    command_base += "pyinstaller --add-data '.version:.' "

    add_file = os.path.join(current_path,
                            "../Interfaces/LWAL_module/fhhfit_1050.dat")
    command_base += "--add-data '" + add_file +\
                    ":XPACDT/Interfaces/LWAL_module' "

    add_file = os.path.join(current_path,
                            "../Interfaces/LWAL_module/fhhfit_1078_new.dat")
    command_base += "--add-data '" + add_file +\
                    ":XPACDT/Interfaces/LWAL_module' "

    add_file = os.path.join(current_path,
                            "../Interfaces/CW_module/cwfit.dat")
    command_base += "--add-data '" + add_file +\
                    ":XPACDT/Interfaces/CW_module' "

    add_file = os.path.join(current_path, "helptext/*.txt")
    command_base += "--add-data '" + add_file + ":helptext' "

    command_base += " --onefile "
    command_base += "--hidden-import='git' "
    command_base += "--hidden-import='XPACDT.System.AdiabaticElectrons' "
    command_base += "--hidden-import='XPACDT.System.SurfaceHoppingElectrons' "
    command_base += "--hidden-import='XPACDT.Dynamics.MassiveAndersen' "
    command_base += "--hidden-import='XPACDT.Dynamics.VelocityVerlet' "
    command_base += "--hidden-import='XPACDT.Sampling.FixedSampling' "
    command_base += "--hidden-import='XPACDT.Sampling.QuasiclassicalSampling' "
    command_base += "--hidden-import='XPACDT.Sampling.ThermostattedSampling' "
    command_base += "--hidden-import='XPACDT.Sampling.WignerSampling' "
    command_base += "--hidden-import='XPACDT.Interfaces.BKMP2' "
    command_base += "--hidden-import='XPACDT.Interfaces.CW' "
    command_base += "--hidden-import='XPACDT.Interfaces.LWAL' "
    command_base += "--hidden-import='XPACDT.Interfaces.EckartBarrier' "
    command_base += "--hidden-import='XPACDT.Interfaces.OneDPolynomial' "
    command_base += "--hidden-import='XPACDT.Interfaces.TullyModel' "
    command_base += "--hidden-import='XPACDT.Interfaces.MorseDiabatic' "
    command_base += "--hidden-import='XPACDT.Interfaces.Morse1D' "
    command_base += "--hidden-import='XPACDT.Interfaces.Dissociation2states' "

    # For xpacdt.py
    command = command_base + "--runtime-tmpdir=\".\" -n xpacdt.exe xpacdt.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    # For genLog.py
    command = command_base + "--runtime-tmpdir=\".\" -n genLog.exe genLog.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    exit
