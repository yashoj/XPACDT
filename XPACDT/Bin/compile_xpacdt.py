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
from pathlib import Path
import subprocess as sp


def get_named_files(folder, base_path, suffix='.py',
                    exclusion=['__init__.py'], contains=""):
    """ Obtain a list of all files ending with a given suffix (e.g. '.py').
    The files are given relative to the XPACDT main folder, i.e.,
    XPACDT/SUBFOLDERs/FILE. A list of filenames to ignore can be given. Also a
    subtring which has to be in each file can be given.

    Parameters
    ----------
    folder : string
        Folder name to search
    base_path : string
        XPACDT base path.
    suffix : string, optional, default '.py'
        Given suffix for files to search.
    exclusion : list of string, optional, default ['__init__.py']
        A list of files to ignore.
    contains : string, optional, default ""
        A substring each file has to contain.

    Returns
    list of string:
        A list containing all files with the given suffix, containing the
        given substring and not in the exclusion list. File paths are given
        relative to the XPACDT base path given.
    """

    files = []
    for suffix_math in Path(folder).resolve().glob('*' + suffix):
        relative_path = suffix_math.relative_to(Path(base_path).resolve())
        filename = relative_path.parts[-1]

        if filename not in exclusion and contains in filename:
            files.append(str(relative_path))

    return files


def discover_hidden_imports(current_path, base_path):
    """ Discover hidden imports for packaging XPACDT and generate the required
    string containing the command line options for PyInstaller. Currently, the
    following imports will be used: All *py files in "Interfaces" except the
    InterfaceTemplate, all SamplingApproaches in "Sampling". All Electron
    Implementations in "System". All Thermostats in "Dynamics" that contain
    'Thermostat' and all nuclei propagators in "Dynamics" that contain
    'Propagator'.

    Parameters
    ----------
    current_path : string
        The path to the folder containing this file. It is used to obtain
        relative paths for the file discovery.
    base_path : string
        The XPACDT base path.

    Returns
    -------
    string:
         A string containing all the command line arguments for hidden imports
         for running the PyInstaller.
    """
    files_to_import = []
    # All Interfaces
    files_to_import += get_named_files(Path(current_path, "../Interfaces"),
                                       base_path,
                                       exclusion=["__init__.py",
                                                  "InterfaceTemplate.py"])

    # All Sampling Methods
    files_to_import += get_named_files(Path(current_path, "../Sampling"),
                                       base_path,
                                       exclusion=["__init__.py",
                                                  "Sampling.py"])

    # All Electrons
    files_to_import += get_named_files(Path(current_path, "../System"),
                                       base_path,
                                       exclusion=["__init__.py", "System.py",
                                                  "Nuclei.py", "Electrons.py"])

    # All Thermostats (Naming convention: ...Thermostat)
    files_to_import += get_named_files(Path(current_path, "../Dynamics"),
                                       base_path, contains="Thermostat")

    # All Nuclei Propagators (Naming convention: ...Propagator)
    files_to_import += get_named_files(Path(current_path, "../Dynamics"),
                                       base_path, contains="Thermostat")

    import_base = ""
    for filename in files_to_import:
        import_base += "--hidden-import='" \
            + filename[:-3].replace("/", ".") + "' "

    return import_base


def discover_data_files(current_path, base_path):
    """ Discover interface data files and generate the required string
    containing the command line options for PyInstaller to include them in the
    final output file. Data files have to end with '.dat' to be included.

    Parameters
    ----------
    current_path : string
        The path to the folder containing this file. It is used to obtain
        relative paths for the file discovery.
    base_path : string
        The XPACDT base path.

    Returns
    -------
    string:
         A string containing all the command line arguments for added data
         files for the PyInstaller.
    """

    interface_path = Path(current_path, '../Interfaces/').resolve()
    allEntries = [x for x in interface_path.iterdir() if x.is_dir()]

    data_files = []
    for path in allEntries:
        data_files += get_named_files(path, base_path, suffix=".dat")

    data_import = ""
    for data_file in data_files:
        add_file = Path(base_path, data_file)

        data_import += "--add-data '" + str(add_file) + ":" \
            + str(Path(data_file).parent) + "' "

    return data_import


if __name__ == "__main__":
    # Get branch and version info.
    # Write to file that will be included in bundle
    current_path = Path(inspect.getsourcefile(lambda: 0)).resolve()

#    current_path = os.path.abspath()
    repo = git.Repo(path=current_path, search_parent_directories=True)
    branch_name = repo.active_branch.name
    hexsha = repo.head.object.hexsha
    version_file = open('.version', 'w')
    version_file.write("Branch: " + branch_name + " \n")
    version_file.write("Commit: " + hexsha + " \n")
    version_file.close()

    current_path = current_path.parent
    xpacdt_base_path = current_path.parent.parent

    command_base = "cd $XPACDTPATH/Bin; "
    command_base += "pyinstaller --add-data '.version:.' "
    add_file = str(Path(current_path, "helptext/*.txt"))
    command_base += "--add-data '" + add_file + ":helptext' "

    # Include PES data files
    command_base += discover_data_files(current_path, xpacdt_base_path)

    # Generate hidden imports
    command_base += " --onefile "
    command_base += "--hidden-import='git' "
    command_base += discover_hidden_imports(current_path, xpacdt_base_path)

    # For xpacdt.py
    command = command_base + "--runtime-tmpdir=\".\" -n xpacdt.exe xpacdt.py; "

    sp.run(command, shell=True, executable="bash")

    # For genLog.py
    command = command_base + "--runtime-tmpdir=\".\" -n genLog.exe genLog.py; "

    sp.run(command, shell=True, executable="bash")

    exit(0)
