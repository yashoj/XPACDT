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

    allEntries = os.listdir(folder)
    files = []
    for entry in allEntries:
        path = os.path.normpath(os.path.join(folder, entry))
        rel_entry = os.path.relpath(path, base_path)

        if entry.endswith(suffix) and os.path.isfile(path):
            if entry not in exclusion and contains in entry:
                files.append(rel_entry)

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
    files_to_import += get_named_files(os.path.join(current_path, "../Interfaces"),
                                       base_path,
                                       exclusion=["__init__.py", "InterfaceTemplate.py"])

    # All Sampling Methods
    files_to_import += get_named_files(os.path.join(current_path, "../Sampling"),
                                       base_path,
                                       exclusion=["__init__.py", "Sampling.py"])

    # All Electrons
    files_to_import += get_named_files(os.path.join(current_path, "../System"),
                                       base_path,
                                       exclusion=["__init__.py", "System.py", "Nuclei.py", "Electrons.py"])

    # All Thermostats (Naming convention: ...Thermostat)
    files_to_import += get_named_files(os.path.join(current_path, "../Dynamics"),
                                       base_path, contains="Thermostat")

    # All Nuclei Propagators (Naming convention: ...Propagator)
    files_to_import += get_named_files(os.path.join(current_path, "../Dynamics"),
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

    interface_path = os.path.normpath(os.path.join(current_path, "../Interfaces/"))

    data_files = []
    allEntries = os.listdir(interface_path)
    for entry in allEntries:
        path = os.path.join(interface_path, entry)
        if os.path.isdir(path):
            data_files += get_named_files(path, base_path, suffix=".dat")

    data_import = ""
    for data_file in data_files:
        add_file = os.path.join(base_path, data_file)

        data_import += "--add-data '" + add_file + ":" \
            + os.path.split(data_file)[0] + "' "

    return data_import


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
    xpacdt_base_path = os.path.split(os.path.split(current_path)[0])[0]

    command_base = "cd $XPACDTPATH/Bin; "
    command_base += "pyinstaller --add-data '.version:.' "
    add_file = os.path.join(current_path, "helptext/*.txt")
    command_base += "--add-data '" + add_file + ":helptext' "

    # Include PES data files
    command_base += discover_data_files(current_path, xpacdt_base_path)

    # Generate hidden imports
    command_base += " --onefile "
    command_base += "--hidden-import='git' "
    command_base += discover_hidden_imports(current_path, xpacdt_base_path)

    # For xpacdt.py
    command = command_base + "--runtime-tmpdir=\".\" -n xpacdt.exe xpacdt.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    # For genLog.py
    command = command_base + "--runtime-tmpdir=\".\" -n genLog.exe genLog.py; "

    p = sp.Popen(command, shell=True, executable="bash")
    p.wait()

    exit
