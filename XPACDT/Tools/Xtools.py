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

""" Module containing some functions customized to the need of XPACDT."""

import os
import pickle


def get_directory_list(folder='./', file_name=None):
    """ Get "trj_*" subfolders in a given folder. If a file name is given, only
    "trj_*" subfolders are returned that contain a file with that name. The
    returned list is sorted.

    Parameters
    ----------
    folder : string, optional, default: './'
        Folder to search for trj_ subfolders.
    file_name : string, optional, default: None
        If given, only "trj_*" subfolders are returned that contain a file with
        that name.

    Returns
    -------
    dirs : list of string
        Sorted list of "trj_*" subfolders.
    """

    allEntries = os.listdir(folder)
    dirs = []
    for entry in allEntries:
        path = os.path.join(folder, entry)
        if entry[0:4] == 'trj_' and os.path.isdir(path):
            if file_name is None or os.path.isfile(os.path.join(path,
                                                                file_name)):
                dirs.append(path)
    dirs.sort()
    return dirs


def get_systems(dirs, file_name, systems):
    """ Obtain a generator over all XPACDT.System to sweep through them. This
    is needed for example in the analysis part.
    The systems are either given as a list of systems or read from pickle
    files in the given list of folders.

    Parameters
    ----------
    dirs : list of strings
        Directories to read the pickle files from.
    file_name : String
        Name of the pickle files to be read.
    systems: list of XPACDT.System
        A list of systems to perform the analysis on. If not given, then the
        systems are read from file.

    Returns
    -------
    Generator over all sytems.
    """

    if dirs is not None:
        return (pickle.load(open(os.path.join(folder_name, file_name), 'rb'))
                for folder_name in dirs)
    elif systems is not None:
        return (system for system in systems)
    else:
        raise RuntimeError("Neither dirs nor systems given!")
