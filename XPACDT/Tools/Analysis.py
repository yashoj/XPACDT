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
import pickle
import numpy as np

import XPACDT.Tools.Bootstrap as bootstrap

# this is all horrible!! Just testing some basic C_xx for simple systems


def do_analysis(parameters, systems=None):

    if systems is None:
        file_name = parameters.get('system').get('picklefile', 'pickle.dat')
        dirs = get_directory_list(parameters.get('system').get('folder'), file_name)
    else:
        dirs = None
        file_name = None

    t_old = None
    for system in get_systems(dirs, file_name, systems):

        # do different stuff for each command
        for key, command in parameters.commands.items():
            times = apply_command(command, system)

            # time consistency check
            if t_old is not None and not times == t_old:
                # TODO more info on which traj - how to do that?
                raise RuntimeError("The time in the trajectories is not "
                                   "aligned for command: " + key)
            t_old = times.copy()

    # todo: structure better for results
    for  key, command in parameters.commands.items():
        # bootstrap
        mm = [bootstrap.bootstrap(data, np.mean) for data in np.array(command['results']).T]

        # TODO: use file from input, add time
        # variable format....
        np.savetxt('cxx.dat',np.array(mm).reshape(-1,2))


def apply_command(command, system):
    # todo actualy implement commands
    x0 = system._log[0][1].x_centroid[0]
    command['results'].append(x0*np.array([log[1].x_centroid[0] for log in system._log]))
    return np.random.rand(10).tolist() #[log[0] for log in system._log]


def get_directory_list(folder='./', file_name=None):
    """ Get trj_ subfolders in a given folder. Condition that file needs to be there"""
    allEntries = os.listdir(folder)
    dirs = []
    for entry in allEntries:
        path = os.path.join(folder, entry)
        if entry[0:4] == 'trj_' and os.path.isdir(path):
            if file_name is None or os.path.isfile(os.path.join(path, file_name)):
                dirs.append(path)
    dirs.sort()
    return dirs


def get_systems(dirs, file_name, systems):
    if dirs is not None:
        return (pickle.load(open(os.path.join(folder_name, file_name), 'rb'))
                for folder_name in dirs)
    elif systems is not None:
        return (system for system in systems)
    else:
        raise RuntimeError("Neither dirs nor systems given!")
