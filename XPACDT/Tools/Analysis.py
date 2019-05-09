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

### this is all horrible!! Just testing some basic C_xx for simple systems

def do_analysis(parameters):
    dirs = get_directory_list(parameters.get('system').get('folder'))
    file_name = parameters.get('system').get('picklefile', 'pickle.dat')
    cxxs = []
    for folder_name in dirs:
        path_file = os.path.join(folder_name, file_name)
        if os.path.isfile(path_file):
            system = pickle.load(open(path_file, 'rb'))
            
        x0 = system._log[0]['nuclei'].x_centroid[0]
        cxxs.append(x0*np.array([log['nuclei'].x_centroid[0] for log in system._log]))
        
    cxx = np.average(cxxs, axis=0)
    np.savetxt('cxx.dat', cxx)

def get_directory_list(folder='./'):
    allEntries = os.listdir(folder)
    dirs = []
    for entry in allEntries:
        if entry[0:4] == 'trj_' and os.path.isdir(folder + entry):
            dirs.append(folder + entry)
    dirs.sort()
    return dirs