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

"""This is a basic implementation of adiabatic electrons, i.e., no electron
dynamics is taken into account."""

import XPACDT.System.Electrons as electrons


class SurfaceHoppingElectrons(electrons.Electrons):
    """ Surface hopping electrons, i.e., no electron dynamics and only the PES
    gradients and energies are passed on.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    """

    def __init__(self, parameters, n_beads):
        basis = parameters.get("SurfaceHoppingElectrons").get("basis", "Adiabatic")

        electrons.Electrons.__init__(self, parameters, n_beads, basis)
