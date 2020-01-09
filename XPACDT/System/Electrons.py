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

"""This is a template for creating and unifying classes that represent the
electrons in the system. """

import sys


class Electrons:
    """
    Template defining the electrons of the system. The template class will
    create a representation of the potential energy surface, but everything
    else needs to be implemented by the specific classes.

    Parameters
    ----------
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    basis : {'adiabatic', 'diabatic'}
        Electronic state basis representation. Default: 'adiabatic'
    """

    def __init__(self, parameters, n_beads, basis='adiabatic'):
        # Set up potential interface
        pes_name = parameters.get("system").get("Interface", None)
        __import__("XPACDT.Interfaces." + pes_name)
        
        self.basis = basis
        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(max(n_beads), **parameters.get(pes_name))

    @property
    def pes(self):
        """XPACDT.Interfaces.InterfaceTemplate : Representation of the PES."""
        return self.__pes
    
    @property
    def basis(self):
        """{'adiabatic', 'diabatic'} : Electronic state basis representation."""
        return self.__basis

    @basis.setter
    def basis(self, b):
        assert (b in ['adiabatic', 'diabatic']),\
               ("Electronic state basis representation not available.")
        self.__basis = b

    def energy(self, R, centroid=False):
        """Calculate the electronic energy at the current geometry.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def gradient(self, R, centroid=False):
        """Calculate the gradient of the electronic energy at the current
        geometry.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        centroid : bool, default False
            If the energy of the centroid should be returned.

        Returns
        -------
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def step(self, time, **kwargs):
        """Advance the electronic subsytem by a given time.

        Parameters
        ----------
        time : float
            The time to advance the electrons in au.

        Returns
        -------
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError
