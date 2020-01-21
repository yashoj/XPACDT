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
    name : str
        The name of the specific electronic method implemented.
    parameters : XPACDT.Input.Inputfile
        Dictonary-like presentation of the input file.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    basis : {'adiabatic', 'diabatic'}
        Electronic state basis representation. Default: 'adiabatic'

    Attributes
    ----------
    name
    pes
    basis
    """

    def __init__(self, name, parameters, n_beads, basis='adiabatic'):

        self.__name = name

        # Set up potential interface
        pes_name = parameters.get("system").get("Interface", None)

        assert(pes_name is not None), \
            ("Potential energy surface interface not specified.")
        assert(pes_name in parameters), \
            ("No input parameters for chosen potential energy surface interface.")

        __import__("XPACDT.Interfaces." + pes_name)

        self.basis = basis
        # TODO: 'n_beads' doesn't need to be passed as parameter but instead
        #       can be inferred from 'parameters.n_beads'; however how to use
        #       this format in unittest? Get input from file?
        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(max(n_beads), **parameters.get(pes_name))

    @property
    def name(self):
        """str : The name of the specific electronic method implemented."""
        return self.__name

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
        
    @property
    def current_state(self):
        """Int : Current electronic state of the system. All beads are in same
        state for now. This needs to be defined by any child class which has
        this property, if not it throws a NotImplemented Error."""
        raise NotImplementedError

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
        This function throws a NotImplemented Error and needs to be
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
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def step(self, R, P, time_propagate, **kwargs):
        """Advance the electronic subsytem by a given time.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        time_propagate : float
            The time to advance the electrons in au.

        Returns
        -------
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def get_population(self, proj, basis_requested):
        """ Get electronic population for a certain adiabatic or diabatic state
        regardless of whichever basis the electron uses.

        Parameters
        ----------
        proj : int
            State to be projected onto in the basis given by `basis_requested`.
        basis_requested : str
            Electronic basis to be used. Can be "adiabatic" or "diabatic".

        Returns
        -------
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError
