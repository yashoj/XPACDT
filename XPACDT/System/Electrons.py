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

"""This is a unfied template for implementing classes that represent the
electrons in the system. """

import sys
import warnings


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
    basis : {'adiabatic', 'diabatic'}
        Electronic state basis representation. Default: 'adiabatic'

    Attributes
    ----------
    name
    pes
    basis
    current_state
    """

    def __init__(self, name, parameters, basis='adiabatic'):

        self.__name = name
        self.basis = basis

        # Set up potential interface
        pes_name = parameters.get("system").get("Interface", None)

        if pes_name is None:
            raise KeyError("\nXPACDT: Potential energy surface interface "
                           "not specified in $system section.")

        if pes_name not in parameters:
            raise KeyError("\nXPACDT: No input parameters for chosen potential"
                           " energy surface interface.")

        __import__("XPACDT.Interfaces." + pes_name)

        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(parameters)

    @property
    def name(self):
        """str : The name of the specific electronic method implemented."""
        return self.__name

    @property
    def pes(self):
        """XPACDT.Interfaces.InterfaceTemplate : Representation of the PES."""
        return self.__pes

    @pes.setter
    def pes(self, p):
        warnings.warn("\nXPACDT: Setting the PES with the setter"
                      " should only be used in the UnitTests or to save memory"
                      " while storing. If you are currently not running a"
                      " UnitTest or storing data, something has gone WRONG!",
                      category=RuntimeWarning)

        self.__pes = p

    @property
    def basis(self):
        """{'adiabatic', 'diabatic'} : Electronic state basis
        representation."""
        return self.__basis

    @basis.setter
    def basis(self, b):
        assert (b in ['adiabatic', 'diabatic']),\
               ("Electronic state basis representation not available.")
        self.__basis = b

    @property
    def current_state(self):
        """Int : Current electronic state of the system. All beads are in same
        state for now.

        Notes
        -----
        This needs to be defined by any child class which has
        this property, if not it throws a NotImplemented Error."""

        raise NotImplementedError("This function must be implemented in all"
                                  " classes deriving from the Electrons"
                                  " template")

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
        (n_beads) ndarray of float /or/ float
        The energy of the electronic system at each bead position or at the
        centroid in au.

        Notes
        -----
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """

        raise NotImplementedError("This function must be implemented in all"
                                  " classes deriving from the Electrons"
                                  " template")

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
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
        The gradient of the electronic energy at each bead position or at the
        centroid in hartree/au.

        Notes
        -----
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """

        raise NotImplementedError("This function must be implemented in all"
                                  " classes deriving from the Electrons"
                                  " template")

    def step(self, R, P, time_propagate, **kwargs):
        """Advance the electronic subsytem by a given time.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` and momenta `P` representing the
            system nuclei in au.
        time_propagate : float
            The time to advance the electrons in au.

        Notes
        -----
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """

        raise NotImplementedError("This function must be implemented in all"
                                  " classes deriving from the Electrons"
                                  " template")

    def get_population(self, proj, basis_requested):
        """ Get electronic population for a certain adiabatic or diabatic
        state. Adiabatic populations can always be obtained. Diabatic
        populations can only be obtained for potentials that are based on a
        diabatic model.

        Parameters
        ----------
        proj : int
            State to be projected onto in the basis given by `basis_requested`.
        basis_requested : str
            Electronic basis to be used. Can be "adiabatic" or "diabatic".

        Returns
        -------
        float
            Electronic population value.

        Notes
        -----
        This function throws a NotImplemented Error and needs to be
        implemented by any child class.
        """

        raise NotImplementedError("This function must be implemented in all"
                                  " classes deriving from the Electrons"
                                  " template")
