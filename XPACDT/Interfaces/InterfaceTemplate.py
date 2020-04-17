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

"""This is a template for creating and unifying classes that represent a
potential in the simulations."""

import numpy as np
from scipy.optimize import minimize as spminimize

import XPACDT.Tools.Gnuplot as gnuplot


class PotentialInterface:
    """
    This is a template for creating and unifying classes that give potential
    energies, gradients and couplings that are required for the dynamics.
    This class implements several methods that are general for any interface,
    e.g., plotting, optimization, storage.

    However, the specific methods to obtain energies, gradients, etc. need to
    be implemented by the derived classes.

    Parameters
    ----------
    name : str
        The name of the specific interface implemented.
    n_dof : int
        Degrees of freedom.
    n_states : int
        Number of electronic states. Default: 1.
    max_n_beads : int, optional
        Maximum number of beads from the (n_dof) list of n_beads. Default: 1.
    primary_basis : {'adiabatic', 'diabatic'}
        Electronic state basis representation in which the interface is
        primarily based on. Default: 'adiabatic'.

    Attributes
    ----------
    name
    n_dof
    n_states
    max_n_beads
    primary_basis
    SAVE_THESHOLD
    DERIVATIVE_STEPSIZE
    """

    def __init__(self, name, n_dof, n_states=1, max_n_beads=1,
                 primary_basis='adiabatic', **kwargs):
        self.__name = name
        self.__n_dof = n_dof
        self.__n_states = n_states
        # Note that here all dof having same number of beads is assumed
        # or at least that the effective position matrix having the max. nbeads
        self.__max_n_beads = max_n_beads

        assert (primary_basis in ['adiabatic', 'diabatic']),\
               ("Electronic state basis representation not available.")
        self.__primary_basis = primary_basis

        # Threshold for when a coordinate is assumed to have changed and the
        # potential needs to be recalculated
        self.__SAVE_THRESHOLD = 1e-8
        # Used for numerical differentiation to obtain the Hessian
        self._DERIVATIVE_STEPSIZE = 1e-4

        self._is_adiabatic_calculated = False
        self._old_R = None
        self._old_P = None
        self._old_S = None

        # Stating some protected variables (starting with '_') explicitly.
        # These are energy, gradient and NAC for beads and centroid in
        # adiabatic and diabatic basis. These should not be accessed on its
        # own as there should be functions that provide them.
        #
        # _adiabatic_energy : (n_states, n_beads) ndarrays of floats
        #     Adiabatic energy of the system for all states and beads.
        # _adiabatic_gradient : (n_states, n_dof, n_beads) ndarrays of floats
        #     Gradient of adiabatic energy of the system for all states and beads.
        # _nac : (n_states, n_states, n_dof, n_beads) ndarrays of floats, optional
        #     Non-adiabatic coupling vector of the system for all states and beads.
        #
        # _adiabatic_energy_centroid : (n_states) ndarrays of floats
        #     Centroid adiabatic energy of the system for all states.
        # _adiabatic_gradient_centroid : (n_states, n_dof) ndarrays of floats
        #     Centroid gradient of adiabatic energy of the system for all states.
        # _nac_centroid : (n_states, n_states, n_dof) ndarrays of floats, optional
        #     Centroid non-adiabatic coupling vector of the system for all states.
        #
        # _diabatic_energy : (n_states, n_states, n_beads) ndarrays of floats, optional
        #     Diabatic energy of the system for all states and beads.
        # _diabatic_gradient : (n_states, n_states, n_dof, n_beads) ndarrays of floats, optional
        #     Gradient of diabatic energy of the system for all states and beads.
        #
        # _diabatic_energy_centroid : (n_states, n_states) ndarrays of floats, optional
        #     Centroid diabatic energy of the system for all states.
        # _diabatic_gradient_centroid : (n_states, n_states, n_dof) ndarrays of floats, optional
        #     Centroid gradient of diabatic energy of the system for all states.

        if (self.primary_basis == 'adiabatic'):
            self._adiabatic_energy = np.zeros((self.n_states,
                                               self.max_n_beads))
            self._adiabatic_gradient = np.zeros((self.n_states, self.n_dof,
                                                 self.max_n_beads))
            self._adiabatic_energy_centroid = np.zeros((self.n_states))
            self._adiabatic_gradient_centroid = np.zeros((self.n_states,
                                                          self.n_dof))

            if (self.n_states > 1):
                self._nac = np.zeros((self.n_states, self.n_states, self.n_dof,
                                      self.max_n_beads))
                self._nac_centroid = np.zeros((self.n_states, self.n_states,
                                               self.n_dof))

        if (self.primary_basis == 'diabatic'):
            self._diabatic_energy = np.zeros((self.n_states, self.n_states,
                                              self.max_n_beads))
            self._diabatic_gradient = np.zeros((self.n_states, self.n_states,
                                                self.n_dof, self.max_n_beads))
            self._diabatic_energy_centroid = np.zeros((self.n_states,
                                                       self.n_states))
            self._diabatic_gradient_centroid = np.zeros((self.n_states,
                                                         self.n_states,
                                                         self.n_dof))

    @property
    def name(self):
        """str : The name of the specific interface implementation."""
        return self.__name

    @property
    def n_dof(self):
        """int : Nuclear degrees of freedom."""
        return self.__n_dof

    @property
    def n_states(self):
        """int : Number of electronic states."""
        return self.__n_states

    @property
    def max_n_beads(self):
        """int : Maximum number of beads from the (n_dof) list of n_beads."""
        return self.__max_n_beads

    @property
    def primary_basis(self):
        """{'adiabatic', 'diabatic'} : Electronic state basis representation in
        which the interface is primarily based on. This should be 'diabatic' if
        the interface uses diabatic basis at all, else it is should be
        'adiabatic'. Default: 'adiabatic'.."""
        return self.__primary_basis

    @property
    def SAVE_THESHOLD(self):
        """float : Theshold for decision of using saved variables vs.
        recomputing properties."""
        return self.__SAVE_THRESHOLD

    @property
    def DERIVATIVE_STEPSIZE(self):
        """float : Step size for numerical derivatives in au."""
        return self._DERIVATIVE_STEPSIZE

    def _calculate_adiabatic_all(self, R, S=None):
        """Calculate the adiabatic energy, gradient and possibly couplings at
        the current geometry. The adiabatic energies are the eigenvalues of the
        electronic Hamiltonian.

        Parameters
        ----------
        R: (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` representing the
            system in au. The first index represents the degrees of freedom and
            the second one the beads.
        S : integer or None, default None
            The current state of the system. If None is given, the values for all
            states are calculated.

        Note
        ----
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError("This function must be implemented in all "
                                  "classes deriving from the InterfaceTemplate"
                                  " template")

    def _calculate_diabatic_all(self, R):
        """
        Calculate and set diabatic matrices for energies and gradients for
        beads and centroid.

        Parameters:
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.

        Note
        ----
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError("This function must be implemented in all "
                                  "classes deriving from the InterfaceTemplate"
                                  " template")

    def _changed(self, R, S=None):
        """Check if we need to recalculate the potential, gradients or
        couplings due to changed positions, etc.
        Currently, this only tests a change in position and this needs to be
        improved.

        Parameters
        ----------
        R: (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions `R` representing the
            system in au. The first index represents the degrees of freedom and
            the second one the beads.
        S : integer or None, default: None
            The current state of the system. Please note that currently no
            check for the state is implemented. However, at some point it might
            make sense to calculate certain quantities (like the gradient) for
            a specfic state only and then a check for a state change needs to
            be implemented.

        Returns
        -------
        True if the quantities need to be recalculated, False otherwise.
        """

        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        assert (R.shape[0] == self.n_dof), "Position array degrees of freedom"\
                                           " does not match!"
        assert (R.shape[1] == self.max_n_beads), "Position array number of"\
                                                 " beads does not match!"

        if self._old_R is None:
            self._old_R = R.copy()
            return True

        if (abs(R - self._old_R) < self.SAVE_THESHOLD).all():
            return False
        else:
            self._old_R = R.copy()
            return True

    def _recalculate_adiabatic(self, R, S=None):
        """Check if adiabatic properties need to be recalulated due to change
        in positions or since only diabatic properties were calculated. If yes,
        it calculates them.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer or None, default None
            The current state of the system. If None is given, the values for all
            states are calculated.
        """
        if self._changed(R, S):
            self._calculate_adiabatic_all(R, S)
            self._is_adiabatic_calculated = True

        elif ((self.primary_basis == 'diabatic')
              and (not(self._is_adiabatic_calculated))):
            self._get_adiabatic_from_diabatic(R, self._get_diabatic_energy_matrix)
            self._is_adiabatic_calculated = True
        else:
            pass

        return

    def adiabatic_energy(self, R, S=0, centroid=False, return_matrix=False):
        """Obtain adiabatic energy of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default: 0
            The current state of the system.
        centroid : bool, default: False
            If the energy of the centroid should be returned.
        return_matrix: bool, default: False
            If adiabatic energies for all states should be returned.

        Returns
        -------
        (n_beads) ndarray of floats if `centroid` is False
                                    and `return_matrix` is False
        /or/ float if `centroid` is True
                   and `return_matrix` is False
        /or/ (n_states, n_beads) ndarrays of floats if `centroid` is False
                                                    and `return_matrix` is True
        /or/ (n_states) ndarrays of floats if `centroid` is True
                                           and `return_matrix` is True

            The energy of the system in hartree at each bead position or at the
            centroid for a particular state or all states.
        """
        self._recalculate_adiabatic(R, S)

        if centroid:
            if return_matrix:
                return self._adiabatic_energy_centroid
            else:
                return self._adiabatic_energy_centroid[S]
        else:
            if return_matrix:
                return self._adiabatic_energy
            else:
                return self._adiabatic_energy[S]

    def adiabatic_gradient(self, R, S=0, centroid=False,
                           return_matrix=False):
        """Obtain adiabatic gradient of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default: 0
            The current state of the system.
        centroid : bool, default: False
            If the gradient of the centroid should be returned.
        return_matrix: bool, default: False
            If gradients for all states should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False
                                           and `return_matrix` is False
        /or/ (n_dof) ndarray of floats if `centroid` is True
                                       and `return_matrix` is False
        /or/ (n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False
                                                           and `return_matrix` is True
        /or/ (n_states, n_dof) ndarrays of floats if `centroid` is True
                                                  and `return_matrix` is True

            The gradient of the system in hartree/au at each bead position or
            at the centroid for a particular state or all states.
        """
        self._recalculate_adiabatic(R, S)

        if centroid:
            if return_matrix:
                return self._adiabatic_gradient_centroid
            else:
                return self._adiabatic_gradient_centroid[S]
        else:
            if return_matrix:
                return self._adiabatic_gradient
            else:
                return self._adiabatic_gradient[S]

    def nac(self, R, SI=None, SJ=None, centroid=False, return_matrix=True):
        """Obtain non-adiabatic coupling (NAC) vector of the system, labelled
        by :math:'\\overrightarrow{d}_{ij}' between electronic adiabatic states
        i and j. It is given mathematically by
        .. math::
            \\overrightarrow{d}_{ij} = \\bra{\\phi^{adiab}_i}
            \\overrightarrow{\\nabla} \\ket{\\phi^{adiab}_j}

        where math:'\\ket{\\phi^{adiab}_i}' is the i-th eigenstate.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI, SJ : integer or None, default: None
            First `SI` and second `SJ` electronic state index. If 'return_matrix'
            is False, these cannot be None.
        centroid : bool, default: False
            If NAC of centroid should be returned.
        return_matrix: bool, default: True
            If entire NAC matrix for all states should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False
                                           and `return_matrix` is False
        /or/ (n_dof) ndarray of floats if `centroid` is True
                                       and `return_matrix` is False
        /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False
                                                                     and `return_matrix` is True
        /or/ (n_states, n_states, n_dof) ndarrays of floats if `centroid` is True
                                                            and `return_matrix` is True

            NAC of the system in au at each bead position or at the centroid
            between two particular states or for all states.
        """

        assert (self.n_states > 1),\
            ("NAC is only defined for more than 1 electronic state.")

        self._recalculate_adiabatic(R, None)

        if centroid:
            if return_matrix:
                return self._nac_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining NAC.")
                return self._nac_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._nac
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining NAC.")
                return self._nac[SI, SJ]

    def diabatic_energy(self, R, SI=None, SJ=None, centroid=False,
                        return_matrix=True):
        """Obtain diabatic matrix or specific elements of it.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI, SJ : integer or None, default: None
            First `SI` and second `SJ` electronic state index. If 'return_matrix'
            is False, these cannot be None.
        centroid : bool, default False
            If the energy of the centroid should be returned.
        return_matrix: bool, default: True
            If entire diabatic energy matrix for all states should be returned.

        Returns
        -------
        (n_beads) ndarray of floats if `centroid` is False
                                    and `return_matrix` is False
        /or/ float if `centroid` is True
                   and `return_matrix` is False
        /or/ (n_states, n_states, n_beads) ndarrays of floats if `centroid` is False
                                                              and `return_matrix` is True
        /or/ (n_states, n_states) ndarrays of floats if `centroid` is True
                                                     and `return_matrix` is True

            The diabatic energy of the system in hartree at each bead position
            or at the centroid between two particular states or for all states.
        """

        assert (self.primary_basis == 'diabatic'),\
            ("Diabatic energy is not defined for the electronic basis used.")

        if self._changed(R, None):
            self._calculate_diabatic_all(R)
            self._is_adiabatic_calculated = False

        if centroid:
            if return_matrix:
                return self._diabatic_energy_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining diabatic energy.")
                return self._diabatic_energy_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._diabatic_energy
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining diabatic energy.")
                return self._diabatic_energy[SI, SJ]

    def diabatic_gradient(self, R, SI=None, SJ=None, centroid=False,
                          return_matrix=True):
        """Obtain diabatic gradient matrix or specific elements of it.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI, SJ : integer or None, default: None
            First `SI` and second `SJ` electronic state index. If 'return_matrix'
            is False, these cannot be None.
        centroid : bool, default False
            If the gradient of the centroid should be returned.
        return_matrix: bool, default: True
            If entire diabatic gradient matrix for all states should be
            returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False
                                           and `return_matrix` is False
        /or/ (n_dof) ndarray of floats if `centroid` is True
                                       and `return_matrix` is False
        /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False
                                                                     and `return_matrix` is True
        /or/ (n_states, n_states, n_dof) ndarrays of floats if `centroid` is True
                                                            and `return_matrix` is True

            The diabatic gradient of the system in hartree/au at each bead
            position or at the centroid between two particular states or for
            all states.
        """

        assert (self.primary_basis == 'diabatic'), \
            ("Diabatic gradient is not defined for the electronic basis used.")

        if self._changed(R, None):
            self._calculate_diabatic_all(R)
            self._is_adiabatic_calculated = False

        if centroid:
            if return_matrix:
                return self._diabatic_gradient_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining diabatic gradient.")
                return self._diabatic_gradient_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._diabatic_gradient
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for"
                     " obtaining diabatic gradient.")
                return self._diabatic_gradient[SI, SJ]

    def _from_internal(self, internal):
        """Transform from a defined set of internal coordinates to the actually
        used coordinates in the potential call.

        Parameters
        ----------
        internal : (n_internal) ndarray of floats
            The internal coordinates. This needs to be defined for each PES
            individually.

        Returns
        -------
        (n_dof) ndarray of floats
            The positions representing the system in au.

        Note
        ----
        This function throws and NotImplemented Error and can be
        implemented by any child class, where applicable.
        """
        raise NotImplementedError("This function can be implemented "
                                  "classes deriving from the InterfaceTemplate"
                                  " template, where applicable.")

    def _energy_wrapper(self, R, S=0, centroid=True, internal=False,
                        picture='adiabatic'):
        """Wrapper function to call energy with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer /or/ tuple of two integers, optional, default 0
            The current state of the system
            /or/ if tuple is given the current off-diagonal element of the
            diabatic matrix
        centroid : bool, default True
            If the energy of the centroid should be returned.
        internal : bool, optional, default False
            Whether 'R' is in internal coordinates. Then it is backtransformed
            to cartesian coordinates in the wrapper.
        picture : string, optional, ('adiabatic', 'diabatic')
            Whether adiabatic or diabatic PES should be returned. Currently,
            only diabatic diagonal elements are returned, i.e., SI=SJ=S.

        Returns
        -------
        float:
        The energy at the given geometry in hartree.
        """

        if picture == 'adiabatic':
            assert(type(S) == int), f"XPACDT: State given to _energy_wrapper"\
                            " not an integer: {S}."
            if internal:
                return self.adiabatic_energy(self._from_internal(R)[:, None],
                                             S, centroid)
            else:
                return self.adiabatic_energy(R[:, None], S, centroid)
        else:
            if internal:
                if type(S) == tuple:
                    return self.diabatic_energy(self._from_internal(R)[:, None],
                                                S[0], S[1], centroid)
                else:
                    return self.diabatic_energy(self._from_internal(R)[:, None],
                                                S, S, centroid)
            else:
                if type(S) == tuple:
                    return self.diabatic_energy(R[:, None], S[0], S[1], centroid)
                else:
                    return self.diabatic_energy(R[:, None], S, S, centroid)

    def _gradient_wrapper(self, R, S=0, centroid=True, internal=False, picture='adiabatic'):
        """Wrapper function to call gradient with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer, default 0 /or/ tuple of two integers
            The current state of the system
            /or/ if tuple is given the current off-diagonal element of the
            diabatic matrix
        centroid : bool, default True
            If the gradient of the centroid should be returned.
        internal : bool, optional, default False
            Whether 'R' is in internal coordinates. Then it is backtransformed
            to cartesian coordinates in the wrapper.
        picture : string, optional, ('adiabatic', 'diabatic')
            Whether adiabatic or diabatic PES should be plotted.

        Returns
        -------
        (n_dof) ndarray of floats:
        The gradient at the given geometry in hartree/au.
        """

        if picture == 'adiabatic':
            assert(type(S) == int), f"XPACDT: State given to"\
                            "  _gradient_wrapper not an integer: {S}."
            if internal:
                return self.adiabatic_gradient(self._from_internal(R)[:, None],
                                               S, centroid)
            else:
                return self.adiabatic_gradient(R[:, None], S, centroid)
        else:
            if internal:
                if type(S) == tuple:
                    return self.diabatic_gradient(self._from_internal(R)[:, None],
                                                  S[0], S[1], centroid)
                else:
                    return self.diabatic_gradient(self._from_internal(R)[:, None],
                                                  S, S, centroid)
            else:
                if type(S) == tuple:
                    return self.diabatic_gradient(R[:, None], S[0], S[1], centroid)
                else:
                    return self.diabatic_gradient(R[:, None], S, S, centroid)

    def _get_diabatic_energy_matrix(self, R):
        """
        Obtain diabatic energy matrix for beads or centroid.
        This function is needed to pass onto adiabatic transformation for
        interfaces with more than 2 electronic states. So should not be used
        independently and should only be implemented in child interface classes
        with more than 2 states.

        Parameters:
        ----------
        R : (n_dof) ndarray of floats /or/ (n_dof, n_beads) ndarray of floats
            The positions of all centroids or beads in the system.

        Returns:
        ----------
        (n_states, n_states) ndarrays of floats
                        if shape of R is (n_dof)
        /or/ (n_states, n_states, n_beads) ndarrays of floats
                        if shape of R is (n_dof, n_beads)

            Diabatic energy matrix.

        Note
        ----
        This function throws and NotImplemented Error and can be
        implemented by any child class, where applicable.
        """
        raise NotImplementedError("This function can be implemented "
                                  "classes deriving from the InterfaceTemplate"
                                  " template, where applicable.")

    def _get_adiabatic_from_diabatic(self, R, func_diabatic_energy=None):
        """
        Calculate and set adiabatic matrices for energies, gradients and
        non-adiabatic coupling (NAC) for beads and centroid using diabatic
        energies and gradients.

        Parameters:
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.
        func_diabatic_energy : function, optional, default None
            Function to get diabatic energies of shape (n_states, n_states)
            or (n_states, n_states, n_beads) ndarrays of floats. Should take
            bead or centroid positions as parameter. And should only be given
            if `self.n_states` is more than 2.
        """

        assert (self.n_states > 1), "Number of states should be 2 or more to"\
                                    " use diabatic to adiabatic"\
                                    " transformation. Here number "\
                                    "of states is: " + str(self.n_states)

        if (self.n_states == 2):
            import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad

            self._adiabatic_gradient = dia2ad.get_adiabatic_gradient(
                self._diabatic_energy, self._diabatic_gradient)

            if self.max_n_beads == 1:
                self._adiabatic_gradient_centroid = (
                    self._adiabatic_gradient.reshape((self.n_states, self.n_dof))).copy()
            else:
                self._adiabatic_gradient_centroid = dia2ad.get_adiabatic_gradient(
                    self._diabatic_energy_centroid, self._diabatic_gradient_centroid)

        else:
            import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad

            assert (func_diabatic_energy is not None), \
                   ("No function to obtain diabatic energies provided.")

            self._adiabatic_gradient = dia2ad.get_adiabatic_gradient(
                R, func_diabatic_energy, self.DERIVATIVE_STEPSIZE)

            if self.max_n_beads == 1:
                self._adiabatic_gradient_centroid = (
                    self._adiabatic_gradient.reshape((self.n_states, self.n_dof))).copy()
            else:
                r_centroid = np.mean(R, axis=1)
                self._adiabatic_gradient_centroid = dia2ad.get_adiabatic_gradient(
                    r_centroid, func_diabatic_energy, self.DERIVATIVE_STEPSIZE)

        # Bead part
        self._adiabatic_energy = dia2ad.get_adiabatic_energy(self._diabatic_energy)
        self._nac = dia2ad.get_NAC(self._diabatic_energy,
                                   self._diabatic_gradient)

        # Centroid part
        if self.max_n_beads == 1:
            self._adiabatic_energy_centroid = (
                self._adiabatic_energy.reshape(self.n_states)).copy()
            self._nac_centroid = (
                self._nac.reshape((self.n_states, self.n_states, self.n_dof))).copy()
        else:
            self._adiabatic_energy_centroid = dia2ad.get_adiabatic_energy(self._diabatic_energy_centroid)
            self._nac_centroid = dia2ad.get_NAC(
                self._diabatic_energy_centroid, self._diabatic_gradient_centroid)

        return

    def minimize_geom(self, R0, S=0, picture='adiabatic'):
        """Find the potential minimum employing the Newton-CG method
        as implemented in Scipy.

        Parameters
        ----------
        R0 : (n_dof) ndarray of floats
             The starting position for the minimization.
        S : integer, optional, Default: 0
            The state to be minimized.
        picture : string, optional, ('adiabatic', 'diabatic')
            Whether adiabatic or diabatic PES should be minimized.

        Returns
        ----------
        fun : float
              The minimal potential value in hartree.
        x : (n_dof) ndarray of floats
           The minimized position in au.

        Raises a RuntimeError if unsuccessful.
        """

        old_thresh = self.__SAVE_THRESHOLD
        self.__SAVE_THRESHOLD = 1e-15
        results = spminimize(lambda x : self._energy_wrapper(x, S=S, picture=picture),
                             R0, method='Newton-CG',
                             jac=lambda x : self._gradient_wrapper(x, S=S, picture=picture))
        self.__SAVE_THRESHOLD = old_thresh

        if results.success:
            return results.fun, results.x
        else:
            raise RuntimeError("Minimization of PES failed with message: "
                               + results.message)

    def find_ts(self, R0):
        """ Find the transition state. """
        raise NotImplementedError("Please implement the transition state"
                                  " search in the InterfaceTemplate.")

    def plot_1D(self, R, dof_i, start, end, step,
                relax=False, internal=False, S=0, picture='adiabatic'):
        """Generate data to plot a potential energy surface along one
        dimension. The other degrees of freedom can either kept fix or can be
        optimized. The results are writte to a file called 'pes_1d.dat' or
        'pes_1d_opti.dat'. The file is formatted as follows. The first column
        gives the points along the scanned coordinate. The second column the
        energy. If relax is True then the full optimized geometry is given
        after the energy.
        A gnuplot template file is also generated to help with plotting the
        data file.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            Starting position for plotting. It either defines the fixed
            positions or the starting point for optimizations.
        dof_i : integer
            Index of the variable that should change; indices start at 0.
        start : float
            Starting value of the coordinate that is scanned in au.
        end : float
            Ending value of the coordinate that is scanned in au.
        step : float
            Step size used along the scan in au.
        relax : bool, optional, Default: False
            Whether all other coordinates are fixed (False) or relaxed (True)
            along the scan.
        internal : bool, optional, Default: False
            Whether R is in internal coordinates and internal coordinates
            should be used throughout the plotting.
        S : integer, default 0 /or/ tuple of two integers
            The state to be plotted,
            /or/ if tuple is given the off-diagonal element of the
            diabatic matrix to be plotted.
        picture : string, optional, ('adiabatic', 'diabatic')
            Whether adiabatic or diabatic PES should be plotted.

        Returns
        -------
        Nothing, but writes a data file 'pes_1d.dat' /or/ 'pes_1d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        if dof_i >= self.n_dof:
            raise ValueError("Requested dof in plotting out of range of the"
                             " requested PES.")
        if S >= self.n_states:
            raise ValueError("Requested state in plotting out of range of the"
                             " requested PES.")
        assert (picture in ['adiabatic', 'diabatic']), "Wrong picture given"\
            " for plotting. Please give either adiabatic or diabatic."

        setup = "set xlabel 'coordinate " + str(dof_i) \
                + "' \nset ylabel 'energy / au'\n"
        gnuplot.write_gnuplot_file('./', 'pes_1d' + ('_opti' if relax else ''),
                                   setup, " using 1:2 w l ls 1 title 'TODO'",
                                   False)

        old_thresh = self.__SAVE_THRESHOLD
        self.__SAVE_THRESHOLD = 1e-15

        # Add header with parameters, etc.
        header = "Data file from 1D potential energy scan. There should be"\
            + " a gnuplot file called 'pes_1d.plt' or 'pes_1d_opti.plt'"\
            + " available to give reasonable help on plotting this file. \n"\
            + "The file was generated for the following interface: \n"\
            + self.name + " \n"\
            + "And with the following parameters: "\
            + "R = " + str(R) + " \n"\
            + "dof_i = " + str(dof_i) + " \n"\
            + "start = " + str(start) + " \n"\
            + "end = " + str(end) + " \n"\
            + "step = " + str(step) + " \n"\
            + "relax = " + str(relax) + " \n"\
            + "internal = " + str(internal) + " \n"\
            + "S = " + str(S) + " \n"\
            + "picture = " + str(picture) + " \n"

        e = []
        if relax:
            R_optimized = []
        try:
            grid = np.arange(start, end+step, step)
        except ZeroDivisionError as e:
            raise type(e)(str(e) + "\nXPACDT: Please give a non-zero step for"
                          "potential plotting.")

        for g in grid:
            if relax:
                constraint = ({'type': 'eq', 'fun': lambda x: x[dof_i] - g})
                R0 = R.copy()
                R0[dof_i] = g
                results = spminimize(lambda x : self._energy_wrapper(x, internal=internal, S=S, picture=picture),
                                     R0, method='SLSQP',
                                     constraints=constraint)

                # Store both the optimized energy and coordinates.
                if results.success:
                    R_optimized.append(results.x.copy())
                    e.append([results.fun])
                else:
                    raise RuntimeError("Minimization of PES failed with "
                                       " message: " + results.message)
            else:
                R[dof_i] = g
                e.append([self._energy_wrapper(R, internal=internal, S=S, picture=picture)])

        pes = np.insert(np.array(e), 0, grid, axis=1)
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))

        np.savetxt('pes_1d' + ('_opti' if relax else '') + '.dat', pes, header=header)

        self.__SAVE_THRESHOLD = old_thresh

    def plot_2D(self, R, dof_i, dof_j, starts, ends, steps,
                relax=False, internal=False, S=0, picture='adiabatic'):
        """Generate data to plot a potential energy surface along two
        dimensions. The other degrees of freedom can either kept fix or can be
        optimized. The results are writte to a file called 'pes_2d.dat' or
        'pes_2d_opti.dat'. The file is formatted as follows. The first and
        second column give the points along the scanned coordinates. The third
        column the energy. If relax is True then the full optimized geometry is
        given after the energy. The file is written in a blocked structure to
        be used in gnuplot, i.e., after each full scan of the second
        coordinate, an additional newline is given.
        A gnuplot template file is also generated to help with plotting the
        data file.


        Parameters
        ----------
        R : (n_dof) ndarray of floats
            Starting position for plotting. It either defines the fixed
            positions or the starting point for optimizations.
        dof_i : integer
            Index of the first variable that should change; indices start at 0.
        dof_j : integer
            Index of the second variable that should change; indices start at 0.
        starts : list of 2 floats
            Starting values of the coordinates that are scanned in au.
        ends : list of 2 floats
            Ending values of the coordinates that are scanned in au.
        steps : list of 2 floats
            Step sizes used along the scan in au.
        relax : bool, optional, Default: False
            Whether all other coordinates are fixed (False) or
            relaxed (True) along the scan.
        internal : bool, optional, Default: False
            Whether R is in internal coordinates and internal coordinates
            should be used throughout the plotting.
        S : integer, default 0 /or/ tuple of two integers
            The state to be plotted,
            /or/ if tuple is given the off-diagonal element of the
            diabatic matrix to be plotted.
        picture : string, optional, ('adiabatic', 'diabatic')
            Whether adiabatic or diabatic PES should be plotted.

        Returns
        -------
        Nothing, but writes a data file 'pes_2d.dat' or 'pes_2d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        if dof_i >= self.n_dof or dof_j >= self.n_dof:
            raise ValueError("Requested dof in plotting out of range of the"
                             " requested PES.")
        if S >= self.n_states:
            raise ValueError("Requested state in plotting out of range of the"
                             " requested PES.")
        assert (picture in ['adiabatic', 'diabatic']), "Wrong picture given"\
            " for plotting. Please give either adiabatic or diabatic."

        setup = "set xlabel 'coordinate " + str(dof_j)\
                + "' \nset ylabel 'coordinate " + str(dof_i) + "' \n"
        gnuplot.write_gnuplot_file('./', 'pes_2d' + ('_opti' if relax else ''),
                                   setup, " using 1:2:3 w l lw 3.2", True)

        old_thresh = self.__SAVE_THRESHOLD
        self.__SAVE_THRESHOLD = 1e-15

        # Add header with parameters, etc.
        header = "# Data file from 2D potential energy scan. There should be"\
            + " a gnuplot file called 'pes_2d.plt' or 'pes_2d_opti.plt'"\
            + " available to give reasonable help on plotting this file. \n"\
            + "# The file was generated for the following interface: \n"\
            + "# " + self.name + " \n"\
            + "# And with the following parameters: "\
            + "# R = " + str(R) + " \n"\
            + "# dof_i, dof_j = " + str(dof_i) + " " + str(dof_j) + " \n"\
            + "# starts = " + str(starts) + " \n"\
            + "# ends = " + str(ends) + " \n"\
            + "# steps = " + str(steps) + " \n"\
            + "# relax = " + str(relax) + " \n"\
            + "# internal = " + str(internal) + " \n"\
            + "# S = " + str(S) + " \n"\
            + "# picture = " + str(picture) + " \n"

        e = []
        if relax:
            R_optimized = []

        try:
            x = np.arange(starts[0], ends[0]+steps[0], steps[0])
            y = np.arange(starts[1], ends[1]+steps[1], steps[1])
        except ZeroDivisionError as e:
            raise type(e)(str(e) + "\nXPACDT: Please give a non-zero step for"
                          "potential plotting.")

        n_grid = len(x) * len(y)
        grid_x, grid_y = np.meshgrid(x, y)

        for g_y in y:
            for g_x in x:

                if relax:
                    constraint = ({'type': 'eq', 'fun': lambda x: x[dof_i] - g_x},
                                  {'type': 'eq', 'fun': lambda x: x[dof_j] - g_y})
                    R0 = R.copy()
                    R0[dof_i] = g_x
                    R0[dof_j] = g_y
                    results = spminimize(lambda x : self._energy_wrapper(x, internal=internal, S=S, picture=picture),
                                         R0, method='SLSQP',
                                         constraints=constraint,
                                         options={'ftol': 1e-6, 'eps': 1e-3})

                    if results.success:
                        R_optimized.append(results.x.copy())
                        e.append([results.fun])
                    else:
                        raise RuntimeError("Minimization of PES failed with"
                                           " message: " + results.message)
                else:
                    R[dof_i] = g_x
                    R[dof_j] = g_y
                    e.append([self._energy_wrapper(R, internal=internal, S=S, picture=picture)])

        grid = np.dstack((grid_y, grid_x)).reshape(n_grid, -1)
        pes = np.hstack((grid, e))
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))

        outfile = open('pes_2d' + ('_opti' if relax else '') + '.dat', 'w')
        outfile.write(header)
        for k, data in enumerate(pes):
            outfile.write(' '.join([str(x) for x in data]))
            outfile.write('\n')
            if (k+1) % len(x) == 0:
                outfile.write('\n')
        outfile.close()

        self.__SAVE_THRESHOLD = old_thresh

    def get_Hessian(self, R):
        """Calculate the Hessian at a given geometry `R`. The Hessian is
        calculated using numerical differentiation of the gradients.
        Currently, only central differences of the gradients is implemented.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            Position for which the Hessian is calculated
            .
        Returns
        -------
        H : (n_dof, n_dof) ndarray of floats
            Hessian of the potential at the given geometry.
        """

        n = len(R)
        H = np.zeros((n, n))
        R_step = R.copy()

        for i in range(len(R)):
            R_step[i] += self.DERIVATIVE_STEPSIZE
            grad_plus = self._gradient_wrapper(R_step)

            R_step[i] -= 2.0 * self.DERIVATIVE_STEPSIZE
            grad_minus = self._gradient_wrapper(R_step)

            H[i] = (grad_plus - grad_minus) / (2.0 * self.DERIVATIVE_STEPSIZE)

            R_step[i] += self.DERIVATIVE_STEPSIZE

        return H
