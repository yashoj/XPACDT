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

"""This is a template for creating and unifying classes that represent a
potential in the simulations."""

import numpy as np
from scipy.optimize import minimize as spminimize

# TODO: what to store
# TODO: which general functions to implement.


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
        Number of electronic states.
    max_n_beads : int
        Maximum number of beads from the (n_dof) list of n_beads.
    bases_used : {'adiabatic', 'diabatic', 'dia2ad'}
        Electronic state basis representations to be used. Default: 'adiabatic'.

    Attributes
    ----------
    name
    n_dof
    n_states
    max_n_beads
    """

    def __init__(self, name, n_dof, max_n_beads, n_states=1,
                 bases_used='adiabatic', **kwargs):
        self.__name = name
        self.__n_dof = n_dof
        self.__n_states = n_states
        # Note that here all dof having same number of beads is assumed
        # or at least that the effective position matrix having the max. nbeads
        self.__max_n_beads = max_n_beads
        self.bases_used = bases_used
        self.__SAVE_THRESHOLD = 1e-8

        self._old_R = None
        self._old_P = None
        self._old_S = None

        # TODO: where to place these? Is here fine?
#         _adiabatic_energy : (n_states, n_beads) ndarrays of floats
#            Adiabatic energy of the system for all states and beads.
#        _adiabatic_gradient : (n_states, n_dof, n_beads) ndarrays of floats
#            Gradient of adiabatic energy of the system for all states and beads.
#        _nac : (n_states, n_states, n_dof, n_beads) ndarrays of floats, optional
#            Non-adiabatic coupling vector of the system for all states and beads.
#    
#        _adiabatic_energy_centroid : (n_states) ndarrays of floats
#            Centroid adiabatic energy of the system for all states.
#        _adiabatic_gradient_centroid : (n_states, n_dof) ndarrays of floats
#            Centroid gradient of adiabatic energy of the system for all states.
#        _nac_centroid : (n_states, n_states, n_dof) ndarrays of floats, optional
#            Centroid non-adiabatic coupling vector of the system for all states.
#    
#        _diabatic_energy : (n_states, n_states, n_beads) ndarrays of floats, optional
#            Diabatic energy of the system for all states and beads.
#        _diabatic_gradient : (n_states, n_states, n_dof, n_beads) ndarrays of floats, optional
#            Gradient of diabatic energy of the system for all states and beads.
#    
#        _diabatic_energy_centroid : (n_states, n_states) ndarrays of floats, optional
#            Centroid diabatic energy of the system for all states.
#        _diabatic_gradient_centroid : (n_states, n_states, n_dof) ndarrays of floats, optional
#            Centroid gradient of diabatic energy of the system for all states.

        if (self.bases_used == 'adiabatic') or (self.bases_used == 'dia2ad'):
            self._adiabatic_energy = np.zeros((self.n_states, self.max_n_beads))
            self._adiabatic_gradient = np.zeros((self.n_states, self.n_dof,
                                       self.max_n_beads))
            self._adiabatic_energy_centroid = np.zeros((self.n_states))
            self._adiabatic_gradient_centroid = np.zeros((self.n_states, self.n_dof))

            if (self.n_states > 1):
                self._nac = np.zeros((self.n_states, self.n_states, self.n_dof,
                                      self.max_n_beads))
                self._nac_centroid = np.zeros((self.n_states, self.n_states,
                                               self.n_dof))

        if (self.bases_used == 'diabatic') or (self.bases_used == 'dia2ad'):
            self._diabatic_energy = np.zeros((self.n_states, self.n_states,
                                              self.max_n_beads))
            self._diabatic_gradient = np.zeros((self.n_states, self.n_states,
                                                self.n_dof, self.max_n_beads))
            self._diabatic_energy_centroid = np.zeros((self.n_states,
                                                       self.n_states))
            self._diabatic_gradient_centroid =  np.zeros((self.n_states, self.n_states, self.n_dof))

    @property
    def name(self):
        """str : The name of the specific interface implementation."""
        return self.__name

    @property
    def n_dof(self):
        """int : Degrees of freedom."""
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
    def bases_used(self):
        """{'adiabatic', 'diabatic', 'dia2ad'} : Electronic state basis representations to be used."""
        return self.__bases_used

    @bases_used.setter
    def bases_used(self, b):
        assert (b in ['adiabatic', 'diabatic', 'dia2ad']),\
               ("Electronic state basis representation not available.")
        self.__bases_used = b

    @property
    def SAVE_THESHOLD(self):
        """float : Theshold for decision of using saved variables vs.
        recomputing properties."""
        return self.__SAVE_THRESHOLD

    @property
    def DERIVATIVE_STEPSIZE(self):
        """float : Step size for numerical derivatives in au."""
        return 1e-4

    def _calculate_adiabatic_all(self, R, P, S=None):
        """Calculate the adiabatic energy, gradient and possibly couplings at
        the current geometry. The adiabatic energies are the eigenvalues of the
        electronic Hamiltonian.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) momenta representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        S : integer, default None
            The current state of the system.

        Returns
        -------
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def _calculate_diabatic_all(self, R):
        """
        Calculate and set diabatic matrices for energies and gradients for
        beads and centroid.

        Parameters:
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The positions of all beads in the system. The first axis is the
            degrees of freedom and the second axis the beads.

        Returns
        -------
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def _changed(self, R, P, S=None):
        """Check if we need to recalculate the potential, gradients or
        couplings due to changed positions, etc.
        Currently, this only tests a change in position and this needs to be
        improved.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        P : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) momenta representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        S : integer, default None
            The current state of the system.

        Returns
        -------
        True if the quantities need to be recalculated, False otherwise.
        """
        # TODO: Where to place asserts so that they are only checked once in the beginning?
        #       Do not need to check them every time right?
        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        assert (R.shape[0] == self.n_dof), "Position array degrees of freedom does not match!"
        assert (R.shape[1] == self.max_n_beads), "Position array number of beads does not match!"
        if P is not None:
            assert (isinstance(P, np.ndarray)), "P not a numpy array!"
            assert (P.ndim == 2), "Momentum array not two-dimensional!"
            assert (P.dtype == 'float64'), "Momentum array not real!"
            assert (P.shape[0] == self.n_dof), "Momentum array degrees of freedom does not match!"
            assert (P.shape[1] == self.max_n_beads), "Momentum array number of beads does not match!"

        if self._old_R is None:
            self._old_R = R.copy()
            return True
        if (abs(R - self._old_R) < self.SAVE_THESHOLD).all():
            return False
        else:
            self._old_R = R.copy()
            return True

    def adiabatic_energy(self, R, S=None, centroid=False, return_matrix=False):
        """Obtain adiabatic energy of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default: None
            The current state of the system.
        centroid : bool, default: False
            If the energy of the centroid should be returned.
        return_matrix: bool, default: False
            If adiabatic energies for all states should be returned.

        Returns
        -------
        (n_beads) ndarray of floats if `centroid` is False and `return_matrix` is False 
        /or/ float if `centroid` is True and `return_matrix` is False
        /or/ (n_states, n_beads) ndarrays of floats if `centroid` is False and `return_matrix` is True
        /or/ (n_states) ndarrays of floats if `centroid` is True and `return_matrix` is True
            The energy of the system in hartree at each bead position or at the
            centroid for a particular state or all states.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)

        if centroid:
            if return_matrix:
                return self._adiabatic_energy_centroid
            else:
                return self._adiabatic_energy_centroid[0 if S is None else S]
        else:
            if return_matrix:
                return self._adiabatic_energy
            else:
                return self._adiabatic_energy[0 if S is None else S]

    def adiabatic_gradient(self, R, S=None, centroid=False, return_matrix=False):
        """Obtain adiabatic gradient of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default: None
            The current state of the system.
        centroid : bool, default: False
            If the gradient of the centroid should be returned.
        return_matrix: bool, default: False
            If gradients for all states should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False and `return_matrix` is False 
        /or/ (n_dof) ndarray of floats if `centroid` is True and `return_matrix` is False
        /or/ (n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False and `return_matrix` is True
        /or/ (n_states, n_dof) ndarrays of floats if `centroid` is True and `return_matrix` is True
            The gradient of the system in hartree/au at each bead position or
            at the centroid for a particular state or all states.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)
            
        if centroid:
            if return_matrix:
                return self._adiabatic_gradient_centroid
            else:
                return self._adiabatic_gradient_centroid[0 if S is None else S]
        else:
            if return_matrix:
                return self._adiabatic_gradient
            else:
                return self._adiabatic_gradient[0 if S is None else S]

    def nac(self, R, SI=None, SJ=None, centroid=False, return_matrix=False):
        """Obtain non-adiabatic coupling (NAC) vector of the system, labelled
        by :math:'\\overrightarrow{d}_{ij}' between electronic adiabatic states
        i and j. It is given mathematically by
        .. math::
            \\overrightarrow{d}_{ij} = \\bra{\\phi^{adiab}_i} \\overrightarrow{\\nabla} \\ket{\\phi^{adiab}_j}

        where math:'\\ket{\\phi^{adiab}_i}' is the i-th eigenstate.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI : integer, default: None
            First electronic state index.
        SJ : integer, default: None
            Second electronic state index.
        centroid : bool, default: False
            If NAC of centroid should be returned.
        return_matrix: bool, default: False
            If entire NAC matrix for all states should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False and `return_matrix` is False 
        /or/ (n_dof) ndarray of floats if `centroid` is True and `return_matrix` is False
        /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False and `return_matrix` is True
        /or/ (n_states, n_states, n_dof) ndarrays of floats if `centroid` is True and `return_matrix` is True
            NAC of the system in au at each bead position or at the centroid
            between two particular states or for all states.
        """
        # Is this assert statement even needed? nac isn't initialized if
        # this isn't fulfilled anyways?
        assert (self.n_states > 1),\
            ("NAC is only defined for more than 1 electronic state.")

        if self._changed(R, None, None):
            self._calculate_all(R, None, None)

        if centroid:
            if return_matrix:
                return self._nac_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining NAC.")
                return self._nac_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._nac
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining NAC.")
                return self._nac[SI, SJ]

    # TODO: should these wrappers have option to return adiabatic or diabatic
    #       energies, in case we decide to plot either one? Or maybe create separate functions for them?
    def _energy_wrapper(self, R, S=0, centroid=True):
        """Wrapper function to do call energy with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.
        centroid : bool, default True
            If the energy of the centroid should be returned.

        Returns
        -------
        float:
        The energy at the given geometry in hartree.
        """
        return self.adiabatic_energy(R.reshape(-1, 1), S, centroid)

    def _gradient_wrapper(self, R, S=0, centroid=True):
        """Wrapper function to do call gradient with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.
        centroid : bool, default True
            If the gradient of the centroid should be returned.

        Returns
        -------
        (n_dof) ndarray of floats:
        The gradient at the given geometry in hartree/au.
        """
        return self.adiabatic_gradient(R.reshape(-1, 1), S, centroid)

    def diabatic_energy(self, R, SI=None, SJ=None, centroid=False, return_matrix=False):
        """Obtain diabatic energy matrix element of the system.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI : integer, default: None
            First electronic state index.
        SJ : integer, default: None
            Second electronic state index.
        centroid : bool, default False
            If the energy of the centroid should be returned.
        return_matrix: bool, default: False
            If entire diabatic energy matrix for all states should be returned.

        Returns
        -------
        (n_beads) ndarray of floats if `centroid` is False and `return_matrix` is False 
        /or/ float if `centroid` is True and `return_matrix` is False
        /or/ (n_states, n_states, n_beads) ndarrays of floats if `centroid` is False and `return_matrix` is True
        /or/ (n_states, n_states) ndarrays of floats if `centroid` is True and `return_matrix` is True
            The diabatic energy of the system in hartree at each bead position
            or at the centroid between two particular states or for all states.
        """
        # Is this assert statement even needed? Diabatic energy isn't initialized if
        # this isn't fulfilled anyways?
        assert ((self.bases_used == 'diabatic') or (self.bases_used == 'dia2ad')),\
            ("Diabatic energy is not defined for the electronic basis used.")

        if self._changed(R, None, None):
            self._calculate_all(R, None, None)
        
        if centroid:
            if return_matrix:
                return self._diabatic_energy_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining diabatic energy.")
                return self._diabatic_energy_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._diabatic_energy
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining diabatic energy.")
                return self._diabatic_energy[SI, SJ]

    def diabatic_gradient(self, R, SI=None, SJ=None, centroid=False, return_matrix=False):
        """Obtain diabatic gradient matrix element of the system.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        SI : integer, default: None
            First electronic state index.
        SJ : integer, default: None
            Second electronic state index.
        centroid : bool, default False
            If the gradient of the centroid should be returned.
        return_matrix: bool, default: False
            If entire diabatic gradient matrix for all states should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats if `centroid` is False and `return_matrix` is False 
        /or/ (n_dof) ndarray of floats if `centroid` is True and `return_matrix` is False
        /or/ (n_states, n_states, n_dof, n_beads) ndarrays of floats if `centroid` is False and `return_matrix` is True
        /or/ (n_states, n_states, n_dof) ndarrays of floats if `centroid` is True and `return_matrix` is True
            The diabatic gradient of the system in hartree/au at each bead
            position or at the centroid between two particular states or for
            all states.
        """
        # Is this assert statement even needed? Diabatic gradient isn't initialized if
        # this isn't fulfilled anyways?
        assert ((self.bases_used == 'diabatic') or (self.bases_used == 'dia2ad')),\
            ("Diabatic gradient is not defined for the electronic basis used.")

        if self._changed(R, None, None):
            self._calculate_all(R, None, None)
            
        if centroid:
            if return_matrix:
                return self._diabatic_gradient_centroid
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining diabatic gradient.")
                return self._diabatic_gradient_centroid[SI, SJ]
        else:
            if return_matrix:
                return self._diabatic_gradient
            else:
                assert ((SI is not None) and (SJ is not None)), \
                    ("State labels or return matrix not specified for obtaining diabatic gradient.")
                return self._diabatic_gradient[SI, SJ]

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
        func_diabatic_energy : function
            Function to get diabatic energies of shape (n_states, n_states)
            or (n_states, n_states, n_beads) ndarrays of floats. Should take
            bead or centroid positions as parameter.
        """

        if self.n_states == 2:
            import XPACDT.Tools.DiabaticToAdiabatic_2states as dia2ad

            self._adiabatic_gradient = dia2ad.get_adiabatic_gradient(
                self._diabatic_energy, self._diabatic_gradient)
            self._adiabatic_gradient_centroid = dia2ad.get_adiabatic_gradient(
                self._diabatic_energy_centroid, self._diabatic_gradient_centroid)

        elif self.n_states == 3:
            import XPACDT.Tools.DiabaticToAdiabatic_Nstates as dia2ad

            assert (func_diabatic_energy is not None), \
                   ("No function to obtain diabatic energies provided.")

            self._adiabatic_gradient = dia2ad.get_adiabatic_gradient(
                R, func_diabatic_energy, self.DERIVATIVE_STEPSIZE)

            r_centroid = np.mean(R, axis=1)
            self._adiabatic_gradient_centroid = dia2ad.get_adiabatic_gradient(
                r_centroid, func_diabatic_energy, self.DERIVATIVE_STEPSIZE)

        self._adiabatic_energy = dia2ad.get_adiabatic_energy(self._diabatic_energy)
        self._adiabatic_energy_centroid = dia2ad.get_adiabatic_energy(self._diabatic_energy_centroid)

        self._nac = dia2ad.get_NAC(self._diabatic_energy, self._diabatic_gradient)
        self._nac_centroid = dia2ad.get_NAC(
            self._diabatic_energy_centroid, self._diabatic_gradient_centroid)

    def minimize_geom(self, R0):
        """Find the potential minimum employing the Newton-CG method
        as implemented in Scipy.

        Parameters
        ----------
        R0 : (n_dof) ndarray of floats
             The starting position for the minimization.

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
        results = spminimize(self._energy_wrapper, R0, method='Newton-CG',
                             jac=self._gradient_wrapper)
        self.__SAVE_THRESHOLD = old_thresh

        if results.success:
            return results.fun, results.x
        else:
            raise RuntimeError("Minimization of PES failed with message: "
                               + results.message)

    def find_ts(self, R0):
        """ Find the transition state. """
        raise NotImplementedError

    def plot_1D(self, R, dof_i, start, end, step, relax=False, S=0):
        """Generate data to plot a potential energy surface along one
        dimension. The other degrees of freedom can either kept fix or can be
        optimized. The results are writte to a file called 'pes_1d.dat' or
        'pes_1d_opti.dat'. The file is formatted as follows. The first column
        gives the points along the scanned coordinate. The second column the
        energy. If relax is True then the full optimized geometry is given
        after the energy.

        TODO: Implement the use of a certain excited state.
        TODO: add variable file name

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            Starting position for plotting. It either defines the fixed
            positions or the starting point for optimizations.
        dof_i : integer
            Index of the variable that should change.
        start : float
            Starting value of the coordinate that is scanned in au.
        end : float
            Ending value of the coordinate that is scanned in au.
        step : float
            Step size used along the scan in au.
        relax : bool, optional, Default: False
            Whether all other coordinates are fixed (False) or relaxed (True)
            along the scan.
        S : integer, optional, Default: 0
            Could be the state, not impemented yet.

        Returns
        -------
        Nothing, but writes a data file 'pes_1d.dat' or 'pes_1d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        # TODO: add some asserts
        e = []
        if relax:
            R_optimized = []
        grid = np.arange(start, end+step, step)

        for g in grid:
            if relax:
                constraint = ({'type': 'eq', 'fun': lambda x: x[dof_i] - g})
                R0 = R.copy()
                R0[dof_i] = g
                results = spminimize(self._energy_wrapper, R0, method='SLSQP',
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
                e.append([self._energy_wrapper(R)])

        pes = np.insert(np.array(e), 0, grid, axis=1)
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))
        np.savetxt('pes_1d' + ('_opti' if relax else '') + '.dat', pes)

    def plot_2D(self, R, dof_i, dof_j, starts, ends, steps, relax=False):
        """Generate data to plot a potential energy surface along two
        dimensions. The other degrees of freedom can either kept fix or can be
        optimized. The results are writte to a file called 'pes_2d.dat' or
        'pes_2d_opti.dat'. The file is formatted as follows. The first and
        second column give the points along the scanned coordinates. The third
        column the energy. If relax is True then the full optimized geometry is
        given after the energy. The file is written in a blocked structure to
        be used in gnuplot, i.e., after each full scan of the second
        coordinate, an additional newline is given.

        TODO: Implement the use of a certain excited state.
        TODO: add variable file name

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            Starting position for plotting. It either defines the fixed
            positions or the starting point for optimizations.
        dof_i : integer
            Index of the first variable that should change.
        dof_j : integer
            Index of the second variable that should change.
        starts : 2 floats
            Starting values of the coordinates that are scanned in au.
        ends : 2 floats
            Ending values of the coordinates that are scanned in au.
        steps : 2 floats
            Step sizes used along the scan in au.
        relax : bool, optional, Default: False
            Whether all other coordinates are fixed (False) or
            relaxed (True) along the scan.
        S : integer, optional, Default: 0
            Could be the state, not impemented yet.

        Returns
        -------
        Nothing, but writes a data file 'pes_2d.dat' or 'pes_2d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        # TODO: add some asserts
        e = []
        if relax:
            R_optimized = []

        # TODO: The ordering of the grids and the compability with gnuplot
        # needs to be confirmed.
        x = np.arange(starts[0], ends[0]+steps[0], steps[0])
        y = np.arange(starts[1], ends[1]+steps[1], steps[1])
        n_grid = len(x) * len(y)
        grid_x, grid_y = np.meshgrid(x, y)

        for g_x in grid_x:
            for g_y in grid_y:

                if relax:
                    constraint = ({'type': 'eq', 'fun': lambda x: x[dof_i] - g_x},
                                  {'type': 'eq', 'fun': lambda x: x[dof_j] - g_y})
                    R0 = R.copy()
                    R0[dof_i] = g_x
                    R0[dof_j] = g_y
                    results = spminimize(self._energy_wrapper, R0,
                                       method='SLSQP', constraints=constraint)

                    if results.success:
                        R_optimized.append(results.x.copy())
                        e.append([results.fun])
                    else:
                        raise RuntimeError("Minimization of PES failed with"
                                           " message: " + results.message)
            else:
                R[dof_i] = g_x
                R[dof_j] = g_y
                e.append([self._energy_wrapper(R)])

        grid = np.dstack((grid_x, grid_y)).reshape(n_grid, -1)
        pes = np.hstack((grid, e))
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))

        outfile = open('pes_2d' + ('_opti' if relax else '') + '.dat', 'w')
        for k, data in enumerate(pes):
            outfile.write(' '.join([str(x) for x in data]))
            outfile.write('\n')
            if (k+1) % len(x) == 0:
                outfile.write('\n')

    def get_Hessian(self, R):
        """Calculate the Hessian at a given geometry R. The Hessian is
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
        original_R = R.copy()

        for i in range(len(R)):
            # TODO: maybe put into some numerics module?
            R[i] += self.DERIVATIVE_STEPSIZE
            grad_plus = self._gradient_wrapper(R)

            R = original_R.copy()
            R[i] -= self.DERIVATIVE_STEPSIZE
            grad_minus = self._gradient_wrapper(R)

            H[i] = (grad_plus - grad_minus) / (2.0 * self.DERIVATIVE_STEPSIZE)

            R = original_R.copy()

        return H
