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


class Interface:
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
    """

    def __init__(self, name, **kwargs):
        self.__name = name

        self._old_R = None
        self._old_P = None
        self._old_S = None
        self._energy = None
        self._gradient = None

        self.__SAVE_THRESHOLD = 1e-8

    @property
    def name(self):
        """str : The name of the specific interface implementation."""
        return self.__name

    @property
    def SAVE_THESHOLD(self):
        """float : Theshold for decision of using saved variables vs.
        recomputing properties."""
        return self.__SAVE_THRESHOLD

    @property
    def DERIVATIVE_STEPSIZE(self):
        """float : Step size for numerical derivatives in au."""
        return 1e-4

    def _calculate_all(self, R, P, S=None):
        """Calculate the energy, gradient and possibly couplings at the current
        geometry.

        Parameters
        ----------
        R : two-dimensional ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        P : two-dimensional ndarray of floats
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

    def _changed(self, R, P, S=None):
        """Check if we need to recalculate the potential, gradients or
        couplings due to changed positions, etc.
        Currently, this only tests a change in position and this needs to be
        improved.

        Parameters
        ----------
        R : two-dimensional ndarray of floats
            The (ring-polymer) positions representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        P : two-dimensional ndarray of floats
            The (ring-polymer) momenta representing the system in au. The
            first index represents the degrees of freedom, the second one the
            beads.
        S : integer, default None
            The current state of the system.

        Returns
        -------
        True if the quantities need to be recalculated, False otherwise.
        """

        assert (isinstance(R, np.ndarray)), "R not a numpy array!"
        assert (R.ndim == 2), "Position array not two-dimensional!"
        assert (R.dtype == 'float64'), "Position array not real!"
        if P is not None:
            assert (isinstance(P, np.ndarray)), "P not a numpy array!"
            assert (P.ndim == 2), "Momentum array not two-dimensional!"
            assert (P.dtype == 'float64'), "Momentum array not real!"

        if self._old_R is None:
            self._old_R = R.copy()
            return True
        if (abs(R - self._old_R) < self.SAVE_THESHOLD).all():
            return False
        else:
            self._old_R = R.copy()
            return True

    def energy(self, R, S=None):
        """Obtain energy of the system in the current state.

        Parameters
        ----------
        R : array of arrays of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default None
            The current state of the system.

        Returns
        -------
        array of float
        The energy of the system at each bead position in hartree.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)

        if S is None:
            return self._energy[0]
        else:
            return self._energy[S]

    def gradient(self, R, S=None):
        """Obtain gradient of the system in the current state.

        Parameters
        ----------
        R : array of arrays of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default None
            The current state of the system.

        Returns
        -------
        array of arrays of floats
        The gradient of the system at each bead position in hartree/au.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)

        if S is None:
            return self._gradient[0]
        else:
            return self._gradient[S]

    def coupling(self, R,):
        """ Obtain coupling. """
        raise NotImplementedError
#        if self._changed(R, None, None):
#            self._calculate
#
#        return self._coupling

    def _energy_wrapper(self, R, S=0):
        """Wrapper function to do call energy with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : array of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.

        Returns
        -------
        The energy at the given geometry in hartree.
        """
        return self.energy(np.array([R]), S)

    def _gradient_wrapper(self, R, S=0):
        """Wrapper function to do call gradient with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : array of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.

        Returns
        -------
        The gradient at the given geometry in hartree/au.
        """
        return self.gradient(np.array([R]), S)

    def minimize_geom(self, R0):
        """Find the potential minimum employing the Newton-CG method
        as implemented in Scipy.

        Parameters
        ----------
        R0 : array of floats
             The starting position for the minimization.

        Returns
        ----------
        fun : float
              The minimal potential value in hartree.
        x : array
           The minimized position in au.

        Raises a RuntimeError if unsuccessful.
        """

        self.__SAVE_THRESHOLD = 1e-15
        results = spminimize(self._energy_wrapper, R0, method='Newton-CG',
                           jac=self._gradient_wrapper)

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
        R : array of floats
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
        R : array of floats
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
        R : array of floats
            Position for which the Hessian is calculated
            .
        Returns
        -------
        H : array of array of floats
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
