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
    """

    def __init__(self, name, **kwargs):
        self.__name = name

        self._old_R = None
        self._old_P = None
        self._old_S = None

        self._energy = None
        self._gradient = None
        self._energy_centroid = None
        self._gradient_gradient = None

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

    def energy(self, R, S=None, centroid=False):
        """Obtain energy of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default None
            The current state of the system.
        centroid : bool, default False
            If the energy of the centroid should be returned.


        Returns
        -------
        (n_beads) ndarray of float /or/ float
        The energy of the system at each bead position or at the centroid
        in hartree.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)

        if centroid:
            return self._energy_centroid[0 if S is None else S]
        else:
            return self._energy[0 if S is None else S]

    def gradient(self, R, S=None, centroid=False):
        """Obtain gradient of the system in the current state.

        Parameters
        ----------
        R : (n_dof, n_beads) ndarray of floats
            The (ring-polymer) positions representing the system in au.
        S : integer, default None
            The current state of the system.
        centroid : bool, default False
            If the gradient of the centroid should be returned.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats /or/ (n_dof) ndarray of floats
        The gradient of the system at each bead position or at the centroid
        in hartree/au.
        """
        if self._changed(R, None, S):
            self._calculate_all(R, None, S)

        if centroid:
            return self._gradient_centroid[0 if S is None else S]
        else:
            return self._gradient[0 if S is None else S]

    def coupling(self, R,):
        """ Obtain coupling. """
        raise NotImplementedError
#        if self._changed(R, None, None):
#            self._calculate
#
#        return self._coupling

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
        This function throws and NotImplemented Error and needs to be
        implemented by any child class.
        """
        raise NotImplementedError

    def _energy_wrapper(self, R, S=0, internal=False):
        """Wrapper function to do call energy with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.
        internal : bool, optional, default False
            Whether 'R' is in internal coordinates. Then it is backtransformed
            to cartesian coordinates in the wrapper.

        Returns
        -------
        float:
        The energy at the given geometry in hartree.
        """

        if internal:
            return self.energy(self._from_internal(R)[:, None], S)
        else:
            return self.energy(R[:, None], S)

    def _gradient_wrapper(self, R, S=0, internal=False):
        """Wrapper function to do call gradient with a one-dimensional array.
        This should only be used for directly accessing the PES and not for any
        dynamics calculation!.

        Parameters
        ----------
        R : (n_dof) ndarray of floats
            The positions representing the system in au.
        S : integer, default 0
            The current state of the system.
        internal : bool, optional, default False
            Whether 'R' is in internal coordinates. Then it is backtransformed
            to cartesian coordinates in the wrapper.

        Returns
        -------
        (n_dof) ndarray of floats:
        The gradient at the given geometry in hartree/au.
        """

        if internal:
            return self.gradient(self._from_internal(R)[:, None], S)
        else:
            return self.gradient(R[:, None], S)

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

    def plot_1D(self, R, dof_i, start, end, step, relax=False, internal=False, S=0):
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
        internal : bool, optional, Default: False
            Whether R is in internal coordinates and internal coordinates
            should be used throughout the plotting.
        S : integer, optional, Default: 0
            Could be the state, not impemented yet.

        Returns
        -------
        Nothing, but writes a data file 'pes_1d.dat' or 'pes_1d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        # TODO: output respective GNUPLOT script - move to some separate area for header
        gnuplotfile = open('pes_1d' + ('_opti' if relax else '') + '.plt', "w")
        gnuplotfile.write("# Automatically generated by XPACDT for analysis plotting with gnuplot...\n")
        gnuplotfile.write("")
        gnuplotfile.write("set term postscript eps enhanced color font \",24\"\n\n")
        gnuplotfile.write("# Green #00C400\n")
        gnuplotfile.write("# Blue #4D99FF\n")
        gnuplotfile.write("# Orange #FF8000\n\n")
        gnuplotfile.write("# solid lines\n")
        gnuplotfile.write("set style line 1 lt 1 dt 1 lc rgb 'black' lw 2\n")
        gnuplotfile.write("set style line 2 lt 1 dt 1 lc rgb '#00C400' lw 2\n")
        gnuplotfile.write("set style line 3 lt 1 dt 1 lc rgb '#4D99FF' lw 2\n")
        gnuplotfile.write("set style line 4 lt 1 dt 1 lc rgb '#FF8000' lw 2\n\n")
        gnuplotfile.write("# long broken: (10,5,10,5) -> bad\n")
        gnuplotfile.write("# short broken: (5,10,5,10) -> ok\n")
        gnuplotfile.write("# dots: (2,10,2,10) -> ok\n")
        gnuplotfile.write("# dash dot: (5,10,2,10) -> bad\n")
        gnuplotfile.write("# dash dot dot: (5,10,2,10,2,10) -> bad\n\n")
        gnuplotfile.write("# short broken\n")
        gnuplotfile.write("set style line 11 pt 1 lt 1 dt (5,10,5,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 12 pt 1 lt 1 dt (5,10,5,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 13 pt 1 lt 1 dt (5,10,5,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 14 pt 1 lt 1 dt (5,10,5,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dots\n")
        gnuplotfile.write("set style line 21 lt 1 dt (2,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 22 lt 1 dt (2,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 23 lt 1 dt (2,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 24 lt 1 dt (2,10,2,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# long broken\n")
        gnuplotfile.write("set style line 31 lt 1 dt (10,5,10,5) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 32 lt 1 dt (10,5,10,5) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 33 lt 1 dt (10,5,10,5) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 34 lt 1 dt (10,5,10,5) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dash dot\n")
        gnuplotfile.write("set style line 41 lt 1 dt (5,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 42 lt 1 dt (5,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 43 lt 1 dt (5,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 44 lt 1 dt (5,10,2,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dash dot dot\n")
        gnuplotfile.write("set style line 51 lt 1 dt (5,10,2,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 52 lt 1 dt (5,10,2,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 53 lt 1 dt (5,10,2,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 54 lt 1 dt (5,10,2,10,2,10) lc rgb '#FF8000' lw 3\n\n")

        gnuplotfile.write("set xlabel 'coordinate " + str(dof_i) + "' \n")
        gnuplotfile.write("set ylabel 'energy / au'\n")
        gnuplotfile.write("set output 'pes_1d" + ('_opti' if relax else '') + ".eps'\n")
        gnuplotfile.write("plot 'pes_1d" + ('_opti' if relax else '') + ".dat' using 1:2 w l ls 1 title 'TODO' \n")
        gnuplotfile.close()

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
                results = spminimize(lambda x : self._energy_wrapper(x, internal=internal),
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
                e.append([self._energy_wrapper(R, internal=internal)])

        pes = np.insert(np.array(e), 0, grid, axis=1)
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))
        np.savetxt('pes_1d' + ('_opti' if relax else '') + '.dat', pes)

    def plot_2D(self, R, dof_i, dof_j, starts, ends, steps, relax=False, internal=False):
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
        starts : tuple of 2 floats
            Starting values of the coordinates that are scanned in au.
        ends : tuple of 2 floats
            Ending values of the coordinates that are scanned in au.
        steps : tuple of 2 floats
            Step sizes used along the scan in au.
        relax : bool, optional, Default: False
            Whether all other coordinates are fixed (False) or
            relaxed (True) along the scan.
        internal : bool, optional, Default: False
            Whether R is in internal coordinates and internal coordinates
            should be used throughout the plotting.
        S : integer, optional, Default: 0
            Could be the state, not impemented yet.

        Returns
        -------
        Nothing, but writes a data file 'pes_2d.dat' or 'pes_2d_opti.dat'.
        Throws RuntimeError if relax is True and minimization failed.
        """

        # TODO: output respective GNUPLOT script - move to some separate area for header
        gnuplotfile = open('pes_2d' + ('_opti' if relax else '') + '.plt', "w")
        gnuplotfile.write("# Automatically generated by XPACDT for analysis plotting with gnuplot...\n")
        gnuplotfile.write("")
        gnuplotfile.write("set term postscript eps enhanced color font \",24\"\n\n")
        gnuplotfile.write("# Green #00C400\n")
        gnuplotfile.write("# Blue #4D99FF\n")
        gnuplotfile.write("# Orange #FF8000\n\n")
        gnuplotfile.write("# solid lines\n")
        gnuplotfile.write("set style line 1 lt 1 dt 1 lc rgb 'black' lw 2\n")
        gnuplotfile.write("set style line 2 lt 1 dt 1 lc rgb '#00C400' lw 2\n")
        gnuplotfile.write("set style line 3 lt 1 dt 1 lc rgb '#4D99FF' lw 2\n")
        gnuplotfile.write("set style line 4 lt 1 dt 1 lc rgb '#FF8000' lw 2\n\n")
        gnuplotfile.write("# long broken: (10,5,10,5) -> bad\n")
        gnuplotfile.write("# short broken: (5,10,5,10) -> ok\n")
        gnuplotfile.write("# dots: (2,10,2,10) -> ok\n")
        gnuplotfile.write("# dash dot: (5,10,2,10) -> bad\n")
        gnuplotfile.write("# dash dot dot: (5,10,2,10,2,10) -> bad\n\n")
        gnuplotfile.write("# short broken\n")
        gnuplotfile.write("set style line 11 pt 1 lt 1 dt (5,10,5,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 12 pt 1 lt 1 dt (5,10,5,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 13 pt 1 lt 1 dt (5,10,5,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 14 pt 1 lt 1 dt (5,10,5,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dots\n")
        gnuplotfile.write("set style line 21 lt 1 dt (2,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 22 lt 1 dt (2,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 23 lt 1 dt (2,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 24 lt 1 dt (2,10,2,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# long broken\n")
        gnuplotfile.write("set style line 31 lt 1 dt (10,5,10,5) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 32 lt 1 dt (10,5,10,5) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 33 lt 1 dt (10,5,10,5) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 34 lt 1 dt (10,5,10,5) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dash dot\n")
        gnuplotfile.write("set style line 41 lt 1 dt (5,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 42 lt 1 dt (5,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 43 lt 1 dt (5,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 44 lt 1 dt (5,10,2,10) lc rgb '#FF8000' lw 3\n\n")
        gnuplotfile.write("# dash dot dot\n")
        gnuplotfile.write("set style line 51 lt 1 dt (5,10,2,10,2,10) lc rgb 'black' lw 3\n")
        gnuplotfile.write("set style line 52 lt 1 dt (5,10,2,10,2,10) lc rgb '#00C400' lw 3\n")
        gnuplotfile.write("set style line 53 lt 1 dt (5,10,2,10,2,10) lc rgb '#4D99FF' lw 3\n")
        gnuplotfile.write("set style line 54 lt 1 dt (5,10,2,10,2,10) lc rgb '#FF8000' lw 3\n\n")

        gnuplotfile.write("set xlabel 'coordinate " + str(dof_j) + "' \n")
        gnuplotfile.write("set ylabel 'coordinate " + str(dof_i) + "' \n")
        gnuplotfile.write("set output 'pes_2d" + ('_opti' if relax else '') + ".eps'\n")
        gnuplotfile.write("unset ztics \n")
        gnuplotfile.write("unset key \n")
        gnuplotfile.write("unset title \n")
        gnuplotfile.write("set contour base \n")
        gnuplotfile.write("set view map \n")
        gnuplotfile.write("unset surface \n")
        gnuplotfile.write("set cntrparam levels incr 0.0,0.01,0.5\n")
        gnuplotfile.write("splot 'pes_2d" + ('_opti' if relax else '') + ".dat' using 1:2:3 w l lw 3.2\n")
        gnuplotfile.close()

        old_thresh = self.__SAVE_THRESHOLD
        self.__SAVE_THRESHOLD = 1e-15

        e = []
        if relax:
            R_optimized = []

        # TODO: The ordering of the grids and the compability with gnuplot
        # needs to be confirmed.
        x = np.arange(starts[0], ends[0]+steps[0], steps[0])
        y = np.arange(starts[1], ends[1]+steps[1], steps[1])
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
                    results = spminimize(lambda x : self._energy_wrapper(x, internal=internal),
                                         R0, method='SLSQP',
                                         constraints=constraint,
                                         options={'ftol': 1e-6, 'eps':1e-3})

                    if results.success:
                        R_optimized.append(results.x.copy())
                        e.append([results.fun])
                    else:
                        raise RuntimeError("Minimization of PES failed with"
                                           " message: " + results.message)
                else:
                    R[dof_i] = g_x
                    R[dof_j] = g_y
                    e.append([self._energy_wrapper(R, internal=internal)])

        grid = np.dstack((grid_y, grid_x)).reshape(n_grid, -1)
        pes = np.hstack((grid, e))
        if relax:
            pes = np.hstack((pes, np.array(R_optimized)))

        outfile = open('pes_2d' + ('_opti' if relax else '') + '.dat', 'w')
        for k, data in enumerate(pes):
            outfile.write(' '.join([str(x) for x in data]))
            outfile.write('\n')
            if (k+1) % len(x) == 0:
                outfile.write('\n')
        outfile.close()

        self.__SAVE_THRESHOLD = old_thresh

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
        R_step = R.copy()

        for i in range(len(R)):
            # TODO: maybe put into some numerics module?
            R_step[i] += self.DERIVATIVE_STEPSIZE
            grad_plus = self._gradient_wrapper(R_step)

            R_step[i] -= 2.0 * self.DERIVATIVE_STEPSIZE
            grad_minus = self._gradient_wrapper(R_step)

            H[i] = (grad_plus - grad_minus) / (2.0 * self.DERIVATIVE_STEPSIZE)

            R_step[i] += self.DERIVATIVE_STEPSIZE

        return H
