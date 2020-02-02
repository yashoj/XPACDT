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

"""Implementation of the velocity verlet propagator."""

import math
import numpy as np
import sys

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo
import XPACDT.System.Electrons as elecInterface
import XPACDT.Tools.Units as units


class VelocityVerlet(object):
    """
    This is an implementation of the Velocity Verlet propagation scheme. It
    can be interfaced with different thermostats and different constraints.

    Parameters
    ----------
    electrons : XPACDT.System.Electrons
        Representation of the electrons that gives the gradients.
    mass : (n_dof) ndarray of floats
        Masses of the system in au.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.
    beta : float, optional, default np.nan
        Inverse temperature for ring polymer springs in a.u.

    Other Parameters
    ----------------
    timestep : float
        Basic timestep for the propagator in a.u.
    rp_transform_type : {'matrix', 'fft'}, optional, default: 'matrix'
        Type of ring polymer normal mode transformation to be used.

    Attributes:
    -----------
    timestep
    electrons
    propagation_matrix
    thermostat
    """

    def __init__(self, electrons, mass, n_beads, beta=np.nan, **kwargs):
        assert (isinstance(electrons, elecInterface.Electrons)), \
            "electrons not derived from System.Electrons!"

        if 'timestep' not in kwargs:
            raise KeyError("\nXPACDT: No timestep given for velocity verlet"
                           "propagator.")

        self.__electrons = electrons
        self.__timestep = units.parse_time(kwargs.get("timestep"))

        self.__thermostat = None
        self.__constraints = None

        rp_transform_type = kwargs.get('rp_transform_type', 'matrix')
        # basic initialization
        self._set_propagation_matrix(beta, n_beads, mass)
        self.RPtransform = RPtrafo.RingPolymerTransformations(n_beads,
                                                              rp_transform_type)
        return

    @property
    def timestep(self):
        """ float : The timestep of the propagator in a.u."""
        return self.__timestep

    @property
    def electrons(self):
        """ XPACDT.System.Electrons : The electrons used in the propagation."""
        return self.__electrons

    @property
    def propagation_matrix(self):
        """ (n_dof, n_beads, 2, 2) ndarray of floats. For each degree of
        freedom (first dimension) and each internal ring polymer degree of
        freedom (second dimension) a matrix for propagating the ring polymer
        with a timestep and beta.
        """
        return self.__propagation_matrix

    @property
    def thermostat(self):
        """ The thermostat used in the propagation."""
        return self.__thermostat

    def attach_thermostat(self, input_parameters, masses):
        """ Create a thermostat and attach it to the propagator.

        Parameters
        ----------
        input_parameters : XPACDT.Inputfile
            Represents all the input parameters for the simulation given in the
            input file.
        masses : (n_dof) ndarray of floats
            The mass of each degree of freedom in au.
        """
        thermo_parameters = input_parameters.get('thermostat')
        if 'method' not in thermo_parameters:
            raise KeyError("\nXPACDT:No thermostat method given.")

        method = thermo_parameters.get('method')
        __import__("XPACDT.Dynamics." + method + "Thermostat")
        self.__thermostat = getattr(sys.modules["XPACDT.Dynamics." + method + "Thermostat"],
                                    method)(input_parameters, masses)

    def propagate(self, R, P, time_propagation):
        """ Advance the given position and momenta for a given time.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The positions `R` and momenta `P` of all beads. The first axis is
            the degrees of freedom and the second axis the beads.
        time_propagation : float
            The amount of time to advance in au.

        Returns
        -------
        Rn, Pn : (n_dof, n_beads) ndarray of floats
             The positions `Rn` and momenta `Pn` of all beads after advancing
             in time. The first axis is the degrees of freedom and the second
             axis is beads.
        """

        Rt, Pt = R.copy(), P.copy()
        n_steps = int((time_propagation + 1e-8) // self.timestep)

        assert(math.isclose(n_steps*self.timestep, time_propagation, abs_tol=1e-6)), \
            "\nXPACDT: Propagation time is not multiple of velocity verlet timestep."

        for j in range(n_steps):
            Rn, Pn = self._step(Rt, Pt)
            Rt, Pt = Rn.copy(), Pn.copy()

        if self.thermostat is not None:
            self.thermostat.apply(Rn, Pn, 0)
        return Rn, Pn

    def _step(self, R, P):
        """ One velocity_verlet step with the internal timestep and for the
        given positions and momenta.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The positions `R` and momenta `P` of all beads. The first axis is
            the degrees of freedom and the second axis the beads.

        Returns
        -------
        rt, pt : (n_dof, n_beads) ndarray of floats
             The positions `rt` and momenta `pt` of all beads after advancing
             in time. The first axis is the degrees of freedom and the second
             axis is beads.
        """
        pt2 = self._velocity_step(P, R)
        if self.thermostat is not None:
            self.thermostat.apply(R, P, 1)

        # TODO: constraints 1

        rt, pt = self._verlet_step(R, pt2)
        # TODO: constraints 2
        if self.thermostat is not None:
            self.thermostat.apply(rt, pt, 2)

        pt = self._velocity_step(pt, rt)
        if self.thermostat is not None:
            self.thermostat.apply(rt, pt, 3)
        # TODO constraints 3

        return rt, pt

    def _velocity_step(self, P, R):
        """ Take a half-timestep for the momenta with the gradient at the given
        position.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The positions `R` and momenta `P` of all beads. The first axis is
            the degrees of freedom and the second axis the beads.

        Returns
        -------
        (n_dof, n_beads) ndarray of floats
        The momenta of all beads after advancing in time. The first axis
        is the degrees of freedom and the second axis the beads.
        """

        return P - 0.5 * self.timestep * self.electrons.gradient(R)

    def _verlet_step(self, R, P):
        """ Take a full timestep for the positions and internal ring-polymer
        degrees of freedom.

        Parameters
        ----------
        R, P : (n_dof, n_beads) ndarray of floats
            The positions `R` and momenta `P` of all beads. The first axis is
            the degrees of freedom and the second axis the beads.

        Returns
        -------
        rt, pt : (n_dof, n_beads) ndarray of floats
             The positions and momenta of all beads after advancing in time.
             The first axis is the degrees of freedom and the second axis the beads.
        """

        rnm = self.RPtransform.to_RingPolymer_normalModes(R)
        pnm = self.RPtransform.to_RingPolymer_normalModes(P)

        # four-dimensional array; For each physical degree of freedom and
        # each ring polymer bead the normal mode position and momentum is
        # stored. A fourth dimension is added for broadcasting with the
        # matrix multiplication below
        nms = np.dstack((pnm, rnm))[:, :, :, None]
        rnm_t = np.zeros(rnm.shape)
        pnm_t = np.zeros(rnm.shape)

        # iteration over all physical degrees of freedom
        for k, nm in enumerate(nms):

            # broadcast multiply matrices sitting in the last two dimensions;
            # here: broadcast over all beads, multiply the 2x2 propagation
            # matrix with the (p_kj, q_kj) vector
            tt = np.matmul(self.propagation_matrix[k], nm)[:, :, 0]

            pnm_t[k] = tt[:, 0]
            rnm_t[k] = tt[:, 1]

        return self.RPtransform.from_RingPolymer_normalModes(rnm_t),\
            self.RPtransform.from_RingPolymer_normalModes(pnm_t)

    def _set_propagation_matrix(self, beta, n_beads, mass):
        """Set the propagation matrices for the momenta and internal ring
        polymer coordinates. It is an array of arrays of two-dimensional
        arrays. The first dimension is the physical degrees of freedom, the
        second one the ring polymer beads.

        For a given degree of freedom j (with mass m_j) and a given ring
        polymer normal mode k (with frequency w_k) the propagation matrix
        is a 2x2 matrix and reads:

        .. math::

        \\begin{pmatrix}
            \\cos(w_k * dt) & -m_j * w_k * sin(w_k * dt) \\
            1/(m_j * w_k) * sin(w_k * dt) &  \\cos(w_k * dt)
            \\end{pmatrix}

        See also: https://aip.scitation.org/doi/10.1063/1.3489925

        Parameters
        ----------
        beta : float
            The inserve temperature of the ring polymer in au
            /or/ np.nan if only 1 bead is used.
        n_beads : (n_dof) list of int
            The number of beads for each degree of freedom.
        mass : (n_dof) ndarray of floats
            Masses of the system in au.

        Returns
        -------
        Nothing, but `self.__propagation matrix` is initialized.
        -------
        """

        n = max(n_beads)
        w = np.array([2.0 * (float(n) / beta) * math.sin(k * math.pi / float(n))
                      for k in range(1, n)])
        ps = []
        for m in mass:
            pm = [np.array([[1.0, 0.0], [self.timestep / m, 1.0]])]

            for wk in w:
                pm.append(np.array(
                        [[math.cos(wk*self.timestep),
                          -m * wk * math.sin(wk*self.timestep)],
                         [1.0/(wk*m) * math.sin(wk*self.timestep),
                          math.cos(wk * self.timestep)]]))
            ps.append(pm)

        self.__propagation_matrix = np.array(ps)
