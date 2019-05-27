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

"""Implementation of the velocity verlet propagator."""

from molmod.units import parse_unit
import numpy as np
import sys

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo
import XPACDT.Interfaces.InterfaceTemplate as template

# TODO: test, benchmark, optimize, docu
# TODO: add thermostatting and constraints.


class VelocityVerlet(object):
    """
    This is an implementation of the Velocity Verlet propagation scheme. It
    can be interfaced with different thermostats and different constraints.

    Parameters
    ----------
    dt : float
        Basic timestep for the propagator in a.u.
    potential : Child of XPACDT.Interfaces.InterfaceTemplate
        Potential function that gives the gradients.
    mass : array of floats
        Masses of the system in au.

    Other Parameters
    ----------------
    beta : float, optional, default None
        Inverse temperature for ring polymer springs in a.u.
    """

    def __init__(self, potential, mass, **kwargs):
        # TODO: basic argument parsing here

        assert (isinstance(potential, template.PotentialInterface)), \
            "potential not derived from InterfaceTemplate!"
        assert ('timestep' in kwargs), "No timestep given for propagator."

        self.potential = potential
        self.mass = mass

        dt_string = kwargs.get("timestep").split()
        self.timestep = float(dt_string[0]) * parse_unit(dt_string[1])

        # optional as keywords
        if 'beta' in kwargs:
            self.beta = float(kwargs.get('beta'))
        else:
            self.__beta = -1.0

        self.__thermostat = None
        self.__constraints = None

        # basic initialization
        self.__propagation_matrix = None
        return

    @property
    def beta(self):
        """ float : Inverse temperature for ring polymer springs in a.u."""
        return self.__beta

    @beta.setter
    def beta(self, f):
        assert (f is None or f > 0), "Beta 0 or less."
        self.__beta = f

    @property
    def timestep(self):
        """ float : The timestep of the propagator in a.u."""
        return self.__timestep

    @timestep.setter
    def timestep(self, f):
        assert (f > 0), "Timestep 0 or less."
        self.__timestep = f

    @property
    def mass(self):
        """ ndarray of floats : The masses of the system in a.u."""
        return self.__mass

    @mass.setter
    def mass(self, a):
        assert ((a > 0).all()), "A mass 0 or less."
        self.__mass = a.copy()

    @property
    def potential(self):
        """ XPACDT.Interface : The potential used in the propagation."""
        return self.__potential

    @potential.setter
    def potential(self, p):
        assert (isinstance(p, template.PotentialInterface)), "potential not"
        "derived from InterfaceTemplate!"
        self.__potential = p

    @property
    def propagation_matrix(self):
        """ four-dimensional ndarray of floats. For each degree of
        freedom (first dimension) and each internal ring polymer degree of
        freedom (second dimension) a matrix for propagating the ring polymer
        with a timestep and beta.
        """
        return self.__propagation_matrix

    @property
    def thermostat(self):
        """ The thermostat used in the propagation."""
        return self.__thermostat

    @thermostat.setter
    def thermostat(self, t):
        self.__thermostat = t

    def attach_thermostat(self, input_parameters, masses):
        """ Create a thermostat and attach it to the propagator.

        Parameters
        ----------
        input_parameters : XPACDT.Inputfile
            Represents all the input parameters for the simulation given in the
            input file.
        masses : ndarray of floats
            The mass of each degree of freedom in au.
        """
        thermo_parameters = input_parameters.get('thermostat')
        assert('method' in thermo_parameters), "No thermostat method given."

        method = thermo_parameters.get('method')
        __import__("XPACDT.Dynamics." + method)
        self.thermostat = getattr(sys.modules["XPACDT.Dynamics." + method],
                                  method)(input_parameters, masses)

    def propagate(self, R, P, time_propagation):
        """ Advance the given position and momenta for a given time.

        Parameters
        ----------
        R : two-dimensional ndarray of floats
            The positions of all beads. The first axis is the degrees of
            freedom and the second axis the beads.
        P : two-dimensional ndarray of floats
            The momenta of all beads. The first axis is the degrees of
            freedom and the second axis the beads.
        time_propagation : float
            The amount of time to advance in au.

        Returns
        -------
        Rn : two-dimensional ndarray of floats
             The positions of all beads after advancing in time. The first
             axis is the degrees of freedom and the second axis the beads.
        Pn : two-dimensional ndarray of floats
             The momenta of all beads after advancing in time. The first axis
             is the degrees of freedom and the second axis the beads.
        """
        # TODO: possibly step size control here.
        # TODO: possibly multiple-timestepping here

        Rt, Pt = R.copy(), P.copy()
        # TODO: handle time not a multiple of timestep? What's the best way?
        n_steps = int((time_propagation + 1e-8) // self.timestep)
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
        R : two-dimensional ndarray of floats
            The positions of all beads. The first axis is the degrees of
            freedom and the second axis the beads.
        P : two-dimensional ndarray of floats
            The momenta of all beads. The first axis is the degrees of
            freedom and the second axis the beads.

        Returns
        -------
        rt : two-dimensional ndarray of floats
             The positions of all beads after advancing in time. The first
             axis is the degrees of freedom and the second axis the beads.
        pt : two-dimensional ndarray of floats
             The momenta of all beads after advancing in time. The first axis
             is the degrees of freedom and the second axis the beads.
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
        P : two-dimensional ndarray of floats
            The momenta of all beads. The first axis is the degrees of
            freedom and the second axis the beads.
        R : two-dimensional ndarray of floats
            The positions of all beads. The first axis is the degrees of
            freedom and the second axis the beads.

        Returns
        -------
        two-dimensional ndarray of floats
        The momenta of all beads after advancing in time. The first axis
        is the degrees of freedom and the second axis the beads.
        """

        return P - 0.5 * self.timestep * self.potential.gradient(R)

    def _verlet_step(self, R, P):
        """ Take a full timestep for the positions and internal ring-polymer
        degrees of freedom.

        Parameters
        ----------
        P : two-dimensional ndarray of floats
            The momenta of all beads. The first axis is the degrees of
            freedom and the second axis the beads.
        R : two-dimensional ndarray of floats
            The positions of all beads. The first axis is the degrees of
            freedom and the second axis the beads.

        Returns
        -------
        rt : two-dimensional ndarray of floats
             The positions of all beads after advancing in time. The first
             axis is the degrees of freedom and the second axis the beads.
        pt : two-dimensional ndarray of floats
             The momenta of all beads after advancing in time. The first axis
             is the degrees of freedom and the second axis the beads.
        """

        # TODO: profile and optimize, and generalize to more D; docu properly
        rnm = RPtrafo.to_RingPolymer_normalModes(R)
        pnm = RPtrafo.to_RingPolymer_normalModes(P)

        n_beads = R.shape[1]
        self._set_propagation_matrix(n_beads)

        # three-dimensional array; For each physical degree of freedom and
        # each ring polymer bead the normal mode position and momentum is
        # stored.
        nms = np.array([list(zip(p, r)) for r, p in zip(rnm, pnm)])
        rnm_t = np.zeros(rnm.shape)
        pnm_t = np.zeros(rnm.shape)

        # iteration over all physical degrees of freedom
        for k, nm in enumerate(nms):

            # broadcast multiply matrices sitting in the last two dimensions;
            # here: broadcast over all beads, multiply the 2x2 propagation
            # matrix with the (p_kj, q_kj) vector
            tt = np.matmul(self.propagation_matrix[k],
                           np.expand_dims(nm, axis=2))[:, :, 0]

            pnm_t[k] = tt[:, 0]
            rnm_t[k] = tt[:, 1]

        return RPtrafo.from_RingPolymer_normalModes(rnm_t),\
            RPtrafo.from_RingPolymer_normalModes(pnm_t)

    def _set_propagation_matrix(self, n):
        """Set the propagation matrices for the momenta and internal ring
        polymer coordinates. It is an array of arrays of two-dimensional
        arrays. The first dimension is the physical degrees of freedom, the
        second one the ring polymer beads.

        For a given degree of freedom j (with mass m_j) and a given ring
        polymer normal mode k (with frequency w_k) the propagation matrix
        is a 2x2 matrix and reads:
            (  cos(w_k * dt)                 , -m_j * w_k * sin(w_k * dt)  )
            (  1/(m_j * w_k) * sin(w_k * dt) , cos(w_k * dt)               )

        See also: https://aip.scitation.org/doi/10.1063/1.3489925

        Parameters
        ----------
        n: int
            The number of beads in the ring polymer.

        Returns
        -------
        Nothing, but self.__propagation matrix is initialized.
        -------
        """

        if self.propagation_matrix is not None:
            return

        w = np.array([2.0 * (n / self.beta) * np.sin(k * np.pi / n)
                      for k in range(1, n)])

        ps = []
        for m in self.mass:
            pm = [np.array([[1.0, 0.0], [self.timestep / m, 1.0]])]

            for wk in w:
                pm.append(np.array(
                        [[np.cos(wk*self.timestep),
                          -m * wk * np.sin(wk*self.timestep)],
                         [1.0/(wk*m) * np.sin(wk*self.timestep),
                          np.cos(wk * self.timestep)]]))
            ps.append(pm)

        self.__propagation_matrix = np.array(ps)
