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

    # TODO: Howto docu keyword arguments?
    beta : float, optional, default None
        Inverse temperature for ring polymer springs in a.u.
    """

    def __init__(self, potential, mass, **kwargs):
        # TODO: basic argument parsing here

        assert (isinstance(potential, template.Interface)), \
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
        """ Inverse temperature for ring polymer springs in a.u."""
        return self.__beta

    @beta.setter
    def beta(self, f):
        assert (f is None or f > 0), "Beta 0 or less."
        self.__beta = f

    @property
    def timestep(self):
        """ The timestep of the propagator in a.u."""
        return self.__timestep

    @timestep.setter
    def timestep(self, f):
        assert (f > 0), "Timestep 0 or less."
        self.__timestep = f

    @property
    def mass(self):
        """ The masses of the system in a.u."""
        return self.__mass

    @mass.setter
    def mass(self, a):
        assert ((a > 0).all()), "A mass 0 or less."
        self.__mass = a.copy()

    @property
    def potential(self):
        """ The potential used in the propagation."""
        return self.__potential

    @potential.setter
    def potential(self, p):
        assert (isinstance(p, template.Interface)), "potential not"
        "derived from InterfaceTemplate!"
        self.__potential = p

    @property
    def thermostat(self):
        """ The thermostat used in the propagation."""
        return self.__thermostat

    @thermostat.setter
    def thermostat(self, t):
        self.__thermostat = t

    def attach_thermostat(self, parameters, masses):
        thermo_parameters = parameters.get('thermostat')
        assert('method' in thermo_parameters), "No thermostat method given."

        method = thermo_parameters.get('method')
        __import__("XPACDT.Dynamics." + method)
        self.thermostat = getattr(sys.modules["XPACDT.Dynamics." + method], method)(parameters, masses)

    def propagate(self, R, P, time):
        # TODO: possibly step size control here.
        # TODO: possibly multiple-timestepping here

        Rt, Pt = R.copy(), P.copy()
        # howto handle time not a multiple of timestep?
        n_steps = int((time + 1e-8) // self.timestep)
        for j in range(n_steps):
            Rn, Pn = self._step(Rt, Pt)
            Rt, Pt = Rn.copy(), Pn.copy()

        if self.thermostat is not None:
            self.thermostat.apply(Rn, Pn, 0)
        return Rn, Pn

    def _step(self, R, P):

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
        return P - 0.5 * self.timestep * self.potential.gradient(R)

    def _verlet_step(self, R, P):
        # TODO: profile and optimize, and generalize to more D; docu properly
        rnm = RPtrafo.to_RingPolymer_normalModes(R)
        pnm = RPtrafo.to_RingPolymer_normalModes(P)

        n_beads = R.shape[1]
        nms = np.array([list(zip(p, r)) for r, p in zip(rnm, pnm)])
        rnm_t = np.zeros(rnm.shape)
        pnm_t = np.zeros(rnm.shape)

        for k, nm in enumerate(nms):

            tt = np.matmul(self.propagation_matrix(n_beads)[k],
                           np.expand_dims(nm, axis=2))[:, :, 0]

            pnm_t[k] = tt[:, 0]
            rnm_t[k] = tt[:, 1]

        return RPtrafo.from_RingPolymer_normalModes(rnm_t),\
            RPtrafo.from_RingPolymer_normalModes(pnm_t)

    def propagation_matrix(self, n):
        # TODO: make a property?
        if self.__propagation_matrix is None:
            self.__propagation_matrix = self._get_propagation_matrix(n)
        return self.__propagation_matrix

    def _get_propagation_matrix(self, n):
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

        return np.array(ps)
