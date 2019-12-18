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

import math
import numpy as np
import sys

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo
import XPACDT.System.Electrons as elecInterface
import XPACDT.Tools.Units as units

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
    electrons : XPACDT.System.Electrons
        Representation of the electrons that gives the gradients.
    mass : (n_dof) ndarray of floats
        Masses of the system in au.
    n_beads : (n_dof) list of int
        The number of beads for each degree of freedom.

    Other Parameters
    ----------------
    beta : float, optional, default None
        Inverse temperature for ring polymer springs in a.u.
    rp_transform_type : {'matrix', 'fft'}, optional, default: 'matrix'
        Type of ring polymer normal mode transformation to be used.

    Attributes:
    -----------
    electrons
    mass
    n_beads : (n_dof) list of int
        Number of beads for each degrees of freedom
    """

    def __init__(self, electrons, mass, n_beads, **kwargs):
        # TODO: basic argument parsing here

        assert (isinstance(electrons, elecInterface.Electrons)), \
            "electrons not derived from System.Electrons!"
        assert ('timestep' in kwargs), "No timestep given for propagator."

        self.electrons = electrons
        self.mass = mass
        self.n_beads = n_beads

        self.timestep = units.parse_time(kwargs.get("timestep"))

        # optional as keywords
        if 'beta' in kwargs:
            self.beta = float(kwargs.get('beta'))
        else:
            # In the case when RPMD is not used (i.e. n_beads=1), 'beta' should
            # not be used anywhere, so setting it to NaN.
            self.__beta = np.nan
        rp_transform_type = kwargs.get('rp_transform_type', 'matrix')

        self.__thermostat = None
        self.__constraints = None

        # basic initialization
        self._set_propagation_matrix()
        self.RPtransform = RPtrafo.RingPolymerTransformations(self.n_beads,
                                                              rp_transform_type)
        return

    @property
    def beta(self):
        """ float or np.nan: Inverse temperature for ring polymer springs in
        a.u. It is NaN if not given in the case of 1 bead for each degree of
        freedom."""
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
        """ (n_dof) ndarray of floats : The masses of the system in a.u."""
        return self.__mass

    @mass.setter
    def mass(self, a):
        assert ((a > 0).all()), "A mass 0 or less."
        self.__mass = a.copy()

    @property
    def electrons(self):
        """ XPACDT.System.Electrons : The electrons used in the propagation."""
        return self.__electrons

    @electrons.setter
    def electrons(self, p):
        assert (isinstance(p, elecInterface.Electrons)), "electrons not"
        "derived from System.Electrons!"
        self.__electrons = p

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
        masses : (n_dof) ndarray of floats
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
        # TODO: possibly step size control here.
        # TODO: possibly multiple-timestepping here

        Rt, Pt = R.copy(), P.copy()
        # TODO: handle time not a multiple of timestep similar to 'propagate'
        # in nuclei; this is only needed if this module has different timestep
        # than nuclei so for multiple steps or for adaptive time step control.
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
        rt : (n_dof, n_beads) ndarray of floats
             The positions of all beads after advancing in time. The first
             axis is the degrees of freedom and the second axis the beads.
        pt : (n_dof, n_beads) ndarray of floats
             The momenta of all beads after advancing in time. The first axis
             is the degrees of freedom and the second axis the beads.
        """

        # TODO: profile and optimize, and generalize to more D; docu properly
        rnm = self.RPtransform.to_RingPolymer_normalModes(R)
        pnm = self.RPtransform.to_RingPolymer_normalModes(P)

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

        return self.RPtransform.from_RingPolymer_normalModes(rnm_t),\
            self.RPtransform.from_RingPolymer_normalModes(pnm_t)

    def _set_propagation_matrix(self):
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

        Returns
        -------
        Nothing, but `self.__propagation matrix` is initialized.
        -------
        """
        # TODO: Make this compatible with different n_beads for each dof
        n = max(self.n_beads)
        w = np.array([2.0 * (float(n) / self.beta) * math.sin(k * math.pi / float(n))
                      for k in range(1, n)])
        ps = []
        for m in self.mass:
            pm = [np.array([[1.0, 0.0], [self.timestep / m, 1.0]])]

            for wk in w:
                pm.append(np.array(
                        [[math.cos(wk*self.timestep),
                          -m * wk * math.sin(wk*self.timestep)],
                         [1.0/(wk*m) * math.sin(wk*self.timestep),
                          math.cos(wk * self.timestep)]]))
            ps.append(pm)

        self.__propagation_matrix = np.array(ps)
