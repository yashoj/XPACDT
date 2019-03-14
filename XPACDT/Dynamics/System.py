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
#  CDTK is free software: you can redistribute it and/or modify
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

import sys

import XPACDT.Dynamics.VelocityVerlet as vv
import XPACDT.Dynamics.Nuclei as nuclei


class System(object):
    """This class is the main representation of the system treated with XPACDT.
    It sets up the calculation, stores all important objects and runs whatever
    is needed.

    Parameters
    ----------


### TODO: stuff we need:
    - number of dof
    - masses/atoms

    - Interface (own classes)

    - Nuclear part (own class)
      -> # beads
      -> positions
      -> velocities
      -> propagator

    - Electronic part (own class)
      -> expansion coefficients
    - -> electron dynamics part

    - restrictions
    - pes where


    """

    def __init__(self, parameters, **kwargs):

        # Set up potential
        pes_name = parameters.get_section("system").get("Interface", None)
        self.__pes = getattr(sys.modules["XPACDT.Interfaces." + pes_name],
                             pes_name)(**parameters.get_section(pes_name))

        self.n_dof = parameters.get_section("system").get("dof")

        # TODO: init nuclei
        self._init_nuclei(parameters)
        # TOOD: init electrons
        self._init_electrons(parameters)

        pass

    @property
    def n_dof(self):
        """int : The number of degrees of freedom in this system."""
        return self.__n_dof

    @n_dof.setter
    def n_dof(self, i):
        assert (int(i) > 0), "Number of degrees of freedom less than 1!"
        self.__n_dof = int(i)

    def _init_nuclei(self, parameters):
        """ Function to set up the nuclei of a system including all associated
        objects like propagators.

        Parameters
        ----------
        parameters : XPACDT input file
            Dictonary-like presentation of the input file.
        """

        # coordinates, masses from input - reshape and test some consistency
        masses = parameters._masses
        if parameters._c_type == 'mass-value':
            coordinates = parameters._coordinates.reshape((self.n_dof, -1))
        elif parameters._c_type == 'xyz':
            coordinates = parameters._coordinates.T.reshape((self.n_dof, -1))

        try:
            momenta = parameters._momenta.reshape(coordinates.shape)
        except ValueError:
            sys.stderr.write("Number of given momenta and coordinates "
                             "does not match! \n")
            sys.exit(-1)

        # set up propagator
        p = parameters.get_section("propagation")
        dt = float(p.get("dt").split()[0])  # TODO: unit conversion here
        kwargs = {}
        if parameters.get_section("rpmd"):
            kwargs["beta"] = float(parameters.get_section("rpmd").get("beta"))
        propagator = vv.VelocityVerlet(dt, self.__pes, masses, **kwargs)

        self.__nuclei = nuclei.Nuclei(self.n_dof, coordinates, momenta,
                                      propagator=propagator,
                                      n_beads=[coordinates.shape[1]])
        print(self.__nuclei.n_beads)

    def _init_electrons(self, parameters):
        """ Function to set up the electrons of a system including all
        associated objects like propagators. Not yet implemented.

        Parameters
        ----------
        parameters : XPACDT input file
            Dictonary-like presentation of the input file.
        """

        self.__electrons = None

    def sample(self):
        """ Sample the system."""
        pass

    def propagate(self):
        """ Propagate the system."""

        # only a basic test right now
        outfile = open("/tmp/blah.dat", 'w')
        for i in range(101):
            outfile.write(str(i*0.1) + " ")
            outfile.write(str(self.__nuclei.x_centroid[0]) + " ")
            outfile.write(str(self.__nuclei.p_centroid[0]) + " ")
            outfile.write("\n")
            self.__nuclei.propagate(0.1)
        outfile.close()
