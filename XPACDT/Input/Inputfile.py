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

"""This class reads and stores an XPACDT input file. A XPACDT input file
consits of sections defining parameters for different parts of the program.
Each section starts with a '$' sign followed by the name of the section
and ends with '$end'. Within each section each line holds exactly one
keyword. Each keyword can then be set to an appropriate value string given
after a '=' character. Blank lines are ignored. Comments can be given and
start with a '#' character."""

import collections
from errno import ENOENT
from io import StringIO
import numpy as np
import os
import re

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo
import XPACDT.Tools.Units as units


class Inputfile(collections.MutableMapping):
    """Basic representation of all the input parameters given to XPACDT. It
    inherits from the MutableMapping Abstract Base Class defined in the
    collections module. This makes the Inputfile behave like a dictonary.

    Parameters
    ----------
    inputfile: str
        Filename of the input file.

    Attributes:
    -----------
    commands
    masses
    n_dof
    n_beads
    beta
    coordinates
    positionShift
    momenta
    momentumShift
    """

    def __init__(self, inputfile):

        self.store = dict()
        self.__momenta = None
        self.__masses = None
        self.__coordinates = None

        self.__positionShift = None
        self.__momentumShift = None

        self._filename = inputfile
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(ENOENT, "Input file does not exist!",
                                    self._filename)

        with open(self._filename, 'r') as infile:
            self._intext = infile.read()

        self._parse_file()

        if 'system' in self:

            if 'dof' in self.get('system'):
                try:
                    self.__n_dof = int(self.get('system').get('dof'))

                except ValueError as e:
                    raise type(e)(str(e) + "\nXPACDT: Number of degrees of"
                                           "freedom cannot be converted to int.")
                if self.__n_dof < 1:
                    raise RuntimeError("\nXPACDT: number of degrees of freedom"
                                       " given is less than 1.")
            else:
                raise RuntimeError("\nXPACDT: number of degrees of freedom"
                                   " not given in the input file.")

            if 'rpmd' in self:
                if 'beads' not in self.get("rpmd"):
                    raise RuntimeError("\nXPACDT: No number of beads "
                                       "given for RPMD in the input file.")
                if 'beta' not in self.get("rpmd"):
                    raise RuntimeError("\nXPACDT: No beta "
                                       "given for RPMD in the input file.")
                self._parse_beads(self.get('rpmd').get('beads'))

                try:
                    self.__beta = float(self.get('rpmd').get('beta'))
                    if self.__beta <= 0:
                        raise RuntimeError("\nXPACDT: Beta has to be greater"
                                           " than 0.")
                except ValueError as e:
                    raise type(e)(str(e) + "\nXPACDT: Beta cannot be"
                                           " converted to float.")
            else:
                self._parse_beads('1')
                # In the case when RPMD is not used (i.e. n_beads=1),
                # 'beta' should not be used anywhere, so setting it to NaN.
                self.__beta = np.nan
        else:
            raise KeyError("\nXPACDT: No system parameters are given in the"
                           " input file.")

        if self.__coordinates is not None:
            self.__format_coordinates()

        self.__commands = {k: self[k] for k in self.keys() if 'command' in k}
        for key in self.commands:
            self.commands[key]['name'] = key
            self.commands[key]['results'] = []

    @property
    def commands(self):
        """dict : Contains all input sections for 'commands' used in
        the analysis."""
        return self.__commands

    @property
    def masses(self):
        """(n_dof) ndarray of floats: Array containing the masses of each
        degree of freedom in au."""
        return self.__masses

    def _parse_masses(self, m):
        """Set the masses for each degree of freedom.

        Parameters
        ----------
        m : (n_dof) ndarray
            Array containing the masses in au.
        """
        if np.any([(i <= 0.0) for i in m]):
            raise ValueError("\nXPACDT: Masses of 0 or below given in input.")
        self.__masses = m.copy()

    @property
    def n_dof(self):
        """int: Number of degrees of freedom."""
        return self.__n_dof

    @property
    def n_beads(self):
        """(n_dof) list of ints: List containing the number of beads for each
        degree of freedom."""
        return self.__n_beads

    def _parse_beads(self, n_string):
        """Set the number of beads from a string given in the input file and
        check for consistency.

        Parameters
        ----------
        n_string : string
            The string defining the number of beads.
        """
        try:
            n = [int(i) for i in n_string.split()]
        except ValueError:
            raise ValueError("\nXPACDT: Number of beads not convertable to"
                             " int.")

        if len(n) != 1 and len(n) != self.n_dof:
            raise ValueError("\nXPACDT: Wrong length for number of beads"
                             " given. Either a single integer or one integer"
                             " per dof should be given.")

        if np.any([(i < 1) for i in n]):
            raise ValueError("\nXPACDT: Number of beads needs to be more than"
                             " zero.")

        if np.any([(i != 1 and (i % 2 != 0)) for i in n]):
            raise ValueError("\nXPACDT: Number of beads not 1 or even.")

        # Keep number of beads same for now
        if np.any([(i != n[0]) for i in n]):
            raise ValueError("\nXPACDT: Number of beads needs to be the same"
                             " for all degrees of freedom.")

        if len(n) == 1:
            self.__n_beads = n * self.n_dof
        else:
            self.__n_beads = n

        self["n_beads"] = self.n_beads
        self["max_n_beads"] = max(self.n_beads)

    @property
    def beta(self):
        """ float : Inverse temperature for ring polymer springs in a.u."""
        return self.__beta

    @property
    def coordinates(self):
        """(n_dof, n_beads) ndarray of floats: Array containing the coordinates
        of each degree of freedom in au. The first axis is the degrees of
        freedom and the second axis the beads."""

        # assure correct format.
        if self._c_type != 'xpacdt':
            self.__format_coordinates()
        return self.__coordinates

    @property
    def positionShift(self):
        """(n_dof) ndarray of floats: Array containing a shift that should
        be applied to the position centroid of each degree of freedom in au."""

        # assure correct format.
        if self._c_type != 'xpacdt':
            self.__format_coordinates()
        return self.__positionShift

    @property
    def momentumShift(self):
        """(n_dof) ndarray of floats: Array containing a shift that should
        be applied to the momentum centroid of each degree of freedom in au."""

        # assure correct format.
        if self._c_type != 'xpacdt':
            self.__format_coordinates()
        return self.__momentumShift

    @property
    def momenta(self):
        """(n_dof, n_beads) ndarray of floats: Array containing the momenta
        of each degree of freedom in au. The first axis is the degrees of
        freedom and the second axis the beads."""

        # assure correct format.
        if self._c_type != 'xpacdt':
            self.__format_coordinates()
        return self.__momenta

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

    def _parse_file(self):
        """
        Parses the text of an input file and saves it as a dictonary of
        sections. Each section then is also a dictonary of set variables.
        """

        no_comment_text = re.sub(r"#.*?\n", "\n", self._intext)
        no_newline_text = re.sub(r"(\n\s*?\n)+", "\n", no_comment_text)
        # *? is non-greedy; DOTALL matches also newlines
        section_texts = re.findall(r"(\$.*?)\$end", no_newline_text,
                                   flags=re.DOTALL | re.IGNORECASE)
        section_texts = [a.strip() for a in section_texts]

        for section in section_texts:
            if section[0:12] == "$coordinates":
                try:
                    match = re.search(r"\$(\w+).*?\n.*type.*=\s*(\S+)(.*)",
                                      section, flags=re.DOTALL)
                    keyword = match.group(1)
                    self._c_type = match.group(2)
                    values = match.group(3)
                except AttributeError:
                    raise AttributeError("\nXPACDT: Coordinate type not given.")

                if self._c_type == 'xyz':
                    self._parse_xyz(values)
                elif self._c_type == 'mass-value':
                    self._parse_mass_value(values)
                else:
                    raise ValueError("\nXPACDT: Coordinate type not"
                                     " understood: " + self._c_type)

            elif section[0:8] == "$momenta":
                d = StringIO(section[8:])
                self.__momenta = np.loadtxt(d, ndmin=2)
            elif section[0:14] == "$positionShift":
                d = StringIO(section[14:])
                self.__positionShift = np.loadtxt(d)
            elif section[0:14] == "$momentumShift":
                d = StringIO(section[14:])
                self.__momentumShift = np.loadtxt(d)
            else:
                match = re.search(r"\$(\w+)\W*(.*)", section,
                                  flags=re.DOTALL)
                keyword = match.group(1)
                try:
                    values = match.group(2)
                except IndexError:
                    values = ""

                if keyword in self:
                    raise KeyError("\nXPACDT: Key '" + keyword +
                                   "' defined twice!")
                else:
                    self[keyword] = self._parse_values(values)

    def _parse_xyz(self, values):
        """
        Parse coordinate input that have an atom symbol and the corresponding
        xyz coordinates per line. The format has to be as follows. The first
        entry per line gives the atom symbol. The next three entries give the
        xyz positions in bohr. Each RPMD bead has to come in a new line!

        The results are stored in self.masses, which are the au (not amu!)
        masses for each atom, and in self.coordinates, which is a two-d
        numpy array of floats.

        Parameters
        ----------
        values : str
            String representation of the input.
        """

        self._c_type = "xyz"
        d = StringIO(values)
        try:
            mc = np.loadtxt(d, ndmin=2,
                            converters={0: lambda s: units.atom_mass(str(s)[2])})
        except AttributeError as e:
            raise type(e)(str(e) + "\nXPACDT: Unknwon atomic symbol given!")
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many coordinates given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self._parse_masses(mc[:, 0])
        self.__coordinates = mc[:, 1:].copy()

    def _parse_mass_value(self, values):
        """
        Parse coordinate input that has a mass and a coordinate value per line.
        The format has to be as follows. The first entry per line gives the
        mass for this degree of freedom (in au, not amu). The next entry is
        the coordinate values for this degree of freedom. All bead values have
        to be given in one line.

        The results are stored in self.masses, which are the au (not amu!)
        masses for each atom, and in self.coordinates, which is a two-d
        numpy array of floats.

        Parameters
        ----------
        values : str
            String representation of the input.
        """

        self._c_type = "mass-value"
        d = StringIO(values)
        try:
            # TODO: This needs to be replaced if different number of beads
            # per DOF can be used!
            mc = np.loadtxt(d, ndmin=2)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many beads given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self._parse_masses(mc[:, 0])
        self.__coordinates = mc[:, 1:].copy()

    def _parse_values(self, values):
        """
        Parses the text of a section in the input file and creates
        a dictonary.

        Parameters
        ----------
        values : str
            The string of values read from the input file.

        Returns
        ----------
        dictonary
            A dictonary representation of the parsed text. For
            each line in the input a key, value pair is generated.
        """

        value_dict = {}
        for key_value_pair in re.split(r"\n", values):
            key_value = re.split(r"=", key_value_pair)

            if len(key_value) == 1:
                value_dict[key_value[0].strip()] = ""

            elif len(key_value) == 2:
                value_dict[key_value[0].strip()] = key_value[1].strip()

            else:
                raise ValueError("\nXPACDT: Too many '=' in a key-Value pair: "
                                 + key_value_pair)

        return value_dict

    def __format_coordinates(self):
        """ Reformat positions to match the desired format, i.e.,  The first
        axis is the degrees of freedom and the second axis the beads. If
        momenta are present we also reformat those. """

        if self._c_type == 'mass-value':
            if self.__coordinates.shape[0] != self.n_dof:
                raise RuntimeError("\nXPACDT: Number of coordinates given do"
                                   " not match n_dof given in the input.")

            # Check if only centroid value is given for more than one beads,
            # if yes, sample free ring polymer distribution
            if (self.__coordinates.shape[1] == 1 and max(self.n_beads) > 1):

                rp_coord = np.zeros((self.n_dof, max(self.n_beads)))
                rp_momenta = np.zeros((self.n_dof, max(self.n_beads)))
                NMtransform_type = self.get('rpmd').get("nm_transform",
                                                        "matrix")
                RPtransform = RPtrafo.RingPolymerTransformations(
                                self.n_beads, NMtransform_type)

                for i in range(self.n_dof):
                    rp_coord[i] = RPtransform.sample_free_rp_coord(
                        self.n_beads[i], self.masses[i], self.beta,
                        self.__coordinates[i, 0])
                    if self.__momenta is not None:
                        rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                            self.n_beads[i], self.masses[i], self.beta,
                            self.__momenta[i, 0])

                self.__coordinates = rp_coord.copy()
                if self.__momenta is not None:
                    self.__momenta = rp_momenta.copy()

            else:
                if self.__coordinates.shape[1] != max(self.n_beads):
                    raise RuntimeError("\nXPACDT: Number of bead coordinates"
                                       "given does not match n_beads given"
                                       " in the input.")

                self.__coordinates = self.__coordinates.reshape((self.n_dof, self["max_n_beads"]))

                try:
                    self.__momenta = self.__momenta.reshape(self.__coordinates.shape)

                # No momenta set
                except AttributeError:
                    pass
                # Wrong number of momenta given
                except ValueError as e:
                    raise type(e)(str(e) + "\nXPACDT: Number of given momenta "
                                           "and coordinates do not match!")

            self._flatten_shifts()
            self._c_type = 'xpacdt'

        elif self._c_type == 'xyz':
            if self.n_dof % 3 != 0:
                raise ValueError("\nXPACDT: Degrees of freedom needs to be"
                                 "multiple of 3 for 'xyz' format")

            if (self.__coordinates.shape[0] != self.n_dof // 3
                and self.__coordinates.shape[0] != np.sum(self.n_beads) // 3):

                raise RuntimeError("\nXPACDT: Number of coordinates given do"
                                   " not match n_dof and n_beads given in"
                                   " the input.")

            # TODO: Need to check if all dof of an atom has same nbeads?

            # Check if all bead values are given for each degree of freedom;
            # if not, sample from free rp distribution
            if (self.__coordinates.shape[0] == np.sum(self.n_beads) // 3):
                # reordering; only works for same number of beads for now!
                n_max = max(self.n_beads)
                self._parse_masses(self.masses[::n_max])
                self._parse_masses(np.array([self.masses[i//3]
                                            for i in range(self.n_dof)]))
                self.__coordinates = np.array([self.__coordinates[i::n_max]
                                               for i in range(n_max)])\
                    .flatten().reshape((self.n_dof, -1), order='F')

                try:
                    self.__momenta = np.array([self.__momenta[i::n_max]
                                               for i in range(n_max)])\
                        .flatten().reshape(self.__coordinates.shape, order='F')

                # No momenta set
                except (AttributeError, TypeError):
                    pass
                # Wrong number of momenta given
                except ValueError as e:
                    raise type(e)(str(e) + "\nXPACDT: Number of given momenta "
                                           "and coordinates does not match!")
            else:
                masses_dof = np.zeros(self.n_dof)
                rp_coord = np.zeros((self.n_dof, max(self.n_beads)))
                rp_momenta = np.zeros((self.n_dof, max(self.n_beads)))
                NMtransform_type = self.get('rpmd').get("nm_transform",
                                                        "matrix")
                RPtransform = RPtrafo.RingPolymerTransformations(
                                self.n_beads, NMtransform_type)

                for i in range(self.n_dof):
                    masses_dof[i] = self.masses[i//3]
                    rp_coord[i] = RPtransform.sample_free_rp_coord(
                        self.n_beads[i], masses_dof[i], self.beta,
                        self.__coordinates[i // 3, i % 3])
                    if self.__momenta is not None:
                        rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                            self.n_beads[i], masses_dof[i], self.beta,
                            self.__momenta[i // 3, i % 3])

                self._parse_masses(masses_dof)
                self.__coordinates = rp_coord.copy()
                if self.__momenta is not None:
                    self.__momenta = rp_momenta.copy()

            self._flatten_shifts()

            self._c_type = 'xpacdt'

    def _flatten_shifts(self):
        """Function to flatten then given position or momentum shift to
        one-dimensional arrays.
        """
        if self.__positionShift is not None:
            self.__positionShift = self.__positionShift.reshape(-1)
            if len(self.__positionShift) != self.n_dof:
                raise RuntimeError("\nXPACDT: Number of coordinates in "
                                   "position shift does not match number "
                                   "of degrees of freedom given: "
                                   + str(self.n_dof) + " != "
                                   + str(len(self.__positionShift)))

        if self.__momentumShift is not None:
            self.__momentumShift = self.__momentumShift.reshape(-1)
            if len(self.__momentumShift) != self.n_dof:
                raise RuntimeError("\nXPACDT: Number of coordinates in "
                                   "momentum shift does not match number "
                                   "of degrees of freedom given: "
                                   + str(self.n_dof) + " != "
                                   + str(len(self.__momentumShift)))

        return

# TODO: we changed some requirements which might fail tests -> check
