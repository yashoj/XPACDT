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
    n_dof : int (optional if specified in input file)
        Number of degrees of freedom in the system.
    n_beads : (n_dof) list of int (optional if specified in input file)
        Number of beads for each degrees of freedom. Default: 1 for each dof
    """

    def __init__(self, inputfile):

        self.store = dict()
        self.__momenta = None
        self.__masses = None
        self.__coordinates = None

        self._filename = inputfile
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(ENOENT, "Input file does not exist!",
                                    self._filename)

        with open(self._filename, 'r') as infile:
            self._intext = infile.read()

        self._parse_file()
        if 'system' in self.store:
            self.n_dof = int(self.get('system').get('dof'))
            if 'rpmd' in self.store:
                assert('beads' in self.get("rpmd")), "No number of beads " \
                       "given for RPMD."
                assert('beta' in self.get("rpmd")), "No beta " \
                                                    "given for RPMD."
                self.n_beads = self.get('rpmd').get('beads')
                self.beta = float(self.get('rpmd').get('beta'))
            else:
                self.n_beads = '1'

        if self.__coordinates is not None:
            self.__format_coordinates()

    @property
    def masses(self):
        """(n_dof) ndarray of floats: Array containing the masses of each
        degree of freedom in au."""
        return self.__masses

    @masses.setter
    def masses(self, m):
        assert (np.all([(i >= 0.0) for i in m])), "Negative mass assigned"
        self.__masses = m

    @property
    def n_dof(self):
        """int: Number of degrees of freedom."""
        return self.__n_dof

    @n_dof.setter
    def n_dof(self, d):
        assert (d > 0), ("Number of degrees of freedom needs to be "
                         "greater than zero")
        self.__n_dof = d

    @property
    def n_beads(self):
        """(n_dof) list of ints: List containing the number of beads for each
        degree of freedom."""
        return self.__n_beads

    @n_beads.setter
    def n_beads(self, n_string):
        try:
            n = [int(i) for i in n_string.split()]
        except ValueError:
            raise ValueError("Number of beads not convertable to int.")

        assert (len(n) == 1 or len(n) == self.n_dof), "Wrong number of " \
                                                      "beads given."
        assert (np.all([(i > 0) for i in n])), ("Number of beads needs to be"
                                                " more than zero")
        assert (np.all([(i == 1 or i % 2 == 0) for i in n])),\
               ("Number of beads not 1 or even")
        # Keep number of beads same for now
        # TODO: Adjust structure to handle different number of beads
        assert (np.all([(i == n[0]) for i in n])), \
               ("Number of beads not same for all degree of freedom")

        if len(n) == 1:
            self.__n_beads = n * self.n_dof
        else:
            self.__n_beads = n

    @property
    def beta(self):
        """ float : Inverse temperature for ring polymer springs in a.u."""
        return self.__beta

    @beta.setter
    def beta(self, b):
        assert (b is None or b > 0), "Beta 0 or less."
        self.__beta = float(b)

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
    def momenta(self):
        """(n_dof, n_beads) ndarray of floats: Array containing the momenta
        of each degree of freedom in au. The first axis is the degrees of
        freedom and the second axis the beads."""

        # assure correct format.
        if self._c_type != 'xpacdt':
            self.__format_coordinates()
        return self.__momenta

# TODO: Do we need to document these basic functions of the interface?
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
        sections. Each section then is also an dictonary of set variables.
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
                    raise IOError("Coordinate type not given.")

                if self._c_type == 'xyz':
                    self._parse_xyz(values)
                elif self._c_type == 'mass-value':
                    self._parse_mass_value(values)
                else:
                    raise IOError("Coordinate type not understood: "
                                  + self._c_type)

            elif section[0:8] == "$momenta":
                d = StringIO(section[8:])
                self.__momenta = np.loadtxt(d)
            else:
                match = re.search(r"\$(\w+).*?\n(.*)", section,
                                  flags=re.DOTALL)
                keyword = match.group(1)
                values = match.group(2)

                if keyword in self.store:
                    # TODO: Use Better Error
                    raise IOError("Key '" + keyword + "' defined twice!")
                else:
                    self.store[keyword] = self._parse_values(values)

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
            # TODO write a small wrapper for isotope masses!
            mc = np.loadtxt(d, ndmin=2,
 #                           converters={0: lambda s: periodic[str(s)[2]].mass})
                            converters={0: lambda s: units.atom_mass(str(s)[2])})
        except AttributeError as e:
            raise type(e)(str(e) + "\nXPACDT: Unknwon atomic symbol given!")
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many coordinates given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self.masses = mc[:, 0].copy()
        self.__coordinates = mc[:, 1:].copy()

    def _parse_mass_value(self, values):
        """
        Parse coordinate input that has a mass and a coordinate value per line.
        The format has to be as follows. The first entry per line gives the
        mass for this degree of freedom (in au, not amu). The next entry is
        the coordinate value for this degree of freedom. All bead values can
        be given in one line if the same number of beads is used for all
        degrees of freedom.

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
            mc = np.loadtxt(d, ndmin=2)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many beads given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self.masses = mc[:, 0].copy()
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
                # TODO: Use Better Error
                raise IOError("Too many '=' in a key-Value pair: "
                              + key_value_pair)

        return value_dict

    def __format_coordinates(self):
        """ Reformat positions to match the desired format, i.e.,  The first
        axis is the degrees of freedom and the second axis the beads. If
        momenta are present we also reformat those. """

        if self._c_type == 'mass-value':
            assert (self.__coordinates.shape[0] == self.n_dof),\
               ("Degrees of freedom and coordinate shape do not match")

            # Check if only centroid value is given for more than one beads,
            # if yes, sample free ring polymer distribution
            if (self.__coordinates.shape[1] == 1
                    and np.all([i != 1 for i in self.n_beads])):
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
                self.__coordinates = self.__coordinates.reshape((self.n_dof, -1))

                try:
                    self.__momenta = self.__momenta.reshape(self.__coordinates.shape)

                # No momenta set
                except AttributeError:
                    pass
                # Wrong number of momenta given
                except ValueError as e:
                    raise type(e)(str(e) + "\nXPACDT: Number of given momenta "
                                           "and coordinates do not match!")

            self._c_type = 'xpacdt'

        elif self._c_type == 'xyz':
            assert (self.n_dof % 3 == 0), ("Degrees of freedom needs to be"
                                           "multiple of 3 for 'xyz' format")
            # TODO: Need to check if all dof of an atom has same nbeads?
            assert ((self.__coordinates.shape[0] == self.n_dof / 3)
                    or (self.__coordinates.shape[0] == int(np.sum(self.n_beads) / 3))),\
                   ("Degrees of freedom and coordinate shape do not match")

            # Check if all bead values are given for each degree of freedom;
            # if not, sample from free rp distribution
            if (self.__coordinates.shape[0] == int(np.sum(self.n_beads) / 3)):
                # reordering; only works for same number of beads for now!
                n_max = max(self.n_beads)
                self.masses = self.masses[::n_max]
                self.masses = np.array([self.masses[i//3]
                                        for i in range(self.n_dof)])
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

                self.masses = masses_dof.copy()
                self.__coordinates = rp_coord.copy()
                if self.__momenta is not None:
                    self.__momenta = rp_momenta.copy()

            self._c_type = 'xpacdt'
