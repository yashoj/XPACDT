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
from molmod.periodic import periodic
import numpy as np
import os
import re


class Inputfile(collections.MutableMapping):
    """Basic representation of all the input parameters given to XPACDT. It
    inherits from the MutableMapping Abstract Base Class defined in the
    collections module.

    Parameters
    ----------
    inputfile: str
        Filename of the input file.
    """

    def __init__(self, inputfile):

        self.store = dict()
        self.__masses = None
        self.__coordinates = None

        self._filename = inputfile
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(ENOENT, "Input file does not exist!",
                                    self._filename)

        with open(self._filename, 'r') as infile:
            self._intext = infile.read()

        self._parse_file()
        if self.coorindates is not None:
            self.__format_coordinates()

    @property
    def masses(self):
        """ndarray of floats: Array containing the masses of each degree of
        freedom in au."""
        return self.__masses

    @masses.setter
    def masses(self, m):
        self.__masses = m

    @property
    def coordinates(self):
        """two-dimensional ndarray of floats: Array containing the coordinates
        of each degree of freedom in au. The first axis is the degrees of
        freedom and the second axis the beads."""

        # assure correct format.
        if self._ctype != 'xpacdt':
            self.__format_coordinates()
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, c):
        self.__coordinates = c

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

        no_comment_text = re.sub("#.*?\n", "\n", self._intext)
        no_newline_text = re.sub("(\n\s*?\n)+", "\n", no_comment_text)
        # *? is non-greedy; DOTALL matches also newlines
        section_texts = re.findall("(\$.*?)\$end", no_newline_text,
                                   flags=re.DOTALL | re.IGNORECASE)
        section_texts = [a.strip() for a in section_texts]

        for section in section_texts:
            if section[0:12] == "$coordinates":
                try:
                    match = re.search("\$(\w+).*?\n.*type.*=\s*(\S+)(.*)",
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
                self._momenta = np.loadtxt(d)
            else:
                match = re.search("\$(\w+).*?\n(.*)", section, flags=re.DOTALL)
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

        d = StringIO(values)
        try:
            # TODO write a small wrapper for isotope masses!
            mc = np.loadtxt(d, ndmin=2,
                            converters={0: lambda s: periodic[str(s)[2]].mass})
        except AttributeError as e:
            raise type(e)(str(e) + "\nXPACDT: Unknwon atomic symbol given!")
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many coordinates given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self.masses = mc[:, 0]
        self.coordinates = mc[:, 1:]

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

        d = StringIO(values)
        try:
            mc = np.loadtxt(d, ndmin=2)
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many beads given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self.masses = mc[:, 0]
        self.coordinates = mc[:, 1:]

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
        for key_value_pair in re.split("\n", values):
            key_value = re.split("=", key_value_pair)

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
        axis is the degrees of freedom and the second axis the beads."""

        if self._c_type == 'mass-value':
            self.coordinates = self.coordinates.reshape((self.n_dof, -1))
            if self._momenta is not None:
                self._momenta = self._momenta.reshape(self.coordinates.shape)
            self._c_type = 'xpacdt'
        elif self._c_type == 'xyz':
            self.positions = self.positions.T.reshape((self.n_dof, -1))
            if self._momenta is not None:
                self._momenta = self._momenta.T.reshape(self.coordinates.shape)
            self._c_type = 'xpacdt'
