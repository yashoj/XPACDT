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

from io import StringIO
from molmod.periodic import periodic
import numpy as np
import os
import re
import sys

from errno import ENOENT


class Inputfile(object):
    """This class reads and stores an XPACDT input file.

    Parameters
    ----------
    filename : str
               Filename of the input file.

    """

    def __init__(self, filename):

        self._filename = filename
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(ENOENT,
                                    "Input file does not exist!",
                                    self._filename)

        with open(self._filename, 'r') as infile:
            self._intext = infile.read()

        self._parse_file()
        return

    def get_section(self, name):
        """
        Obtain a dictonary of a given section in the input file.

        Parameters
        ----------
        name : str
            The name of the requested section.

        Returns
        -------
        out : dict
            A dictonary containing all Key-Value pairs of the requested
            section.
        """

        return self._input.get(name)

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

        self._input = {}
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

                if keyword in self._input:
                    # TODO: Use Better Error
                    raise IOError("Key '" + keyword + "' defined twice!")
                else:
                    self._input[keyword] = self._parse_values(values)

        return

    def _parse_xyz(self, values):
        """
        Parse coordinate input that have an atom symbol and the corresponding
        xyz coordinates per line. The format has to be as follows. The first
        entry per line gives the atom symbol. The next three entries give the
        xyz positions in bohr. Each RPMD bead has to come in a new line!

        The results are stored in self._masses, which are the au (not amu!)
        masses for each atom, and in self._coordinates, which is a two-d
        numpy array of floats.

        Parameters
        ----------
        values : str
            String representation of the input.
        """

        d = StringIO(values)
        try:
            # TODO write a small wrapper for isotope masses!
            mc = np.loadtxt(d, ndmin=2, converters={0: lambda s:
                            periodic[str(s)[2]].mass})
        except AttributeError as e:
            raise type(e)(str(e) + "\nXPACDT: Unknwon atomic symbol given!")
        except ValueError as e:
            raise type(e)(str(e) + "\nXPACDT: Too few/many coordinates given. "
                                   "Please check the error for the line "
                                   "number with the first inconsistency.")

        self._masses = mc[:, 0]
        self._coordinates = mc[:, 1:]

        return

    def _parse_mass_value(self, values):
        """
        Parse coordinate input that has a mass and a coordinate value per line.
        The format has to be as follows. The first entry per line gives the
        mass for this degree of freedom (in au, not amu). The next entry is
        the coordinate value for this degree of freedom. All bead values can
        be given in one line if the same number of beads is used for all
        degrees of freedom.

        The results are stored in self._masses, which are the au (not amu!)
        masses for each atom, and in self._coordinates, which is a two-d
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

        self._masses = mc[:, 0]
        self._coordinates = mc[:, 1:]

        return

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

    # TODO: this needs to go away!
    # Dictionary of element names and atomic number:
    periodic_table = {
            'H'  :  { 'atomic_number': 1,
                      'atomic_mass'  : 1.007825 * 1822.8885300626
                    },
            'D'  :  { 'atomic_number': 1,
                      'atomic_mass'  : 2.0140 * 1822.8885300626
                    },
            'He' :  { 'atomic_number': 2,
                      'atomic_mass'  : 4.00260 * 1822.8885300626  # He 4
                    },
            'Li' :  { 'atomic_number': 3,
                      'atomic_mass'  : 7.016003 * 1822.8885300626  # Li 7
                    },
            'Be' :  { 'atomic_number': 4,
                      'atomic_mass'  : 9.012182 * 1822.8885300626
                    },
            'B'  :  { 'atomic_number': 5,
                      'atomic_mass'  : 11.009305 * 1822.8885300626 # B 11
                    },
            'C'  :  { 'atomic_number': 6,
                      'atomic_mass'  : 12.000000 * 1822.8885300626  # C 12
                    },
            'N'  :  { 'atomic_number': 7,
                      'atomic_mass'  : 14.003074 * 1822.8885300626  # N 14
                    },
            'O'  :  { 'atomic_number': 8,
                      'atomic_mass'  : 15.994915 * 1822.8885300626  # O 16
                    },
            'F'  :  { 'atomic_number': 9,
                      'atomic_mass'  : 18.9984032 * 1822.8885300626
                    },
            'Ne' :  { 'atomic_number': 10,
                      'atomic_mass'  : 19.992435 * 1822.8885300626  # Ne 20
                    },
            'I' :  { 'atomic_number': 53,
                      'atomic_mass' : 126.904473 * 1822.8885300626  # I 53
                    }
            }
