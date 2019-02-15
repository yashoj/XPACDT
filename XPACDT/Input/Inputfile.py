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

import os
import re

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

        self._parseFile()
        return

    def getSection(self, name):
        """
        Obtain a dictonary of a given section in the input file.

        Parameters
        ----------
        name : str
               The name of the requested section.

         Returns
         ----------
         dict
             A dictonary containing all Key-Value pairs of the requested
             section.
        """

        return self._input.get(name)

    def _parseFile(self):
        """
        Parses the text of an input file and saves it as a dictonary of
        sections. Each section then is also an dictonary of set variables.
        """

        no_comment_text = re.sub("#.*\n", "", self._intext)
        # *? is non-greedy; DOTALL matches also newlines
        section_texts = re.findall("(\$.*?)\$end", no_comment_text,
                                   flags=re.DOTALL | re.IGNORECASE)
        section_texts = [a.strip() for a in section_texts]

        self._input = {}
        for section in section_texts:
            match = re.search("\$(\w+)\n(.*)", section, flags=re.DOTALL)
            keyword = match.group(1)
            values = match.group(2)

            if keyword in self._input:
                # TODO: Use Better Error
                raise IOError("Key '" + keyword + "' defined twice!")
            else:
                self._input[keyword] = self._parseValues(values)

        return

    def _parseValues(self, values):
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
