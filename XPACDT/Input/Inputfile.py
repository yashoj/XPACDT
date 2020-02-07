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
import yaml

import XPACDT.Dynamics.RingPolymerTransformations as RPtrafo
import XPACDT.Tools.Units as units

from XPACDT.Input.Error import XPACDTInputError
from XPACDT.Tools.Coordinates import parse_xyz, parse_mass_value


class Inputfile(collections.MutableMapping):
    """Basic representation of all the input parameters given to XPACDT. It
    inherits from the MutableMapping Abstract Base Class defined in the
    collections module. This makes the Inputfile behave like a dictionnary.

    For covenience all attributes can be accessed either through the standard
    way (`Inputfile.attribute`) or through the dictionnary syntax
    (`Inputfile["attribute"]`).

    Parameters
    ----------
    inputfile: str
        Filename of the input file.

    Attributes:
    -----------
    commands
    masses
    atom_symbols
    n_dof
    n_beads
    max_n_beads
    beta
    coordinates
    positionShift
    momenta
    momentumShift
    """

    def __init__(self, inputfile):

        self.store = dict()

        self._filename = inputfile
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(ENOENT, "Input file does not exist!",
                                    self._filename)

        with open(self._filename, 'r') as infile:
            self._intext = infile.read()

        self._coordinates_string = None
        self._raw_momenta = None

        self._parse_file()

        try:
            system = self["system"]
        except KeyError:
            raise XPACDTInputError("The $system section is missing.")

        try:
            self["n_dof"] = int(system["dof"])
        except KeyError:
            raise XPACDTInputError(section="system", key="dof")
        except ValueError as e:
            raise XPACDTInputError("The number of degree of freedom can "
                                   "not be converted to int.",
                                   section="system",
                                   key="dof",
                                   caused_by=e)

        if ("positionShift" in self and
                len(self["positionShift"]) != self["n_dof"]):
            raise XPACDTInputError(
                "Position shift dimension does not match the number "
                "of degrees of freedom",
                section="positionShift")

        if self.n_dof < 1:
            raise XPACDTInputError("The number of degree of freedom must be "
                                   "greater or equal to 1.",
                                   section="system",
                                   key="dof")

        if "rpmd" in self:
            rpmd = self["rpmd"]

            try:
                self._parse_beads(rpmd["beads"])
            except KeyError:
                raise XPACDTInputError(section="rpmd", key="beads")

            try:
                self["beta"] = float(rpmd["beta"])
            except KeyError:
                raise XPACDTInputError(section="rpmd", key="beta")
            except ValueError as e:
                raise XPACDTInputError("beta can not be converted to float",
                                       section="rpmd",
                                       key="beta",
                                       caused_by=e)
        else:
            self._parse_beads('1')
            # In the case when RPMD is not used (i.e. n_beads=1),
            # 'beta' should not be used anywhere, so setting it to NaN.
            self["beta"] = np.nan

        if self._coordinates_string is not None:
            self.__parse_coordinates_string()

        self["commands"] = {k: self[k] for k in self.keys() if 'command' in k}
        for key in self.commands:
            self.commands[key]['name'] = key
            self.commands[key]['results'] = []

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
        except ValueError as e:
            raise XPACDTInputError("Number of beads not convertable to int.",
                                   section="rpmd",
                                   key="beads",
                                   caused_by=e)

        if len(n) != 1 and len(n) != self.n_dof:
            raise XPACDTInputError(
                "Wrong length for number of beads given. Either a single "
                "integer or one integer per dof should be given.",
                section="rpmd",
                key="beads")

        if np.any([(i < 1) for i in n]):
            raise XPACDTInputError("Number of beads needs to be more than"
                                   " zero.",
                                   section="rpmd",
                                   key="beads")

        if np.any([(i != 1 and (i % 2 != 0)) for i in n]):
            raise XPACDTInputError("Number of beads must be either 1 or even.",
                                   section="rpmd",
                                   key="beads")

        # Keep number of beads same for now
        if np.any([(i != n[0]) for i in n]):
            raise NotImplementedError("Different number of beads for each "
                                      "degrees of freedom is not yet "
                                      "available.")

        if len(n) == 1:
            # Clone the number of beads for each degree of freedom
            # NOTE works because n is a list
            self["n_beads"] = n * self.n_dof
        else:
            self["n_beads"] = n

        self["max_n_beads"] = max(self.n_beads)

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if key in self.store:
            raise KeyError(f"Key can only be set once in InputFile, but {key} "
                           f"recieved a new value {value}.")

        if key == "masses":
            if np.any([(m <= 0.0) for m in value]):
                raise XPACDTInputError("Negative mass given.")

        self.store[self.__keytransform__(key)] = value

    def __contains__(self, key):
        return key in self.store

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

    def __getattr__(self, attribute):
        # Checking if the attribute is "store" avoid infinite recurstion when
        # loading from pickle file.
        if attribute != "store" and attribute in self.store:
            return self[attribute]

        raise AttributeError(
            f"Object of type Inputfile has no attribute named {attribute}.")

    def __deepcopy__(self, memo):
        return Inputfile(self._filename)

    def __str__(self):
        d = {**self.store}
        for key, val in d.items():
            try:
                # Transform every numpy array to a list of lists
                d[key] = val.tolist()
            # When the value is not a numpy array, the .tolist() function
            # does not exist and an AttributeError is raise. In this case
            # nothing need to be done as the object will be directly
            # serializable by YAML.
            except AttributeError:
                pass
        return yaml.dump(d)

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
                    match = re.search(r"\$(\w+).*?\n.*type\s*=\s*(\S+)(.*)",
                                      section, flags=re.DOTALL)
                    keyword = match.group(1)
                    self._c_type = match.group(2)
                    # We defer the parsing of the coordinates to later, for
                    # when the number of beads is known.
                    self._coordinates_string = match.group(3)
                except AttributeError:
                    raise XPACDTInputError(section="coordinates", key="type")

                if self._c_type not in ['xyz', 'mass-value']:
                    raise XPACDTInputError(
                        f"Invalid coordinate type {self._c_type}. Allowed "
                        "types are 'xyz' and 'mass-value'.",
                        section="coordinates",
                        key="type")

            elif section[0:8] == "$momenta":
                d = StringIO(section[8:])
                self._raw_momenta = np.loadtxt(d, ndmin=2)
            elif section[0:14] == "$positionShift":
                d = StringIO(section[14:])
                self["positionShift"] = np.loadtxt(d).flatten()
            elif section[0:14] == "$momentumShift":
                d = StringIO(section[14:])
                self["momentumShift"] = np.loadtxt(d).flatten()
            else:
                match = re.search(r"\$(\w+)\W*(.*)", section,
                                  flags=re.DOTALL)
                keyword = match.group(1)
                try:
                    values = match.group(2)
                except IndexError:
                    values = ""

                if keyword in self:
                    raise XPACDTInputError(
                        "Section defined twice.",
                        section=keyword)
                self[keyword] = self._parse_values(values)

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
                raise XPACDTInputError("Too many '=' in the following line:\n"
                                       f"{key_value}")

        return value_dict

    def __parse_coordinates_string(self):
        """ Reformat positions to match the desired format, i.e.,  The first
        axis is the degrees of freedom and the second axis the beads. If
        momenta are present we also reformat those. """

        if self._c_type == 'mass-value':
            self["masses"], coord = parse_mass_value(self._coordinates_string)

        if self._c_type == 'xyz':
            self["atom_sybols"], self["masses"], coord = \
                parse_xyz(input_string=self._coordinates_string,
                          n_beads=self["n_beads"])

        if self._raw_momenta is not None:
            try:
                self._raw_momenta = self._raw_momenta.reshape(coord.shape)
            except ValueError as e:
                raise XPACDTInputError(
                    "Number of momenta and coordinates does not match",
                    section="coordinates/momenta",
                    caused_by=e)

        dof_given, beads_given = coord.shape

        # Check if only centroid value is given for more than one beads,
        # if yes, sample free ring polymer distribution
        if (beads_given == 1 and self["max_n_beads"] > 1):
            rp_coord = np.zeros((self.n_dof, self["max_n_beads"]))
            rp_momenta = np.zeros((self.n_dof, self["max_n_beads"]))
            NMtransform_type = self['rpmd'].get("nm_transform", "matrix")
            RPtransform = RPtrafo.RingPolymerTransformations(
                            self.n_beads, NMtransform_type)

            for i in range(self.n_dof):
                rp_coord[i] = RPtransform.sample_free_rp_coord(
                    self.n_beads[i], self.masses[i], self.beta,
                    coord[i, 0])
                if self._raw_momenta is not None:
                    rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                        self.n_beads[i], self.masses[i], self.beta,
                        self._raw_momenta[i, 0])

            self["coordinates"] = rp_coord.copy()

            if self._raw_momenta is not None:
                self["momenta"] = rp_momenta.copy()

        elif beads_given == self["max_n_beads"]:
            self["coordinates"] = coord

            if self._raw_momenta is not None:
                self["momenta"] = self._raw_momenta
        else:
            raise XPACDTInputError(
                f"Number of bead coordinates ({beads_given}) given "
                "does not match the given number of beads "
                f"({self.max_n_beads}).",
                section="rpmd",
                key="n_beads")

        self._c_type = 'xpacdt'
