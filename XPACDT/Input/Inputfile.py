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

        self._raw_coordinates = None
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

        if self._raw_coordinates is not None:
            self.__format_coordinates()

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
            raise XPACDTInputError("Number of beads not 1 nor even.",
                                   section="rpmd",
                                   key="beads")

        # Keep number of beads same for now
        if np.any([(i != n[0]) for i in n]):
            raise NotImplementedError("Different number of beads for each "
                                      "degrees of freedom has not yet been "
                                      "implemented.")

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
            raise KeyError(f"Key can only be set once in InputFile, but {key}"
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

        raise AttributeError()

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
                    match = re.search(r"\$(\w+).*?\n.*type.*=\s*(\S+)(.*)",
                                      section, flags=re.DOTALL)
                    keyword = match.group(1)
                    self._c_type = match.group(2)
                    values = match.group(3)
                except AttributeError:
                    raise XPACDTInputError(section="coordinates", key="type")

                try:
                    if self._c_type == 'xyz':
                        atom_symbols, masses, coord = parse_xyz(
                                                        input_string=values)
                        self["atom_symbols"] = atom_symbols
                        self["masses"] = masses
                        self._raw_coordinates = coord
                    elif self._c_type == 'mass-value':
                        masses, coord = parse_mass_value(values)
                        self["masses"] = masses
                        self._raw_coordinates = coord
                    else:
                        raise XPACDTInputError(
                            f"Invalid coordinate type {self._c_type}. Allowed "
                            "types are xyz and mass-value.",
                            section="coordinates",
                            key="type")
                except ValueError as e:
                    raise XPACDTInputError(
                        "Coordinate data was not correctly formatted.",
                        section="coordinates",
                        caused_by=e)

            elif section[0:8] == "$momenta":
                d = StringIO(section[8:])
                self._raw_momenta = np.loadtxt(d, ndmin=2)
            elif section[0:14] == "$positionShift":
                d = StringIO(section[14:])
                self["positionShift"] = np.loadtxt(d).flatten()

                if len(self["positionShfit"]) != self["n_dof"]:
                    raise XPACDTInputError(
                        "Position shift dimension does not match the number "
                        "of degrees of freedom")
            elif section[0:14] == "$momentumShift":
                d = StringIO(section[14:])
                self["momentumShift"] = np.loadtxt(d).flatten()

                if len(self["positionShfit"]) != self["n_dof"]:
                    raise XPACDTInputError(
                        "Position shift dimension does not match the number "
                        "of degrees of freedom")
            else:
                match = re.search(r"\$(\w+)\W*(.*)", section,
                                  flags=re.DOTALL)
                keyword = match.group(1)
                try:
                    values = match.group(2)
                except IndexError:
                    values = ""

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

    def __format_coordinates(self):
        """ Reformat positions to match the desired format, i.e.,  The first
        axis is the degrees of freedom and the second axis the beads. If
        momenta are present we also reformat those. """

        if self._raw_momenta is not None:
            if self._raw_coordinates.shape != self._raw_momenta.shape:
                raise XPACDTInputError(
                    "Number of momenta and coordinates does not match",
                    section="coordinates/momenta")

        if self._c_type == 'mass-value':
            dof_given, beads_given = self._raw_coordinates.shape
            if dof_given != self.n_dof:
                raise XPACDTInputError(
                    f"Number of coordinates ({dof_given}) given does "
                    f"not match n_dof given in the input ({self.n_dof}).",
                    section="coordinates")

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
                        self._raw_coordinates[i, 0])
                    if self._raw_momenta is not None:
                        rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                            self.n_beads[i], self.masses[i], self.beta,
                            self._raw_momenta[i, 0])

                self["coordinates"] = rp_coord.copy()

                if self._raw_momenta is not None:
                    self["momenta"] = rp_momenta.copy()

            else:
                if beads_given != self["max_n_beads"]:
                    raise XPACDTInputError(
                        f"Number of bead coordinates ({beads_given}) given "\
                        f"does not match the expected number of beads "
                        f"({self.max_n_beads}).",
                        section="coordinates")

                shape = (self.n_dof, self["max_n_beads"])
                self["coordinates"] = self._raw_coordinates.reshape(shape)

                if self._raw_momenta is not None:
                    self["momenta"] = self._raw_momenta.reshape(shape)

        elif self._c_type == 'xyz':
            if self.n_dof % 3 != 0:
                raise XPACDTInputError(
                    "Degrees of freedom needs to be  multiple of 3 for "
                    f"'xyz' format, {self.n_dof} was given.",
                    section="coordinates")

            dof_given, beads_given = self._raw_coordinates.shape
            if (dof_given != self.n_dof // 3
                    and dof_given != np.sum(self.n_beads) // 3):

                raise XPACDTInputError(
                    f"Number of coordinates given does has n_dof "
                    "({dof_given}). This doesn't match any of the allowed "
                    f"values ({self.n_dof // 3} or "
                    f"{np.sum(self.n_beads) // 3}).",
                    section="coordinates")

            # TODO: Need to check if all dof of an atom has same nbeads?

            # Check if all bead values are given for each degree of freedom;
            # if not, sample from free rp distribution
            if (dof_given == np.sum(self.n_beads) // 3):
                # reordering; only works for same number of beads for now!
                n_max = max(self.n_beads)
                # Bypass the 'only write once' policy by accessing .store
                # directly
                self.store["masses"] = self.masses[::n_max]
                self["coordinates"] = np.array(
                    [self._raw_coordinates[i::n_max] for i in range(n_max)])\
                    .flatten().reshape((self.n_dof, -1), order='F')

                if self._raw_momenta is not None:
                    self["momenta"] = np.array(
                        [self._raw_momenta[i::n_max] for i in range(n_max)])\
                        .flatten().reshape(self["coordinates"].shape, order='F')

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
                        self._raw_coordinates[i // 3, i % 3])
                    if self._raw_momenta is not None:
                        rp_momenta[i] = RPtransform.sample_free_rp_momenta(
                            self.n_beads[i], masses_dof[i], self.beta,
                            self._raw_momenta[i // 3, i % 3])

                self["masses"] = masses_dof
                self["coordinates"] = rp_coord.copy()
                if self._raw_momenta is not None:
                    self["momenta"] = rp_momenta.copy()

        self._c_type = 'xpacdt'
