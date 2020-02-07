#!/usr/bin/env python3

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

""" This module is used to deal with coordinates parsing and formatting.
"""

import numpy as np

from io import StringIO
from pathlib import Path

from XPACDT.Tools.Units import atom_mass, angstrom_to_bohr


def parse_mass_value(input_string):
    """
    Parse coordinate input that has a mass and a coordinate value per line.
    The format has to be as follows. The first entry per line gives the
    mass for this degree of freedom (in au, not amu). The next entry is
    the coordinate values for this degree of freedom. All bead values have
    to be given in one line.

    Parameters
    ----------
    input_string : str
        String representation of the input.

    Return
    ------
    masses : (n_dof,) ndarray of floats
        Array of the masses in the system.

    coordinates : (n_dof, n_beads) ndarray of floats
        Array of the coordinates of the masses.
    """
    d = StringIO(input_string)
    try:
        # TODO: This needs to be replaced if different number of beads
        # per DOF can be used!
        mc = np.loadtxt(d, ndmin=2)
    except ValueError:
        raise ValueError("The mass-value data is not correctly formatted."
                         f"mass-value data:\n{input_string}")

    masses = mc[:, 0]
    coordinates = mc[:, 1:]
    return masses, coordinates


def format_xyz(atoms, r, full=True):
    """
    Transform the given nuclear degrees of freedom `r` in a string in
    XYZ format.

    Parameters
    ----------
    atoms : (n_atoms,) list of str
        Atom names

    r : (n_dof,) ndarray of floats
        Nuclei positions

    full : bool, optional, default: True
        Whether to return full file content or to exclude 2 first lines
    """
    n_atoms = len(atoms)
    positions = r.reshape(n_atoms, 3)
    lines = [" ".join([atom, *map(str, pos)])
             for atom, pos in zip(atoms, positions)]
    core = "\n".join(lines)

    if not full:
        return core

    return f"{n_atoms}\nbohr\n{core}"


def parse_xyz(input_string=None, filename=None, n_beads=None, n_dof=None):
    """
    Parse a XYZ formatted string (passed with the `input_string` keyword), or
    file (which name or path is given by the `filename` keyword).

    Strings are expected to only exactly contain the coordinates, and to use
    bohr (a.u.) as unit.

    Files are expected to have 2 special lines at the start. The first
    containing the number of atoms (this line is ignored here) and the second
    being a comment line. If the second line contains the words "bohr" or
    "a.u." this is used as unit, otherwise the unit used is Angstrom. In any
    case the output is in bohr. This follows the Molcas convention for XYZ
    files.

    If `n_beads` is given, the input is interpreting as having multiple beads,
    the atoms being repeated for as many times as there are beads associated to
    them.

    Raise `ValueError` if the input can not be parsed.

    Parameters
    ----------
    input_string : str, optional. Default: None
        A string containing coordinates in the XYZ format (in bohr)

    filename : path-like, optional. Default: None
        Path to a file containing coordinates in the XYZ format

    n_beads : list of int or None, optional. Default: None
        If not None, return an array of shape (n_dof, max(n_beads)) for the
        coordinates. Currently only constant number of beads is supported.
        If n_beads is None, the shape of the coordinates returned is (n_dof,).
        In all cases, the shape of the atom list and masses is (n_dof,). Can
        only be None if n_dof is too.

    n_dof : int or None, optional. Default: None
        If not None, specify the expected number of degree of freedom. Can only
        be None if n_beads is too.

    Return
    ------
    atom_symbols : (n_dof,) ndarray of strings
        Names of the atoms

    masses : (n_dof,) ndarray of floats
        Masses of the atoms for each degree of freedom.

    coord : (n_dof,) or (n_dof, max(n_beads)) ndarray of floats
        Coordinates of the atoms in bohr
    """

    conversion_factor = 1.0

    if input_string is not None:
        skip = 0
    elif filename is not None:
        skip = 2
        input_string = Path(filename).read_text()
        if "bohr" not in input_string.lower() and "a.u." not in input_string.lower():
            conversion_factor = angstrom_to_bohr
    else:
        raise ValueError("Either intput_string or filename should not be "
                         "None.")

    if (n_beads is None) != (n_dof is None):
        raise ValueError("n_beads and n_dof should either both be None or "
                         f"both be specified.")

    # The atom symbols are in the first column
    # NOTE: A new StringIO object must be created at np.loadtxt run as it is
    # consumed when read.
    atom_symbols = np.loadtxt(StringIO(input_string), skiprows=skip, usecols=0,
                              dtype=str)
    masses = np.array(list(map(atom_mass, atom_symbols)))

    try:
        # Columns 1 to 3 contains the coordinates
        coord = np.loadtxt(StringIO(input_string), skiprows=skip, usecols=(1, 2, 3))
    except Exception:
        raise ValueError("The XYZ data is not correctly formatted.\n"
                         f"XYZ data:\n{input_string}")

    if n_beads is not None and np.max(n_beads)*n_dof == len(coord)*3:
        max_n_beads = np.max(n_beads)
        if np.any(np.asarray(n_beads) != max_n_beads):
            raise NotImplementedError(
                "Variable number of beads not yet supported.")

        n_dof = 3*len(coord) // max_n_beads
        beads = np.zeros((n_dof, max_n_beads))

        for k in range(max_n_beads):
            b = coord[k::max_n_beads, :]
            beads[:, k] = np.hstack(b)

        atom_symbols = atom_symbols[::max_n_beads]
        masses = masses[::max_n_beads]
        coord = beads
    else:
        coord = np.hstack(coord)

        if n_dof is not None:
            coord = coord[:, None]

    if n_dof is not None and len(coord) != n_dof:
        raise ValueError(
            f"Number of degree of freedom given ({n_dof}) does not match "
            "the input.")


    # Masses and symbols are expected to be given for each dof so we have to
    # repeat each of them 3 times
    atom_symbols = np.hstack([(a, a, a) for a in atom_symbols])
    masses = np.hstack([(m, m, m) for m in masses])

    return atom_symbols, masses, conversion_factor*coord
