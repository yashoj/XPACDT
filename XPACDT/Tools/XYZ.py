from io import StringIO

import numpy as np


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


def parse_xyz(values=None, filename=None):
    """
    Parse a XYZ formatted string (passed with the `values` keyword), or file
    (which name or path is given by the `filename` keyword).

    Strings are expected to only exactly contain the coordinates, and to use
    bohr (a.u.) as unit.

    Files are expected to have 2 useless lines at the start, and to use
    Angstrom as unit.

    Parameters
    ----------
    values : string
        A string containing coordinates in the XYZ format

    filename : path-like
        Path to a file containing coordinates in the XYZ format

    Return
    ------
    atoms : array of strings
        Names of the atoms

    coord : (n_atoms, 3) ndarray of floats
        Coordinates of the atoms
    """
    # TODO replace InputFile._parse_xyz
    # TODO better checks for valid input
    # TODO read comment line to determine units when loading a file ?
    # TODO unit transform

    skip = 2

    if values is not None:
        skip = 0
        filename = StringIO(values)

    atoms = np.loadtxt(filename, skiprows=skip, usecols=0, dtype=str)
    coord = np.loadtxt(filename, skiprows=skip, usecols=(1, 2, 3))

    return atoms, coord
