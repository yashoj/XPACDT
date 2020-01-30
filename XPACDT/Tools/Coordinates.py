import numpy as np

from io import StringIO
from pathlib import Path

from XPACDT.Tools.Units import atom_mass, angstrom_to_bohr


def parse_mass_value(string):
    """
    Parse coordinate input that has a mass and a coordinate value per line.
    The format has to be as follows. The first entry per line gives the
    mass for this degree of freedom (in au, not amu). The next entry is
    the coordinate values for this degree of freedom. All bead values have
    to be given in one line.

    # TODO Correct the docstring
    The results are stored in self.masses, which are the au (not amu!)
    masses for each atom, and in self.coordinates, which is a two-d
    numpy array of floats.

    Parameters
    ----------
    string : str
        String representation of the input.
    """
    d = StringIO(string)
    try:
        # TODO: This needs to be replaced if different number of beads
        # per DOF can be used!
        mc = np.loadtxt(d, ndmin=2)
    except ValueError:
        raise ValueError("The mass-value data is not correctly formatted."
                         f"mass-value data:\n{string}")

    masses = mc[:, 0]
    coord = mc[:, 1:]
    return masses, coord


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


def parse_xyz(string=None, filename=None):
    """
    Parse a XYZ formatted string (passed with the `values` keyword), or file
    (which name or path is given by the `filename` keyword).

    Strings are expected to only exactly contain the coordinates, and to use
    bohr (a.u.) as unit.

    Files are expected to have 2 special lines at the start. The first
    containing the number of atoms (this line is ignored here) and the second
    being a comment line. If the second line contains the words "bohr" or
    "a.u." this is used as unit, otherwise the unit used is Angstrom. In any
    case the output is in bohr.

    This follows the Molcas convention for XYZ files.

    Parameters
    ----------
    string : str
        A string containing coordinates in the XYZ format (in bohr)

    filename : path-like
        Path to a file containing coordinates in the XYZ format

    Return
    ------
    atom_symbols : (n_atoms,) array of strings
        Names of the atoms

    masses : ...

    coord : (3*n_atoms,) ndarray of floats
        Coordinates of the atoms in bohr
    """

    conversion_factor = 1.0

    if string is not None:
        skip = 0
        filename = StringIO(string)
    elif filename is not None:
        skip = 2
        string = Path(filename).read_text().lower()
        if "bohr" not in string and "a.u." not in string:
            conversion_factor = angstrom_to_bohr
    else:
        raise ValueError("Either values or filename should not be None.")

    # The atom symbols are in the first column
    atom_symbols = np.loadtxt(filename, skiprows=skip, usecols=0, dtype=str)
    try:
        # Columns 1 to 3 contains the coordinates
        coord = np.loadtxt(filename, skiprows=skip, usecols=(1, 2, 3))
    except Exception:
        raise ValueError("The XYZ data is not correctly formatted.\n"
                         f"XYZ data:\n{string}")

    coord = conversion_factor*np.hstack(coord)
    masses = np.array(list(map(atom_mass, atom_symbols)))

    return atom_symbols, masses, coord
