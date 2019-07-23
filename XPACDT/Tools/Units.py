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

"""Module to handly units."""


from scipy.constants import physical_constants, femto, pico, atto, centi, pi
# from mendeleev import element
from pyne.data import atomic_mass

# Boltzmann constant in atomic units
boltzmann = physical_constants['electron volt-hartree relationship'][0] * physical_constants['Boltzmann constant in eV/K'][0]
# Normal mode eigenvalues to wave numbers
nm_to_cm = centi/physical_constants['Bohr radius'][0]/physical_constants['inverse fine-structure constant'][0] / 2.0 / pi


def parse_time(time_string):
    """ Takes a string with a time and converts it to the numerical value
    in atomic units.
    """

    value, unit = time_string.split()

    if unit == "au":
        conversion = 1.0
    elif unit == "fs":
        conversion = femto / physical_constants['atomic unit of time'][0]
    elif unit == "ps":
        conversion = pico / physical_constants['atomic unit of time'][0]
    elif unit == "as":
        conversion = atto / physical_constants['atomic unit of time'][0]
    else:
        raise RuntimeError("Unit conversion failed:", unit)

    return float(value) * conversion


def atom_mass(symbol):
    """ Return the mass of an atom in atomic units.
    """

    conversion = physical_constants['atomic mass constant'][0] / physical_constants['atomic unit of mass'][0]
    return atomic_mass(symbol) * conversion
#    return atomic_mass(symbol) * conversion
#    return element(symbol).atomic_weight * conversion
