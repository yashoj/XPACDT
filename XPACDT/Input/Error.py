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

"""Module defining the errors related to the input file.
"""


class XPACDTInputError(Exception):
    """
    Exception raise whenever an input file can not be parsed correctly, either
    due to wrong formatting, missing key or inconsistency.

    Parameters
    ----------
    msg : str, optional. Default: "Missing key".
        The cause of the error.

    section : str, optional. Default: None.
        The section of the input file that is faulty.

    key : str, optional. Default: None.
        The key associated with a faulty value.

    given : str, optional. Default: None.
        The value given in the input file.

    caused_by : Exception, optional. Default: None.
        An exception that caused the current failure.
    """
    def __init__(self, msg="Missing key",
                 section=None, key=None, given=None, caused_by=None):
        if section is not None:
            msg += f"\n[Section] {section}"

        if key is not None:
            msg += f"\n    [key] {key}"

        if given is not None:
            msg += f"\n  [given] {given}"

        if caused_by is not None:
            msg += ("\nThis error was caused by the following error:\n"
                    f"{type(caused_by)}: {caused_by}")

        super().__init__(msg)