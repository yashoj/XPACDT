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

""" Module to perform bootstrapping analysis."""

import numpy as np


def bootstrap(data, function, n_bootstrap=1000, is_2D=False):
    """Performs a basic bootstrapping analysis to obtain the standard error
    of a certain property averaged over the ensemble of trajectories.
    See: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

    Parameters
    ----------
    data : (nsamples, nvalues) ndarray of floats
        The data to be used in the bootstrapping.
    function : function
        The function for which the bootstrapping should be calculate, e.g..
        np.mean, np.percentile, histogram, ...
        The function is assumed to return 'nfunc' values. A histogram for
        example will return 'nbins' values.
    n_bootstrap : int, optional
        Number of bootstrapping resamples to take. Default: 1000
    is_2D : bool, optional
        Set if the function is a 2d histogram. Then no resampling is possible
        currently.

    Returns:
        m : (nfunc) ndarray of floats
            The mean value of the function over the sample
        s : (nfunc) ndarray of floats
            The standard error as obtained from the bootstrapping for the
            values in m.
    """

    nsamples = len(data)
    if n_bootstrap == 1 or is_2D:
        values = [function(data)]
        n_bootstrap = 1
    else:
        values = [function(data[np.random.randint(nsamples, size=nsamples)])
                  for i in range(n_bootstrap)]

    m = np.mean(np.array(values).reshape(n_bootstrap, -1), axis=0)
    if n_bootstrap == 1:
        s = np.zeros_like(m)
    else:
        s = np.std(np.array(values).reshape(n_bootstrap, -1), axis=0)

    return m, s
