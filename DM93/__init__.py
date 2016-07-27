#-------------------------- LICENCE BEGIN ---------------------------
# This file is part of DaleyMenard93.
#
# DaleyMenard93 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DaleyMenard93 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DaleyMenard93.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors - Martin Deshaies-Jacques, Richard Menard
#
# Copyright 2016 - Air Quality Research Division, Environnement Canada
#-------------------------- LICENCE END -----------------------------


''' Daley and Menard 1993 Kalman Filter 1D lab '''

from gridCls import Grid
from DM93Lib import *
from corrModelLib import Covariance, Uncorrelated,  Foar, Soar, Gaussian

