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

"""Module to handle units."""

import re
from scipy.constants import physical_constants, femto, pico, atto, centi, pi
import numpy as np


# Boltzmann constant in atomic units
boltzmann = physical_constants['electron volt-hartree relationship'][0] \
          * physical_constants['Boltzmann constant in eV/K'][0]

# Normal mode eigenvalues to wave numbers
nm_to_cm = centi/physical_constants['Bohr radius'][0] \
         / physical_constants['inverse fine-structure constant'][0] / 2.0 / pi

# Conversion of Hartree energy to eV and vice versa
hartree_to_eV = physical_constants['Hartree energy in eV'][0]
eV_to_hartree = 1.0 / hartree_to_eV

# Conversion of atomic unit of time to fs
au_to_fs = physical_constants['atomic unit of time'][0] / femto


def parse_time(time_string):
    """ Takes a string with a time and converts it to the numerical value
    in atomic units.

    Parameters
    ----------
    time_string : string
        Input string defining the time value and its unit.

    Returns
    -------
    float
        Given time value in atomic units.
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
        raise RuntimeError("\nXPACDT: Unit conversion failed:", unit)

    return float(value) * conversion


def atom_mass(symbol):
    """ Return the mass of an atom in atomic units. The mass for the most
    abundant isotope is returned if just the symbol is given. Other isotopes
    can be accessed by also giving the mass number. All elements with stable
    isotopes are implemented.

    Parameters
    ----------
    symbol : string
        The atom symbol to get the mass for. Please note that specific isotopes
        can be accessed as X-N, where X is the atom symbol and N is the mass
        number for that isotope. Specific other symbols are allowed like 'D'
        for deuterium and 'Mu' for Muonium.

    Return
    ------
    float :
        mass of the given element (or isotope) in atomic units
        (not atomic mass units).
    """

    if symbol == 'D':
        standard_symbol = 'H-2'
    else:
        symbol_groups = re.search(r"(\d*)-?([a-zA-Z]*)-?(\d*)", symbol)
        number = symbol_groups.group(1).strip() + symbol_groups.group(3).strip()
        atom = symbol_groups.group(2).strip()
        standard_symbol = atom
        if len(number) > 0:
            standard_symbol += "-" + number

    conversion = physical_constants['atomic mass constant'][0]\
        / physical_constants['atomic unit of mass'][0]
    try:
        return atoms.get(standard_symbol) * conversion
    except TypeError:
        raise KeyError("\nXPACDT: Error obtaining element weight: "
                       + standard_symbol)


def atom_symbol(mass):
    """ Return the symbol of an atom.  Any reference to isotopes is removed.

    Parameters
    ----------
    mass : float
        Mass of the atom in au.

    Returns
    -------
    symbol : string
        The symbol of the atom.
    """

    # au to amu
    conversion = physical_constants['atomic unit of mass'][0] \
        / physical_constants['atomic mass constant'][0]
    mass_amu = mass*conversion

    dist = np.Inf
    for s, m in atoms.items():
        if abs(m - mass_amu) < dist:
            symbol = s
            dist = abs(m - mass_amu)

    return_symbol = symbol.split("-")[0]
    if return_symbol == 'Mu' or return_symbol == 'e':
        return_symbol = 'H'
    return return_symbol


# from http://www.ciaaw.org/atomic-masses.htm
# Weight of the most common isotope given for just the atom symbol
# Non-Common Isotopes are labeled: X-N, where X is the atom symbol and N is the
# Isotope Mass Number. Deuterium would, e.g., be H-2.
# TODO: expand (see below)
# TODO: good source for Mu mass; http://goldbook.iupac.org/terms/view/M04069 ?
atoms = {'e': 0.0005485799093287202,
         'H': 1.0078250322, 'H-2': 2.0141017781, 'Mu': 0.113977478,
         'He': 4.0026032545, 'He-3': 3.016029322,
         'Li': 7.01600344, 'Li-6': 6.01512289,
         'Be': 9.0121831,
         'B': 11.009930517, 'B-10': 10.0129369,
         'C': 12.0, 'C-13': 13.003354835,
         'N': 14.003074004, 'N-15': 15.000108899,
         'O': 15.994914619, 'O-17': 16.999131757, 'O-18': 17.999159613,
         'F': 18.998403163,
         'Ne': 19.99244018, 'Ne-21': 20.9938467, 'Ne-22': 21.9913851,
         'Na': 22.98976928,
         'Mg': 23.98504170, 'Mg-25': 24.9858370, 'Mg-26': 25.9825930,
         'Al': 26.9815384,
         'Si': 27.976926535, 'Si-29': 28.976494665, 'Si-30': 29.9737701,
         'P': 30.973761998,
         'S': 31.972071174, 'S-33': 32.97145891, 'S-34': 33.9678670, 'S-36': 35.967081,
         'Cl': 34.9688527, 'Cl-37': 36.9659026,
         'Ar': 39.96238312, 'Ar-36': 35.9675451, 'Ar-38': 37.962732,
         'K': 38.96370649, 'K-40': 39.9639982, 'K-41': 40.96182526,
         'Ca:': 39.9625909, 'Ca-42': 41.958618, 'Ca-43': 42.958766, 'Ca-44': 43.955481, 'Ca-46': 45.95369, 'Ca-48': 47.9525229,
         'Sc': 44.955908,
         'Ti': 47.9479409, 'Ti-46': 45.952627, 'Ti-47': 46.9517577, 'Ti-49': 48.9478646, 'Ti-50': 49.9447858,
         'V': 50.943957, 'V-50': 49.947156,
         'Cr': 51.940505, 'Cr-50': 49.946041, 'Cr-53': 52.940647, 'Cr-54': 53.938878,
         'Mn': 54.938043,
         'Fe': 55.934936, 'Fe-54': 53.939608, 'Fe-57': 56.935392, 'Fe-58': 57.933274,
         'Co': 58.933194,
         'Ni': 57.935342, 'Ni-60': 59.930785, 'Ni-61': 60.931055, 'Ni-62': 61.928345, 'Ni-64': 63.927966,
         'Cu': 62.929597, 'Cu-65': 64.927790,
         'Zn': 63.929142, 'Zn-66': 65.926034, 'Zn-67': 66.927127, 'Zn-68': 67.924844, 'Zn-70': 69.92532,
         'Ga': 68.925573, 'Ga-71': 70.924702,
         'Ge': 73.92117776, 'Ge-70': 69.924249, 'Ge-72': 71.9220758, 'Ge-73': 72.9234590, 'Ge-76': 75.9214027,
         'As': 74.921595,
         'Se': 79.916522, 'Se-74': 73.9224759, 'Se-76': 75.9192137, 'Se-77': 76.9199141, 'Se-78': 77.917309, 'Se-82': 81.916699,
         'Br': 78.918338, 'Br-81': 80.916288,
         'Kr': 83.91149773, 'Kr-78': 77.920366, 'Kr-870': 79.916378, 'Kr-82': 81.91348115, 'Kr-83': 82.91412652, 'Kr-86': 85.91061063,
         'Rb': 84.91178974, 'Rb-87': 86.90918053,
         'Sr': 87.90561226, 'Sr-84': 83.913419, 'Sr-86': 85.90926073, 'Sr-87': 86.90887750,
         'Y': 88.90584,
         'Zr': 89.9046988, 'Zr-91': 90.9056402, 'Zr-92': 91.9050353, 'Zr-94': 93.906313, 'Zr-96': 95.9082776,
         'Nb': 92.90637,
         'Mo': 97.905404, 'Mo-92': 91.906807, 'Mo-94': 93.905084, 'Mo-95': 94.9058374, 'Mo-96': 95.9046748, 'Mo-97': 96.906017, 'Mo-100': 99.907468,
         'Ru': 101.904340, 'Ru-96': 95.907589, 'Ru-98': 97.90529, 'Ru-99': 98.905930, 'Ru-100': 99.904211, 'Ru-101': 100.905573, 'Ru-104': 103.90543,
         'Rh': 102.90549,
         'Pd': 105.903480, 'Pd-102': 101.905632, 'Pd-104': 103.904030, 'Pd-105': 104.905079, 'Pd-108': 107.903892, 'Pd-110': 109.905173,
         'Ag': 106.90509, 'Ag-109': 108.904756,
         'Cd': 113.903365, 'Cd-106': 105.906460, 'Cd-108': 107.904184, 'Cd-110': 109.903008, 'Cd-111': 110.904184, 'Cd-112': 111.902764, 'Cd-113': 112.904408, 'Cd-116': 115.904763,
         'In': 114.90387877, 'In-113': 112.904060,
         'Sn': 119.902202, 'Sn-112': 111.904825, 'Sn-114': 113.9027801, 'Sn-115': 114.9033447, 'Sn-116': 115.9017428, 'Sn-117': 116.902954, 'Sn-118': 117.901607, 'Sn-119': 118.903311, 'Sn-122': 121.90344, 'Sn-124': 123.905277,
         'Sb': 120.90381, 'Sb-123': 122.90421, 
         'Te':129.90622275, 'Te-120': 119.90406 , 'Te-122':121.90304, 'Te-123':122.90427, 'Te-124':123.90282, 'Te-125':124.90443, 'Te-126':125.90331, 'Te-128':127.904461,
         'I': 126.90447,
         'Xe-132': 131.90415509, 'Xe-124': 123.90589, 'Xe-126': 125.90430, 'Xe-128': 127.903531, 'Xe-129': 128.90478086, 'Xe-130': 129.90350935, 'Xe-131': 130.90508414, 'Xe-134': 133.90539303, 'Xe-136': 135.90721448, 
         'Cs': 132.90545196,
         'Ba': 137.905247, 'Ba-130': 129.90632, 'Ba-132': 131.905061, 'Ba-134':133.904508 , 'Ba-135': 134.905689, 'Ba-136': 135.904576, 'Ba-137':136.905827, 
         'La': 138.90636, 'La-138': 137.90712,
         'Ce': 139.90545, 'Ce-136': 135.907129,  'Ce-138': 137.90599,  'Ce-142': 141.90925,
         'Pr': 140.90766,
         'Nd': 141.90773, 'Nd-143': 142.90982, 'Nd-144': 143.91009, 'Nd-145': 144.91258, 'Nd-146': 145.91312, 'Nd-148': 147.91690, 'Nd-150': 149.920902,
         'Sm': 151.919739, 'Sm-144': 143.91201, 'Sm-147': 146.91490, 'Sm-148': 147.91483, 'Sm-149': 148.917191, 'Sm-150': 149.917282, 'Sm-154': 153.92222,
         'Eu': 152.921237, 'Eu-151': 150.919857,
         'Gd': 157.924112, 'Gd-152': 151.919799, 'Gd-154': 153.920873, 'Gd-155': 154.922630, 'Gd-156': 155.922131, 'Gd-157': 156.923968, 'Gd-160': 159.927062,
         'Td': 158.925354,
         'Dy': 63.929181, 'Dy-156': 155.924284,  'Dy-158': 157.92441,  'Dy-160': 159.925203,  'Dy-161': 160.926939,  'Dy-162': 161.926804,  'Dy-163': 162.928737, 
         'Ho': 164.930328, 
         'Er': 167.932376, 'Er-162': 161.928787,  'Er-164': 163.929207,  'Er-166': 165.930299,  'Er-167': 166.932054,  'Er-170': 169.93547, 
         'Tm': 168.934218, 
         'Yb': 173.93886755, 'Yb-168': 167.933889, 'Yb-170': 169.93476725, 'Yb-171': 170.93633152, 'Yb-172': 171.93638666, 'Yb-173': 172.93821622, 'Yb-176': 175.9425747,
         'Lu': 175.942692, 'Lu-175': 174.940777, 
         'Hf': 179.94656, 'Hf-174': 173.94005,  'Hf-176': 175.94141,  'Hf-177': 176.94323,  'Hf-178': 177.94371,  'Hf-179': 178.94583, 
         'Ta': 180.94800, 'Ta-180': 179.94747, 
         'W': 183.950933, 'W-180': 179.94671,  'W-182': 181.948206,  'W-183': 182.950224,  'W-186': 185.954365, 
         'Re': 186.955752, 'Re-185': 184.952958,
         'Os': 191.96148,  'Os-184': 183.952493,  'Os-186': 185.953838,  'Os-187': 186.955750,  'Os-188': 187.955837,  'Os-189': 188.958146,  'Os-190': 189.958446, 
         'Ir': 192.962924, 'Ir-191': 190.960591,
         'Pt': 194.964794, 'Pt-190': 189.959950, 'Pt-192': 191.96104, 'Pt-194': 193.962683, 'Pt-196': 195.964955, 'Pt-198': 197.96790,
         'Au': 196.966570,
         'Hg': 201.970644, 'Hg-196': 195.96583, 'Hg-198': 197.966769, 'Hg-199': 198.968281, 'Hg-200': 199.968327, 'Hg-201': 200.970303, 'Hg-204': 203.973494,
         'Tl': 204.974427, 'Tl-203': 202.972344,
         'Pb': 207.976652, 'Pb-204': 203.973043, 'Pb-206': 205.974465, 'Pb-207': 206.975897,
         'Bi': 208.98040,
         'Th': 232.03805, 'Th-230': 230.033132,
         'Pa': 231.03588,
         'U': 238.05079,'U-234': 234.040950, 'U-235': 235.043928  
}

# from http://www.ciaaw.org/atomic-masses.htm
#1 	H 	hydrogen 	1 	  1.007 825 0322(6)
#2 	  2.014 101 7781(8)
#2 	He 	helium 	3 	  3.016 029 322(2)
#4 	  4.002 603 2545(4)
#3 	Li 	lithium 	6 	  6.015 122 89(1)
#7 	  7.016 003 44(3)
#4 	Be 	beryllium 	9 	  9.012 1831(5)
#5 	B 	boron 	10 	 10.012 9369(1)
#11 	 11.009 305 17(8)
#6 	C 	carbon 	12 	 12(exact)
#13 	 13.003 354 835(2)
#7 	N 	nitrogen 	14 	 14.003 074 004(2)
#15 	 15.000 108 899(4)
#8 	O 	oxygen 	16 	 15.994 914 619(1)
#17 	 16.999 131 757(5)
#18 	 17.999 159 613(5)
#9 	F 	fluorine 	19 	 18.998 403 163(6)
#10 	Ne 	neon 	20 	 19.992 440 18(1)
#21 	 20.993 8467(3)
#22 	 21.991 3851(1)
#11 	Na 	sodium 	23 	 22.989 769 28(2)
#12 	Mg 	magnesium 	24 	 23.985 041 70(9)
#25 	 24.985 8370(3)
#26 	 25.982 5930(2)
#13 	Al 	aluminium 	27 	 26.981 5384(3)
#14 	Si 	silicon 	28 	 27.976 926 535(3)
#29 	 28.976 494 665(4)
#30 	 29.973 7701(2)
#15 	P 	phosphorous 	31 	 30.973 761 998(5)
#16 	S 	sulfur 	32 	 31.972 071 174(9)
#33 	 32.971 458 91(1)
#34 	 33.967 8670(3)
#36 	 35.967 081(2)
#17 	Cl 	chlorine 	35 	 34.968 8527(3)
#37 	 36.965 9026(4)
#18 	Ar 	argon 	36 	 35.967 5451(2)
#38 	 37.962 732(2)
#40 	 39.962 383 12(2)
#19 	K 	potassium 	39 	 38.963 706 49(3)
#40* 	 39.963 9982(4)
#41 	 40.961 825 26(3)
#20 	Ca 	calcium 	40 	 39.962 5909(2)
#42 	 41.958 618(1)
#43 	 42.958 766(2)
#44 	 43.955 481(2)
#46 	 45.953 69(2)
#48* 	 47.952 5229(6)
#21 	Sc 	scandium 	45 	 44.955 908(5)
#22 	Ti 	titanium 	46 	 45.952 627(1)
#47 	 46.951 7577(8)
#48 	 47.947 9409(8)
#49 	 48.947 8646(8)
#50 	 49.944 7858(8)
#23 	V 	vanadium 	50* 	 49.947 156(3)
#51 	 50.943 957(3)
#24 	Cr 	chromium 	50 	 49.946 041(3)
#52 	 51.940 505(3)
#53 	 52.940 647(3)
#54 	 53.938 878(3)
#25 	Mn 	manganese 	55 	 54.938 043(2)
#26 	Fe 	iron 	54 	 53.939 608(3)
#56 	 55.934 936(2)
#57 	 56.935 392(2)
#58 	 57.933 274(3)
#27 	Co 	cobalt 	59 	 58.933 194(3)
#28 	Ni 	nickel 	58 	 57.935 342(3)
#60 	 59.930 785(3)
#61 	 60.931 055(3)
#62 	 61.928 345(3)
#64 	 63.927 966(3)
#29 	Cu 	copper 	63 	 62.929 597(3)
#65 	 64.927 790(5)
#30 	Zn 	zinc 	64 	 63.929 142(5)
#66 	 65.926 034(5)
#67 	 66.927 127(5)
#68 	 67.924 844(5)
#70 	 69.925 32(2)
#31 	Ga 	gallium 	69 	 68.925 573(8)
#71 	 70.924 702(6)
#32 	Ge 	germanium 	70 	 69.924 249(6)
#72 	 71.922 0758(5)
#73 	 72.923 4590(4)
#74 	 73.921 177 76(8)
#76* 	 75.921 4027(1)
#33 	As 	arsenic 	75 	 74.921 595(6)
#34 	Se 	selenium 	74 	 73.922 4759(1)
#76 	 75.919 2137(1)
#77 	 76.919 9141(5)
#78 	 77.917 309(1)
#80 	 79.916 522(6)
#82* 	 81.916 699(3)
#35 	Br 	bromine 	79 	 78.918 338(7)
#81 	 80.916 288(6)
#36 	Kr 	krypton 	78* 	 77.920 366(2)
#80 	 79.916 378(5)
#82 	 81.913 481 15(4)
#83 	 82.914 126 52(6)
#84 	 83.911 497 73(3)
#86 	 85.910 610 63(3)
#37 	Rb 	rubidium 	85 	 84.911 789 74(3)
#87* 	 86.909 180 53(4)
#38 	Sr 	strontium 	84 	 83.913 419(8)
#86 	 85.909 260 73(4)
#87 	 86.908 877 50(3)
#88 	 87.905 612 26(4)
#39 	Y 	yttrium 	89 	 88.905 84(1)
#40 	Zr 	zirconium 	90 	 89.904 6988(8)
#91 	 90.905 6402(7)
#92 	 91.905 0353(7)
#94 	 93.906 313(1)
#96* 	 95.908 2776(8)
#41 	Nb 	niobium 	93 	 92.906 37(1)
#42 	Mo 	molybdenum 	92 	 91.906 807(1)
#94 	 93.905 084(1)
#95 	 94.905 8374(8)
#96 	 95.904 6748(8)
#97 	 96.906 017(1)
#98 	 97.905 404(1)
#100* 	 99.907 468(2)
#44 	Ru 	ruthenium 	96 	 95.907 589(1)
#98 	 97.905 29(5)
#99 	 98.905 930(3)
#100 	 99.904 211(3)
#101 	100.905 573(3)
#102 	101.904 340(3)
#104 	103.905 43(2)
#45 	Rh 	rhodium 	103 	102.905 49(2)
#46 	Pd 	palladium 	102 	101.905 632(4)
#104 	103.904 030(9)
#105 	104.905 079(8)
#106 	105.903 480(8)
#108 	107.903 892(8)
#110 	109.905 173(5)
#47 	Ag 	silver 	107 	106.905 09(2)
#109 	108.904 756(9)
#48 	Cd 	cadmium 	106 	105.906 460(8)
#108 	107.904 184(8)
#110 	109.903 008(3)
#111 	110.904 184(3)
#112 	111.902 764(2)
#113* 	112.904 408(2)
#114 	113.903 365(2)
#116* 	115.904 763(1)
#49 	In 	indium 	113 	112.904 060(2)
#115* 	114.903 878 77(8)
#50 	Sn 	tin 	112 	111.904 825(2)
#114 	113.902 7801(2)
#115 	114.903 3447(1)
#116 	115.901 7428(6)
#117 	116.902 954(3)
#118 	117.901 607(3)
#119 	118.903 311(5)
#120 	119.902 202(6)
#122 	121.903 44(2)
#124 	123.905 277(7)
#51 	Sb 	antimony 	121 	120.903 81(2)
#123 	122.904 21(1)
#52 	Te 	tellurium 	120 	119.904 06(2)
#122 	121.903 04(1)
#123 	122.904 27(1)
#124 	123.902 82(1)
#125 	124.904 43(1)
#126 	125.903 31(1)
#128* 	127.904 461(6)
#130* 	129.906 222 75(8)
#53 	I 	iodine 	127 	126.904 47(3)
#54 	Xe 	xenon 	124 	123.905 89(1)
#126 	125.904 30(3)
#128 	127.903 531(7)
#129 	128.904 780 86(4)
#130 	129.903 509 35(6)
#131 	130.905 084 14(6)
#132 	131.904 155 09(4)
#134 	133.905 393 03(6)
#136* 	135.907 214 48(5)
#55 	Cs 	caesium 	133 	132.905 451 96(6)
#56 	Ba 	barium 	130* 	129.906 32(2)
#132 	131.905 061(7)
#134 	133.904 508(2)
#135 	134.905 689(2)
#136 	135.904 576(2)
#137 	136.905 827(2)
#138 	137.905 247(2)
#57 	La 	lanthanum 	138* 	137.907 12(2)
#139 	138.906 36(2)
#58 	Ce 	cerium 	136 	135.907 129(3)
#138 	137.905 99(3)
#140 	139.905 45(1)
#142 	141.909 25(2)
#59 	Pr 	praseodymium 	141 	140.907 66(1)
#60 	Nd 	neodymium 	142 	141.907 73(1)
#143 	142.909 82(1)
#144* 	143.910 09(1)
#145 	144.912 58(1)
#146 	145.913 12(1)
#148 	147.916 90(2)
#150* 	149.920 902(9)
#62 	Sm 	samarium 	144 	143.912 01(1)
#147* 	146.914 90(1)
#148* 	147.914 83(1)
#149 	148.917 191(9)
#150 	149.917 282(9)
#152 	151.919 739(8)
#154 	153.922 22(1)
#63 	Eu 	europium 	151* 	150.919 857(9)
#153 	152.921 237(9)
#64 	Gd 	gadolinium 	152 	151.919 799(8)
#154 	153.920 873(8)
#155 	154.922 630(8)
#156 	155.922 131(8)
#157 	156.923 968(8)
#158 	157.924 112(8)
#160 	159.927 062(9)
#65 	Tb 	terbium 	159 	158.925 354(8)
#66 	Dy 	dysprosium 	156 	155.924 284(8)
#158 	157.924 41(2)
#160 	159.925 203(5)
#161 	160.926 939(5)
#162 	161.926 804(5)
#163 	162.928 737(5)
#164 	163.929 181(5)
#67 	Ho 	holmium 	165 	164.930 328(7)
#68 	Er 	erbium 	162 	161.928 787(6)
#164 	163.929 207(5)
#166 	165.930 299(8)
#167 	166.932 054(8)
#168 	167.932 376(8)
#170 	169.935 47(1)
#69 	Tm 	thulium 	169 	168.934 218(6)
#70 	Yb 	ytterbium 	168 	167.933 889(8)
#170 	169.934 767 25(7)
#171 	170.936 331 52(9)
#172 	171.936 386 66(9)
#173 	172.938 216 22(8)
#174 	173.938 867 55(8)
#176 	175.942 5747(1)
#71 	Lu 	lutetium 	175 	174.940 777(8)
#176* 	175.942 692(8)
#72 	Hf 	hafnium 	174* 	173.940 05(2)
#176 	175.941 41(1)
#177 	176.943 23(1)
#178 	177.943 71(1)
#179 	178.945 83(1)
#180 	179.946 56(1)
#73 	Ta 	tantalum 	180* 	179.947 47(2)
#181 	180.948 00(1)
#74 	W 	tungsten 	180* 	179.946 71(1)
#182 	181.948 206(5)
#183 	182.950 224(5)
#184 	183.950 933(5)
#186 	185.954 365(8)
#75 	Re 	rhenium 	185 	184.952 958(6)
#187* 	186.955 752(5)
#76 	Os 	osmium 	184* 	183.952 493(6)
#186* 	185.953 838(5)
#187 	186.955 750(5)
#188 	187.955 837(5)
#189 	188.958 146(5)
#190 	189.958 446(5)
#192 	191.961 48(2)
#77 	Ir 	iridium 	191 	190.960 591(9)
#193 	192.962 924(9)
#78 	Pt 	platinum 	190* 	189.959 950(5)
#192 	191.961 04(2)
#194 	193.962 683(3)
#195 	194.964 794(3)
#196 	195.964 955(3)
#198 	197.967 90(2)
#79 	Au 	gold	197 	196.966 570(4)
#80 	Hg 	mercury 	196 	195.965 83(2)
#198 	197.966 769(3)
#199 	198.968 281(4)
#200 	199.968 327(4)
#201 	200.970 303(5)
#202 	201.970 644(5)
#204 	203.973 494(3)
#81 	Tl 	thallium 	203 	202.972 344(8)
#205 	204.974 427(8)
#82 	Pb 	lead 	204 	203.973 043(8)
#206 	205.974 465(8)
#207 	206.975 897(8)
#208 	207.976 652(8)
#83 	Bi 	bismuth 	209* 	208.980 40(1)
#90 	Th 	thorium 	230* 	230.033 132(8)
#232* 	232.038 05(1)
#91 	Pa 	protactinium 	231* 	231.035 88(1)
#92 	U 	uranium 	234* 	234.040 950(8)
#235* 	235.043 928(8)
#238* 	238.050 79(1) 
