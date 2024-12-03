# a list of helper classes and functions which makes working with "atoms" a bit easier
import math
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R


# introduce some classes to work with atom-like objects
# I think that these classes are similar to the ones from any other chemistry software
# e.g. Tinker uses similar notation, as well as elements package
class AtomType:
    def __init__(self, chemical_symbol, weight, color, size):
        self.name = chemical_symbol
        self.weight = weight
        self.color = color
        self.size = size


class Atom:
    def __init__(self, atomtype: AtomType, x, y, z):
        self.atomtype = atomtype
        self.x = x
        self.y = y
        self.z = z

    # allows print() to correctly print out a list of objects
    def __repr__(self):
        return str(self)

    # if an object is printed, what will be displayed is a string that has the following format:
    # atom type, x, y, z
    def __str__(self):
        return f"{self.atomtype.name} {self.x} {self.y} {self.z}"

    # generates a single line (string), i.e. atom entry, for .xyz file format
    def xyz(self):
        return ' '.join([self.atomtype.name, str(self.x), str(self.y), str(self.z)])

    # returns a numpy vector of length 3 that contains (x, y, z) coordinates of the atom
    def getxyz(self):
        return np.array([self.x, self.y, self.z])

    # assigns new coordinates to the atom from a given vector v of length 3
    def assxyz(self, v):
        self.x = v[0]
        self.y = v[1]
        self.z = v[2]


# home-made small periodic table
PT = dict(hydrogen=AtomType("H", 1.00784, "w", 10),
          nitrogen=AtomType("N", 14.0067, "b", 50),
          lead=AtomType("Pb", 207.2, "k", 100),
          carbon=AtomType("C", 12.011, "r", 50),
          iodide=AtomType("I", 126.90447, "m", 75),
          tin=AtomType("Sn", 118.710, "g", 75),
          copper=AtomType("Cu", 63.546, "b", 60),
          oxygen=AtomType("O", 15.999, "r", 50))


# get an atom_type based on chemical symbol (!!! must be one of the atom_types available in PT )
# case-sensitive
def assign_atom_type(chemical_symbol):
    return [j for i, j in PT.items() if j.name == chemical_symbol][0]


# reads atomic data from an .xyz file while dropping first two lines
# it stores the data in "Atom" class objects
def read_xyz(address):
    structure = []
    with open(address, "r") as f:
        read_data = f.read()
    l1 = read_data.splitlines()
    del l1[0:2]
    for line in l1:
        l2 = line.split()
        structure.append(Atom(assign_atom_type(l2[0]), float(l2[1]), float(l2[2]), float(l2[3])))
    return structure


# creates a new_atom of the same atom_type shifted from an input atom by a vector (ref_x, ref_y, ref_z)
def atom_shift(atom, ref_x, ref_y, ref_z):
    new_atom = copy.deepcopy(atom)
    new_atom.x = atom.x + ref_x
    new_atom.y = atom.y + ref_y
    new_atom.z = atom.z + ref_z
    return new_atom


# calculates a geometrical distance between 2 atoms in 3D
def atom_distance(atom1, atom2):
    return math.sqrt((atom1.x - atom2.x)**2 +
                     (atom1.y - atom2.y)**2 +
                     (atom1.z - atom2.z)**2)


# creates a copy of an atom by rotating it around a specified axis by a "deg" angle in degrees
# the original atom remains unchanged
def wrotor(atom, axis, deg):
    a = copy.deepcopy(atom)
    r = R.from_euler(axis, deg, degrees=True)
    rez = r.apply([atom.x, atom.y, atom.z])
    a.x = rez[0]
    a.y = rez[1]
    a.z = rez[2]
    return a


# converts unit cell dimensions into 3 basis vectors in the form of unit cell tensor.
# output format is a list of strings
# for unit cell tensor, see: https://en.wikipedia.org/wiki/Fractional_coordinates
def uc_dimensions_to_vectors_string(a, b, c, alpha, betta, gamma):
    cx = c * math.cos(math.radians(betta))
    cy = (c * (math.cos(math.radians(alpha)) - math.cos(math.radians(betta)) * math.cos(math.radians(gamma)))
          / math.sin(math.radians(gamma)))
    cz = math.sqrt(c ** 2 - cx ** 2 - cy ** 2)
    return [f'{a} 0.0 0.0',
            f'{b * math.cos(math.radians(gamma))} {b * math.sin(math.radians(gamma))} 0.0',
            f'{cx} {cy} {cz}']


# same as previous function, but output format is numpy array
def uc_dimensions_to_vectors_varray(a, b, c, alpha, betta, gamma):
    cx = c * math.cos(math.radians(betta))
    cy = (c * (math.cos(math.radians(alpha)) - math.cos(math.radians(betta)) * math.cos(math.radians(gamma)))
          / math.sin(math.radians(gamma)))
    cz = math.sqrt(c ** 2 - cx ** 2 - cy ** 2)
    return np.array([[a, 0.0, 0.0],
                     [b * math.cos(math.radians(gamma)), b * math.sin(math.radians(gamma)), 0.0],
                     [cx, cy, cz]])


# eliminates duplicates in a list while retaining the order
def f12(seq):
    # Raymond Hettinger
    # https://twitter.com/raymondh/status/944125570534621185
    return list(dict.fromkeys(seq))


# calculates a normal vector to a plane defined by 3 objects of "Atom" type
def plane_norm(atom1, atom2, atom3):
    # coordinates of 3 atoms
    p1 = np.array([atom1.x, atom1.y, atom1.z])
    p2 = np.array([atom2.x, atom2.y, atom2.z])
    p3 = np.array([atom3.x, atom3.y, atom3.z])

    # form 2 in-plane vectors
    v1 = p3 - p1
    v2 = p2 - p1

    return np.cross(v1, v2)


# calculates angles of rotation around x-axis and z-axis for a given vector such that if rotated by these angles
# along corresponding axis, a vector will become parallel to y-axis
def to_y(x, y, z):
    rx = -math.atan2(z, y)                  # or +math.atan2(z,-y)
    y2 = y*math.cos(rx) - z * math.sin(rx)  # -> (x,y2,0)
    return rx, math.atan2(x, y2)


# calculates an angle between two vectors in 3D
# returns angle in radians
def angle_bw_2_vectors(np_vector1, np_vector2):
    return np.arccos(np.dot(np_vector1/np.linalg.norm(np_vector1), np_vector2/np.linalg.norm(np_vector2)))


# rotates a vector v around given axis, represented by a vector ax=[x, y, z], by a given angle theta
def universal_wrotor(v, axis, theta):
    u = axis / np.linalg.norm(axis)
    one_m_cos = 1 - math.cos(theta)
    sn = math.sin(theta)
    cs = math.cos(theta)
    rr = [[cs + u[0]**2 * one_m_cos, u[0]*u[1]*one_m_cos - u[2]*sn, u[0]*u[2]*one_m_cos + u[1]*sn],
          [u[1]*u[0]*one_m_cos + u[2]*sn, cs + u[1]**2 * one_m_cos, u[1]*u[2]*one_m_cos - u[0]*sn],
          [u[2]*u[0]*one_m_cos - u[1]*sn, u[2]*u[1]*one_m_cos + u[0]*sn, cs + u[2]**2 * one_m_cos]]
    return np.matmul(rr, v)
