import math
import matplotlib.pyplot as plt
import numpy as np


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Vector(Point):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"

    def normalize(self):
        abs = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        self.x = self.x / abs
        self.y = self.y / abs
        self.z = self.z / abs

    @staticmethod
    def cross_product(a, b):
        return Vector(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x)

    @staticmethod
    def create_vector_verticies(p1, p2):
        return Vector(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)


class Facet:
    def __init__(self, v1, v2, v3):
        self.verticies = [v1, v2, v3]
        vector1 = Vector.create_vector_verticies(v1, v2)
        vector2 = Vector.create_vector_verticies(v1, v3)
        normal = Vector.cross_product(vector1, vector2)
        normal.normalize()
        self.normal = normal


# a1 = Point(2, 3, 4)
# b1 = Point(5, 3, 6)
# c1 = Point(6, 7, 9)
#
# a2 = Point(1, 1, 1)
# b2 = Point(1, 2, 1)
# c2 = Point(2, 1, 1)
#
# polygon1 = Facet(a1, b1, c1)
# polygon2 = Facet(a2, b2, c2)
# print(polygon1.normal)
# print(polygon2.normal)


phi = math.pi + math.pi / 4
tg = math.tan(phi)


def rad_to_deg(radian):
    return (radian * 180) / math.pi


def deg_to_rad(degree):
    return (degree * math.pi) / 180


def theory_triangle_BSP(step=1, angle_const_name="phi", const_angle=0, a=1, b=1, lambda_=0.03):
    rcs = []
    for angle in range(360):
        if angle_const_name == "phi":
            rcs.append(theory_triangle_RCS(const_angle, angle, a, b, lambda_))
        else:
            rcs.append(theory_triangle_RCS(angle, const_angle, a, b, lambda_))


def arccos(x):
    if x > 1:
        x = 1
    elif x < -1:
        x = -1
    return math.acos(x)


def theory_triangle_RCS(phi, theta, a=0.1, b=0.1, lambda_=0.03):
    phiT = math.atan(math.sin(deg_to_rad(phi)) / math.cos(deg_to_rad(theta)))
    thetaT = arccos(
        math.cos(math.atan(math.sin(deg_to_rad(phi)) / math.cos(deg_to_rad(theta)))) * math.tan(deg_to_rad(theta)))
    sigma_m = (4 * math.pi * (a ** 2) * b ** 2) / (lambda_ ** 2)
    k = (2 * math.pi) / lambda_
    first_expr = (sigma_m * (math.cos(phiT) * math.cos(thetaT)) ** 2) / (
            (k * a * math.sin(phiT) * math.cos(thetaT)) ** 2 - (
            k * b * math.sin(thetaT)) ** 2) ** 2

    second_expr = ((math.sin(k * a * math.sin(phiT) * math.cos(thetaT))) ** 2 - (
        math.sin(k * b * math.cos(thetaT))) ** 2) ** 2

    third_expr = (k * b * math.sin(thetaT)) ** 2 * (
            np.sinc(2 * k * a * math.sin(phiT) * math.cos(thetaT) / math.pi) - np.sinc(
        2 * k * b * math.sin(thetaT) / math.pi)) ** 2

    rcs = first_expr * (second_expr + third_expr)
    return rcs


rcs_arr = []
angles_rad = []
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
angles = range(360)
for angle in angles:
    calc_rcs = theory_triangle_RCS(phi=angle, theta=math.pi / 2)
    if calc_rcs == 0:
        rcs_arr.append(0)
    else:
        rcs_arr.append(calc_rcs)
    angles_rad.append(deg_to_rad(angle))
print(len(rcs_arr))
plt.polar(angles_rad, rcs_arr)
plt.show()
