import os
import sys
from copy import deepcopy

import numpy as np
import math

arr = lambda *args, **kwargs: np.array(*args, **kwargs)

class Object:
    """An object in the world
    """
    def __init__(self, name, size : float = 0,
                r: np.array = None,
                v : np.array = arr([0, 0, 0], dtype = float),
                a: np.array = arr([0, 0, 0], dtype = float)) -> None:
        self.name = name
        self.size = size

        self.r = r  # position
        self.v = v  # velocity
        self.a = a  # acceleration

        # -- get its bbox relative to the center
        oft = [ [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
               [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]]
        self.bbox = np.array(oft, dtype = float) * size / 2

    def __str__(self) -> str:
        s = self.name
        s += ", size: " + str(self.size)
        s += ", r: " + str(self.r)
        s += ", v: " + str(self.v)
        return s

    def valid(self) -> bool:
        return self.r is not None

    def move(self, t) -> None:
        """state change after a period of time t
        """
        self.r += self.v * t + self.a * (t**2) * 0.5
        self.v += self.a * t

    def copy(self):
        return deepcopy(self)


class Blob:
    """The representation of an object on the screen
    Note that ct is the projection of the center of the object.
    It is not necessarily at the center of the bounding box 

    bbox: [x0, y0, x1, y1]
    """
    def __init__(self, name = "", 
                 ct : np.array = np.array([0, 0], dtype = float), 
                 bbox : np.array = np.array([0, 0, -1, -1], dtype = float)) -> None:
        self.name = name
        self.ct = ct
        self.bbox = bbox

    def valid(self) -> bool:
        """ whether it is an valid image
        """
        return self.bbox[2] >= self.bbox[0] and self.bbox[3] >= self.bbox[1]
    
    def __str__(self) -> str:
        s = self.name
        s += ", ct: " + str(self.ct)
        s += ", bbox: " + str(self.bbox)
        return s 
    
    def copy(self):
        return deepcopy(self)



class Projector:
    """World coordinates r = (x, y, z) mapped to
    screen coordinates u = (a, b)

    The pinhole is located at (0, 0, 0)
    """

    def __init__(self, h0: np.array, h1: np.array, 
                 hs : float, zoom: float = 1000) -> None:
        self.h0 = h0 / math.sqrt(h0 @ h0)
        self.h1 = h1 / math.sqrt(h1 @ h1)
        self.h2 = np.cross(self.h0, self.h1)
        self.hs = hs
        self.zoom = zoom

        # -- objects: "name", (x, y, z), v, a
        self.objs = []

        # -- the screen coordinates of the objects, including
        # center and bounding boxes
        self.scr = []

    def __str__(self) -> str:
        s = "h0, h1, h2:"
        s += str(self.h0) + " " + str(self.h1)
        s += " " + str(self.h2) + ", hs:" + str(self.hs)
        s += ", zoom: " + str(self.zoom)
        return s


    def addObject(self, ob : Object) -> None:
        """Add an object to this world
        """
        self.objs.append(ob)
        self.scr.append(Blob(ob.name))
    

    def getScreenCoordinate(self, r:np.array) -> np.array:
        """Compute the screen coordinate of a point
        """
        k = -self.hs / (r @ self.h0)
        a = -k * (r @ self.h1)
        b = -k * (r @ self.h2)
        return np.array([a, b]) * self.zoom


    def toScreen(self, ob: Object) -> Blob:
        """Project the object to the screen
        """
        br = Blob(name = ob.name)
        br.ct[0:2] = self.getScreenCoordinate(ob.r)
        
        # -- projection of borders
        n = len(ob.bbox)
        s = np.zeros([n, 2])
        for i in range(n):
            s[i] = self.getScreenCoordinate(ob.bbox[i] + ob.r)
        s1 = np.min(s, axis = 0)
        s2 = np.max(s, axis = 0)
        br.bbox[:2] = s1
        br.bbox[2:] = s2

        return br.copy()

