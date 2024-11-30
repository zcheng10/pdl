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

        self.r = r.astype("float") if r is not None else None  # position
        self.v = v.astype("float") # velocity
        self.a = a.astype("float")  # acceleration

        # -- get its bbox relative to the center
        oft = [ [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
               [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]]
        self.bbox = np.array(oft, dtype = float) * size / 2

    def __str__(self) -> str:
        s = self.name
        s += ", size: " + str(self.size)
        s += ", r: " + str(self.r)
        s += ", v: " + str(self.v)
        s += ", a:" + str(self.a)
        return s

    def valid(self) -> bool:
        return self.r is not None

    def move(self, t) -> None:
        """state change after a period of time t
        """
        self.r += self.v * t + self.a * (t**2) * 0.5
        self.v += self.a * t

    def reflected(self, ground:float = -1.5) -> None:
        if self.r[2] <= ground:
                self.r[2] = ground
                if self.v[2] < 0:
                    self.v[2] = - self.v[2]

    def deflected(self, impulse: np.array) -> None:
        """Sudden change of v
        """
        self.v += impulse

    def acc(self, slow : float = 0, curve: float = 0,
            gravity:float = 10) -> None:
        """slowdonw the velocity by a factor 
           and centrapital acceleartion
        """
        basez = arr([0, 0, 1], dtype = float)
        self.a =  -basez * gravity
        if slow > 0 and slow < 1:
            self.a += self.v * (slow - 1)
        
        b = np.cross(self.v, basez)
        self.a += b * curve

    def clone(self):
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
    
    def clone(self):
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

        return br.clone()
    

class CaseGenerator:
    """Generating trajectories
    """
    def __init__(self) -> None:
        self.p = p = Projector(
            h0 = arr([1, 0, 0], dtype = float),
            h1 = arr([0, 1, 0], dtype = float),
            hs = 0.2)
        
        self.ts = []    # objects
        self.curves = []    # curves for each object
        self.slows = []     # slow down factors for each object
        self.dt = 0.03  # time interval, in sec
        self.num = 300  # number of frames to simulate

    def config(self, slow_range : float, curve_range: float,
               v_range:tuple) -> None:
        """Parameters:
        slow_range: the max slow_down
        curve_range: curve in (-curve_range, +curve_range)
        """

    def addObject(self, name:str, r, v, size: float = 0.22):
        self.ts.append(Object(name, size, 
                         r = arr(r, dtype = float),
                         v = arr(v, dtype = float))
        )

    





