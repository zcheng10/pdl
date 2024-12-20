import os
import sys
from copy import deepcopy

import numpy as np
import math
import random

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
            gravity:float = 10,
            disturb: float = None) -> None:
        """slowdonw the velocity by a factor 
           and centrapital acceleartion
        """
        basez = arr([0, 0, 1], dtype = float)
        self.a =  -basez * gravity
        if slow > 0 and slow < 1:
            self.a += self.v * (slow - 1)
        
        b = np.cross(self.v, basez)
        self.a += b * curve

        if disturb is not None:
            # -- randomly multiply a factor
            f = random.uniform(disturb[0], disturb[1])
            self.a *= (1 + f)
        
        # print("self.a =", self.a)

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
        if k >= 0:
            # behind the camera
            return arr([-10, -10]) * self.zoom 
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
    
class Sample:
    """A sample trajectory
    """
    def __init__(self, frames:int = None) -> None:
        """
        Args:
            frames: the number of frames in this sample. If None,
                will allow for flexible # of frames
        """
        self.frames = [] if frames is None else [0] * frames
        self.cnt = 0
        pass

    def __str__(self) -> str:
        s = ""
        for i in self.frames:
            s += str(np.round(i, 3)) + "\n"
        return s

    def add(self, ob: Object, br: Blob) -> None:
        if self.cnt < len(self.frames):
            self.frames[self.cnt] = np.concatenate(
                (ob.r, ob.v, ob.a, br.ct, br.bbox)
            )
        else:
            self.frames.append(np.concatenate(
                (ob.r, ob.v, ob.a, br.ct, br.bbox)
            ))
        self.cnt += 1

class CaseGenerator:
    """Generating trajectories
    """
    def __init__(self) -> None:
        self.pj = p = Projector(
            h0 = arr([1, 0, 0], dtype = float),
            h1 = arr([0, 1, 0], dtype = float),
            hs = 0.2)
        
        self.ts = []    # objects
        self.curves = []    # curves for each object
        self.slows = []     # slow down factors for each object
        self.dt = 0.03  # time interval, in sec
        self.num = 300  # number of frames to simulate

    def config(self, slow_range : float = None, 
               curve_range: float = None,
               r_range: tuple = None, 
               v_range: tuple = None, 
               gravity: float = 10,
               ground: float = -1.5,
               disturb: float = 0.1) -> None:
        """
        Args:
            slow_range: the max slow_down, slow factor is in (0, slow_range)
            curve_range: curve in (-curve_range, +curve_range)
            r_range: ( (r0_min, r0_max), (r1_min, r1_max), (r2_min, r2_max))
                Range of location in h0, h1, h2 directions
            v_range: ( (v0_min, v0_max), (v1_min, v1_max), (v2_min, v2_max))
                range of velocity in h0, h1 and h2 directions
            gravity: the value of the gravitational constant
            ground: the location of the ground wrt to the camera
            disturb: the random disturb to its acceleration, i.e. the acceleration 
                will become a*(1-d), where d is a uniform random value in 
                (-distrub, disturb)
        """
        if slow_range is not None:
            self.slow_range = (0, slow_range)
    
        if curve_range is not None:
            self.curve_range = (-curve_range * self.dt, curve_range * self.dt)

        if v_range is not None:
            self.v_range = v_range

        if r_range is not None:
            self.r_range = r_range
    
        self.gravity = gravity
        self.ground = ground
        self.disturb = (-disturb, disturb)
        
    def gen(self, num = 300, file:str = None)->list:
        """Generate tajectories
        Args:
            file: the file name. If not None, the generated
                data will be saved to the file
        Returns:
            the generated data
        """
        pall = []   # list of 2D array
        for j, t in enumerate(self.ts):
            sm = Sample(num)
            for i in range(num):
                t.move(self.dt)
                br = self.pj.toScreen(t)
                sm.add(t, br)

                t.acc(slow = self.slows[j],
                    curve = self.curves[j],
                    disturb = self.disturb,
                    gravity = self.gravity)
                t.reflected()
            pall.append(np.round(sm.frames, 3))

        if file is not None:
            self.write(file, pall)
        return pall


    def addRandomObjects(self, cases: int):
        """Add random objects according to config
        Args:
            cases: the number of objects to be added
        """
        for i in range(cases):
            r, v = [0, 0, 0], [0, 0, 0]
            for j, rg in enumerate(self.r_range):
                r[j] = random.uniform(rg[0], rg[1])
            for j, vg in enumerate(self.v_range):
                v[j] = random.uniform(vg[0], vg[1])
            slow = random.uniform(*self.slow_range)
            curve = random.uniform(*self.curve_range)
            self.addObject("ball" + str(i+1),
                           r, v, slow, curve)


    def addObject(self, name:str, r, v,
                slow: float = 0,
                curve: float = 0,
                size: float = 0.22):
        self.ts.append(Object(name, size, 
                         r = arr(r, dtype = float),
                         v = arr(v, dtype = float))
        )
        self.curves.append(curve)
        self.slows.append(slow) 

    def write(self, file:str, pall):
        clean = lambda x : x.replace("[", "").replace("]", "").strip()
        fp = open(file, "w")
        for i in range(len(pall)):
            # comment: name, slow, curve, disturb0, disturb1, number
            s = [self.ts[i].name, 
                 np.round(self.slows[i], 2), 
                 np.round(self.curves[i], 2),
                 np.round(self.disturb[0], 2),
                 np.round(self.disturb[1], 2),
                 len(pall[0])]
            fp.write("# " + ", ".join([str(x) for x in s]) + "\n")
            for x in pall[i]:
                s = np.array2string(x, separator=", ", 
                                    max_line_width = 1000,
                                    formatter={'float_kind': lambda x: f"{x:.3f}"})
                #fp.write(s + "\n")
                fp.write(clean(s) + "\n")
        fp.close()

    @staticmethod
    def read(file):
        """Read all cases from a file
        Returns:
            palls: list of 2D arrays,
                each 2D array, [num, 15], is a case
        """
        palls, pall = [], []
        cnt = 0
        fp = open(file)
        text = fp.read()
        fp.close()
        lst = text.split("\n")

        for s in lst:
            s = s.strip()
            if not s:
                continue
            if s.startswith("#"):
                if len(pall) > 0:
                    palls.append(pall)
                    
                # comment line
                a = s[1:].split(",")
                name = a[0] 
                slow, curve, turb0, turb1 = [float(x) for x in a[1:5]]
                num = int(a[5])
                pall = np.zeros([num, 15])
                cnt = 0
            else:
                # data
                a = s.split(",")
                if len(a) == 15:
                    y = [float(x) for x in a]
                    pall[cnt, :] = arr(y)
                    cnt += 1
        
        if len(pall) > 0:
            palls.append(pall)
        return palls








