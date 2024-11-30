import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2

arr = lambda *args, **kwargs: np.array(*args, **kwargs)

class P3:
    basex = arr([1,0,0], dtype = float)
    basey = arr([0,1,0], dtype = float)
    basez = arr([0,0,1], dtype = float)

    @staticmethod
    def line(p1, p2, text = None, ax = plt, **kwargs):
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]
        ax.plot(x, y, z, **kwargs)
        if text is not None:
            ax.text(x[1], y[1], z[1], text)

    @staticmethod
    def lines(lines, ax = plt, **kwargs):
        for p in lines:
            if len(p) < 2:
                continue
            text = None if len(p) <= 2 else p[2]
            P3.line(p[0], p[1], text = text, ax = ax, **kwargs)

    @staticmethod
    def coordinates(st, length, ax = plt, **kwargs):
        lines = []
        tags = ["x", "y", "z"]
        for i in range(3):
            ed = arr(st, dtype = "float")
            ed[i] += length
            lines.append([st, ed, tags[i]])
        P3.lines(lines, ax = ax, **kwargs)

    @staticmethod
    def box(st, size, ax = plt, **kwargs):
        if True:
            oft = [ [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
                [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]]
            pts = np.array(oft, dtype = float) * size + st
            box = [[0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
                ]
            for i in box:
                P3.line(pts[i[0]], pts[i[1]], ax = ax, **kwargs)
        else:
            P3.rectangle_3d(st, arr([size, 0, 0]),
                arr([0, size, 0]), arr([0, 0, size]), ax = ax, **kwargs)

    @staticmethod
    def rectangle_3d(corner, h0, h1, h2, color='blue', alpha=0.3, ax = plt):
        """
        Plots a 3D rectangle (cuboid) on the given Axes3D object.

        Parameters:
        - ax: matplotlib 3D axes.
        - corner: (x, y, z) tuple for one corner of the rectangle.
        - width: Length along the x-axis.
        - height: Length along the y-axis.
        - depth: Length along the z-axis.
        - color: Color of the rectangle.
        - alpha: Transparency of the rectangle.
        """
        # Define the vertices of the cuboid
        vertices = [
            corner,
            corner + h0,
            corner + h0 + h1,
            corner + h1,
            corner + h2,
            corner + h0 + h2,
            corner + h0 + h1 + h2,
            corner + h1 + h2,
        ]
        
        # Define the six faces of the cuboid
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front
            [vertices[4], vertices[5], vertices[6], vertices[7]]   # Back
        ]
        
        # Add the faces to the plot
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))
