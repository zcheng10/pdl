import matplotlib.pyplot as plt
import numpy as np
import cv2

class P3:
    @staticmethod
    def plot_line(p1, p2, text = None, ax = plt, **kwargs):
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]
        ax.plot(x, y, z, **kwargs)
        if text is not None:
            ax.text(x[1], y[1], z[1], text)

    @staticmethod
    def plot_lines(lines, ax = plt, **kwargs):
        for p in lines:
            if len(p) < 2:
                continue
            text = None if len(p) <= 2 else p[2]
            P3.plot_line(p[0], p[1], text = text, ax = ax, **kwargs)

    @staticmethod
    def plot_coordinates(st, length, ax = plt, **kwargs):
        lines = []
        tags = ["x", "y", "z"]
        for i in range(3):
            ed = st.copy()
            ed[i] += length
            lines.append([st, ed, tags[i]])
        P3.plot_lines(lines, ax = ax, **kwargs)

    @staticmethod
    def plot_box(st, size, ax = plt, **kwargs):
        oft = [ [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
               [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]]
        pts = np.array(oft, dtype = float) * size + st
        box = [[0, 1], [1, 2], [2, 3], [3, 0],
               [4, 5], [5, 6], [6, 7], [7, 4],
               [0, 4], [1, 5], [2, 6], [3, 7]
               ]
        for i in box:
            P3.plot_line(pts[i[0]], pts[i[1]], ax = ax, **kwargs)