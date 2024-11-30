import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append("../")
from src.pnn import *

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-300, 1000)  # X-axis range
ax.set_ylim(-50, 300)  # Y-axis range

# Initialize the ball

ball, = plt.plot([], [], 'o', color='blue', markersize=12)
ref, = plt.plot([], [], 'o', color='green', markersize=12)

# Initial position and velocity of the ball
r = (20, 0, 0.1)
v = (0, 20, 20)
a = (0, 0, -10)
fb = objectTensor(r, v, a, size = 0.1).unsqueeze(0)
fb.requires_grad = False
dt = 0.01
p = ProjectorNN(basis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
inited = False
px, py = [], []

# Animation function
def update(frame):
    global fb, p, dt, inited, px, py
    
    # Update position
    fb = MotionNN.moved(fb, dt)
    fc = object2box(fb, sz = 0.1)
    a = p(fc)
    pos = a.flatten().detach().numpy()
    sz = max(pos[1]-pos[0], pos[3] - pos[2])

    # Update the ball's position
    px.append(pos[0])
    py.append(pos[2])
    ball.set_data(pos[0], pos[2])
        #(px, py)
        # 
    ball.set_markersize(int(sz))

    if not inited:
        ref.set_data(pos[0], pos[2])
        inited = True

    MotionNN.reflected(fb)
    MotionNN.acc(fb, curve = 0.2, slow = 0)
    # print("fb =", fb)
    return ball,

# Create the animation
ani = FuncAnimation(fig, update, frames=300, interval=20, blit=True)

# Display the animation
plt.show()
