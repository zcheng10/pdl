import sys
from pworld import *
from illustrate import *
from matplotlib.animation import FuncAnimation

anim2D = anim3D = False
readIn = None

if len(sys.argv) >= 2:
    anim3D = (sys.argv[1] == "3")
    anim2D = (sys.argv[1] == "2")

if len(sys.argv) >= 3:
    readIn = sys.argv[2]

legendOn = ("-l" in sys.argv)

if readIn is not None:
    # -- get data from this file
    pa = CaseGenerator.read(readIn)
    tsn = len(pa)
    num = len(pa[0])

    pa = arr(pa)    # n, num, 15 
    px = pa[:, :, 9]
    py = pa[:, :, 10]
    pall = pa[:, :, 0:3]
    curves = [i+1 for i in range(tsn)]

else:
    p = Projector(h0 = P3.basex, h1 = P3.basey, hs = 0.2)
    ts = [Object("ball", size = 0.1, 
            r = arr([10, 0, -1.5]),
            v = arr([-20, 15, 11]))
    ]
    ts.append(ts[0].clone())

    curves = [0, 0.5]

    dt = 0.03
    for i, t in enumerate(ts):
        t.acc(slow = 0.1 * dt, curve = curves[i] )

    num = 300
    tsn = len(ts)
    px = np.zeros([tsn, num])
    py = np.zeros([tsn,num])
    pall = np.zeros([tsn, num, 3])

    for j, t in enumerate(ts):
        for i in range(300):
            t.move(dt)
            br = p.toScreen(t)
            px[j, i] = br.ct[0]
            py[j, i] = br.ct[1]
            pall[j, i, :] = t.r.copy()

            t.acc(slow = 0.1 * dt, curve = curves[j])
            t.reflected()

# -- get range
px_range = (np.min(px), np.max(px))
py_range = (np.min(py), np.max(py))
pall_xrange = (np.min(pall[:, :, 0]), np.max(pall[:, :, 0]))
pall_yrange = (np.min(pall[:, :, 1]), np.max(pall[:, :, 1]))
pall_zrange = (np.min(pall[:, :, 2]), np.max(pall[:, :, 2]))

print("px_range =", px_range)
print("py_range =", py_range)

if False:
    print(px)
    plt.figure()
    plt.plot(px[0,:], py[0,:])
    plt.plot(px[1,:], py[1,:])
    plt.show()
    exit()

if anim3D:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ball = [0] * tsn
    for i in range(tsn):
        ball[i], = ax.plot([], [], [], ".", label = "curve {}".format(curves[i]))
    ax.set_xlim(pall_xrange)
    ax.set_ylim(pall_yrange)
    ax.set_zlim(pall_zrange)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.axis("equal")
    if legendOn:
        ax.legend()
    ax.view_init(elev=30, azim=120)  # Adjust angles as needed

    cnt = 0
    # Animation function
    def update(frame):
        global cnt, pall
        for i in range(tsn):
            ball[i].set_data(pall[i, :cnt, 0], pall[i, :cnt, 1])  # Update X and Y
            ball[i].set_3d_properties(pall[i, :cnt, 2])        # Update Z
        cnt += 1
        if cnt >= num:
            cnt %= num
        return *ball,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=300, interval=20, blit=True)

    # Display the animation
    plt.show()

if anim2D:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ball = [0] * tsn
    for i in range(tsn):
        ball[i], = ax.plot([], [], ".",  label = "curve{}".format(curves[i]))
    # ax.axis("equal")
    ax.set_xlim(px_range)
    ax.set_ylim(py_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if legendOn:
        ax.legend()
        
    cnt = 0
    # Animation function
    def update(frame):
        global cnt, px, py
        for i in range(tsn):
            ball[i].set_data(px[i, :cnt], py[i, :cnt])  # Update X and Y
        cnt += 1
        if cnt >= num:
            cnt %= num
        return *ball,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=300, interval=20, blit=True)

    # Display the animation
    plt.show()