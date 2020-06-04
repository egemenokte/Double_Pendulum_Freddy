from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

#%% CHANGE THESE PARAMETERS FOR DIFFERENT EFFECTS! 
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m 
L2 = 1.0  # length of pendulum 2 in m 
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
L=L1+L2 #Total Length

# th1s and th2s are the initial angles (degrees)
# w1s and w2s are the initial angular velocities (degrees per second)
th1s=[180,180,-90,45] #order is arm,arm,leg,leg
w1s=[0,0,100,0]
th2s=[179,178.9,40,0]
w2s = [0,0,100,0]

# th1s=[180,180,180,180] #order is arm,arm,leg,leg
# w1s=[0,0,0,0]
# th2s=[179,178.9,-179,-178.9]
# w2s = [0,0,0,0]

#%%

#%%
def derivs(state, t): #double pendulum functions

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

#%% Animation Functions

def init(): #initial plot function
    for i in range(N):
        line[str(i)].set_data([], [])
    time_text.set_text('')

    return line['0'],line['1'], time_text


def animate(i): #main plot function
    for j in range(N):
        thisx = [0, x1[str(j)][i], x2[str(j)][i]]
        thisy = [initial[j], initial[j]+y1[str(j)][i], initial[j]+y2[str(j)][i]]
        line[str(j)].set_data(thisx, thisy)
    
    time_text.set_text(time_template % (i*dt))
    return line['0'],line['1'],line['2'],line['3'], time_text
#%%

#define empty x and y coordinates for pendulums
x1={}
x2={}
y1={}
y2={}
N=len(th1s)
initial=[0,0,-2,-2] #initial offset positions #legs are 2 steps down
for i in range(N):# create a time array from 0..100 sampled at 0.05 second steps

    dt = 0.05
    t = np.arange(0, 20, dt)
    

    th1 = th1s[i]
    w1 = w1s[i]
    th2 = th2s[i]
    w2 = w2s[i]
    
    # initial state
    state = np.radians([th1, w1, th2, w2])
    
    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    x1[str(i)] = L1*sin(y[:, 0]) #first node
    y1[str(i)] = -L1*cos(y[:, 0]) 
    
    x2[str(i)] = L2*sin(y[:, 2]) + x1[str(i)] #second node
    y2[str(i)] = -L2*cos(y[:, 2]) + y1[str(i)] 
    
#%% Plotting begins here

#initialize   
plt.close('all')
fig = plt.figure(figsize=(8,8))
fig.set_tight_layout(True)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-L-0.1, L+0.1), ylim=(-L-2.01, L+0.1))
ax.set_aspect('equal')


lww=5 #line weight
ax.set_title('Double Pendulum Freddy',fontsize=18)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

line={} #these line elements will plot positions of arms and legs (4 of them)
line['0'], = ax.plot([], [], 'o-', lw=lww,color='black',markersize=8) 
line['1'], = ax.plot([], [], 'o-', lw=lww,color='black',markersize=8)
line['2'], = ax.plot([], [], 'o-', lw=lww,color='black',markersize=8)
line['3'], = ax.plot([], [], 'o-', lw=lww,color='black',markersize=8)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes,fontsize=14)
time_template = 'Time = %.1fs'

#Torso
ln, = ax.plot([], [], 'o-', lw=lww,markersize=4,color='black')
ln.set_data([0,0],[0,-2])


#Head is a circle
ln3, = ax.plot([], [], 'o-', lw=lww,markersize=4,color='black')
t = np.linspace(0,2*np.pi,100)
xc = 0
yc = 0.5
r = 0.5
xapprox = r*np.cos(t) + xc
yapprox = r*np.sin(t) + yc
ln3, = ax.plot(xapprox, yapprox , 'o-', lw=lww,markersize=4,color='black')

#first eye, small circle
t = np.linspace(0,2*np.pi,100)
xc = -0.15
yc = 0.65
r = 0.01
xapprox = r*np.cos(t) + xc
yapprox = r*np.sin(t) + yc
ln4, = ax.plot(xapprox, yapprox , 'o-', lw=lww,markersize=4,color='black')

#second eye, small circle
t = np.linspace(0,2*np.pi,100)
xc = 0.15
yc = 0.65
r = 0.01
xapprox = r*np.cos(t) + xc
yapprox = r*np.sin(t) + yc
ln5, = ax.plot(xapprox, yapprox , 'o-', lw=lww,markersize=1,color='black')

#smile, half circle
t = np.linspace(np.pi,2*np.pi,50)
xc = 0.0
yc = 0.4
r = 0.15
xapprox = r*np.cos(t) + xc
yapprox = r*np.sin(t) + yc
ln5, = ax.plot(xapprox, yapprox , 'o-', lw=lww-2,markersize=4,color='black')

#%% Animation function
ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dt*800, blit=True, init_func=init)


plt.show()
