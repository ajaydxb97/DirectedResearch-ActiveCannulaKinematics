import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from mpl_toolkits import mplot3d


def plot_3D(ax, r1, r2, r3, label_str=''):
    ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=1, label=label_str)
    ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=2)
    ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=3)
    ax.scatter(r1[-1,0], r1[-1,1], r1[-1,2])

    max_range = 0.2
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(0)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(0)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(0.3) 
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

class ActiveCannula(object):

    def __init__(self, num_tubes=int(), tube_lengths=[], curve_lengths=[], initial_q=[], E=[], J=[], I=[], G=[], Ux=[], Uy=[]):

        self.n = num_tubes
        self.tube_lengths = np.array(tube_lengths)
        self.curve_lengths = np.array(curve_lengths)
        self.q_0 = np.array(initial_q)                           

        # constants
        self.J = np.array(J) # second moment of inertia
        self.I = np.array(I) # inertia
        self.E = np.array(E) # stiffness
        self.G = np.array(G) # torsion constant

        #x,y components of constant curvatures
        self.Ux = np.array(Ux)
        self.Uy = np.array(Uy)


    def get_cannula_shape(self, q, uz_0):

        q = np.array(q)
        uz_0 = np.array(uz_0)

        uz0 = uz_0.copy()                           
        B = q[:self.n] + self.q_0[:self.n]

        #initial angles
        alpha = (q[-self.n:] + self.q_0[-self.n:]) - B * uz0 
        alpha_1 = alpha[0].copy()

        # segmenting tubes
        (L, d_tip, EE, UUx, UUy) = self.segmenting(B)

        SS = L.copy()
        for i in np.arange(len(L)):
            SS[i] = np.sum(L[:i+1])

        S = SS[SS+np.min(B) > 0] + np.min(B)
        E = np.zeros((self.n, len(S)))
        Ux = np.zeros((self.n, len(S)))
        Uy = np.zeros((self.n, len(S)))
        for i in np.arange(self.n):
            E[i,:] = EE[i,SS+np.min(B)>0]
            Ux[i,:] = UUx[i,SS+np.min(B)>0]
            Uy[i,:] = UUy[i,SS+np.min(B)>0]

        span   = np.hstack((0, S))
        Length = np.array([], dtype=np.int64).reshape(0,1)
        r      = np.array([], dtype=np.int64).reshape(0,3)
        U_z    = np.array([], dtype=np.int64).reshape(0,3)

        r0 = np.array([[0, 0, 0]]).transpose()
        R0 = np.array([ [np.cos(alpha_1), np.sin(alpha_1), 0],
                        [-np.sin(alpha_1), np.cos(alpha_1), 0],
                        [0, 0, 1] ])
        R0 = R0.reshape(9,1,order='F')

        ## Solving ode for shape
        for seg in np.arange(len(S)):

            s_span = [span[seg], span[seg+1]-0.0000001]
            y0_1 = np.vstack([r0, R0])

            y0_2 = np.zeros((2*self.n,1))
            y0_2[self.n:2*self.n] = np.reshape(alpha.copy(), (self.n,1))
            y0_2[0:self.n] = np.reshape(uz0.copy(), (self.n,1))

            y_0 = np.vstack([y0_2, y0_1]).flatten()

            EI = E[:,seg] * self.I.transpose()
            GJ = self.G * self.J
            ode_sols = solve_ivp(lambda s,y: self.ode(s,y,Ux[:,seg],Uy[:,seg],EI,GJ,self.n), s_span, y_0, method='RK23')
            s = ode_sols.t[:, np.newaxis]
            y = ode_sols.y.transpose()

            shape = np.array([y[:,2*self.n], y[:,2*self.n+1], y[:,2*self.n+2]]).transpose()

            Length = np.vstack([Length, s])
            r = np.vstack([r, shape])
            U_z = np.vstack([U_z, y[:,0:self.n]])

            r0 = shape[-1][:, np.newaxis]
            R0 = y[-1, 2*self.n+3:2*self.n+12][:, np.newaxis]
            uz0 = U_z.copy()[-1]

        Uz = np.zeros((self.n,1))
        for i in np.arange(self.n):
            index =  np.argmin(np.abs(Length-d_tip[i]+0.0001) )
            Uz[i] = U_z[index, i]

        r1 = r.copy()
        tube2_end = np.argmin(np.abs(Length-d_tip[1]))
        r2 = np.array([r[0:tube2_end,0], r[0:tube2_end,1], r[0:tube2_end,2]]).transpose()
        tube3_end = np.argmin(np.abs(Length-d_tip[2]))
        r3 = np.array([r[0:tube3_end,0], r[0:tube3_end,1], r[0:tube3_end,2]]).transpose()

        return (r1, r2, r3, Uz)


    def ode(self, s, y, Ux, Uy, EI, GJ, n):

        dydt = np.zeros(2*n+12)
        ux = np.zeros((n,1))
        uy = np.zeros((n,1))

        # calculating tube's curvatures in x and y direction
        for i in np.arange(n):
            ux[i] = (1/(EI[0]+EI[1]+EI[2])) * (
                    EI[0]*Ux[0]*np.cos(y[n+i]-y[n+0]) + EI[0]*Uy[0]*np.sin(y[n+i]-y[n+0]) +
                    EI[1]*Ux[1]*np.cos(y[n+i]-y[n+1]) + EI[1]*Uy[1]*np.sin(y[n+i]-y[n+1]) +
                    EI[2]*Ux[2]*np.cos(y[n+i]-y[n+2]) + EI[2]*Uy[2]*np.sin(y[n+i]-y[n+2]) 
            )

            uy[i]= (1/(EI[0]+EI[1]+EI[2])) * (
                    -EI[0]*Ux[0]*np.sin(y[n+i]-y[n+0]) + EI[0]*Uy[0]*np.cos(y[n+i]-y[n+0]) +
                    -EI[1]*Ux[1]*np.sin(y[n+i]-y[n+1]) + EI[1]*Uy[1]*np.cos(y[n+i]-y[n+1]) +
                    -EI[2]*Ux[2]*np.sin(y[n+i]-y[n+2]) + EI[2]*Uy[2]*np.cos(y[n+i]-y[n+2]) 
            )

        # odes for twist
        for i in np.arange(n):
            dydt[i] =  ((EI[i])/(GJ[i])) * (ux[i]*Uy[i] -  uy[i]*Ux[i] )
            dydt[n+i] =  y[i]

        e3 = np.array([[0, 0, 1]]).transpose()
        uz = y[0:n]

        R1 = np.array([ [y[2*n+3], y[2*n+4], y[2*n+5]], 
                        [y[2*n+6], y[2*n+7], y[2*n+8]], 
                        [y[2*n+9], y[2*n+10], y[2*n+11]] ])

        u_hat = np.array([  [0, -uz[0], uy[0]],
                            [uz[0], 0, -ux[0]],
                            [-uy[0], ux[0], 0] ])

        # odes
        dr1 = R1@e3
        dR1 = R1@u_hat.astype(float)

        dydt[2*n+0] = dr1[0]
        dydt[2*n+1] = dr1[1]
        dydt[2*n+2] = dr1[2]

        dR = dR1.flatten()
        for i in np.arange(3, 12):
            dydt[2*n+i] = dR[i-3]

        return dydt


    #tube segmentation
    def segmenting(self, B):

        d1 = self.tube_lengths + B # position of tip of the tubes
        d2 = d1 - self.curve_lengths # position of transition point
        points = np.hstack((0, B, d2, d1))

        # finding length of each tube

        index = np.argsort(points)
        L = points[index]
        L = 1e-5*np.floor(1e5*np.diff(L)) #length of each segment 

        EE = np.zeros((self.n,len(L)))
        II = np.zeros((self.n,len(L)))
        GG = np.zeros((self.n,len(L)))
        JJ = np.zeros((self.n,len(L)))
        UUx = np.zeros((self.n,len(L)))
        UUy = np.zeros((self.n,len(L)))

        for i in np.arange(self.n): # 1:3
            a = np.argmin(np.abs(index-i+1)) 
            b = np.argmin(np.abs(index-(1*self.n+i+1))) 
            c = np.argmin(np.abs(index-(2*self.n+i+1))) 
            if L[a]==0:
                a=a+1
            if L[b]==0:
                b=b+1
            if c<len(L):
                if L[c]==0:
                    c=c+1
            EE[i,a:c]  = self.E[i]
            UUx[i,b:c] = self.Ux[i]
            UUy[i,b:c] = self.Uy[i]

        l = L[np.nonzero(L)]
        E = np.zeros((self.n,len(l)))
        Ux = np.zeros((self.n,len(l)))
        Uy = np.zeros((self.n,len(l)))   
        for i in np.arange(self.n):  
            E[i,:] = EE[i,~(L==0)]
            Ux[i,:] = UUx[i,~(L==0)]
            Uy[i,:] = UUy[i,~(L==0)]
        L = L[np.nonzero(L)]

        return (L, d1, E, Ux, Uy)





if __name__ == "__main__":

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    uz_0 = np.array([0.0, 0.0, 0.0])
    q = np.array([.2, 0, 0, np.pi, np.pi, 0])  #q = [rhos, alphas]
    initial_q = [-0.2858, -0.2025, -0.0945, 0, 0, 0]
    tube_lengths =[0.431, 0.332, 0.174]
    curve_lengths=[0.103, 0.113, 0.134]

    # constants
    J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686]) # second moment of inertia
    I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432]) # inertia
    E = np.array([ 6.3599738368e+10, 5.4548578304e+10, 4.7473091968e+10]) # stiffness
    G = np.array([2.67113017712e+10, 2.175541656e+10, 2.89822923392e+10] ) # torsion constant

    Ux = np.array([21.3, 13.108, 3.5])
    Uy = np.array([0, 0, 0])

    ctr = ActiveCannula(3, tube_lengths, curve_lengths, initial_q, E, J, I, G, Ux, Uy)

    (r1,r2,r3,Uz) = ctr.get_cannula_shape(q, uz_0)
    plot_3D(ax, r1, r2, r3, 'rhos=[0.2,0.,0.], alphas=[pi,pi,0]')

    ax.legend()
    plt.show()