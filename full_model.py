from dolfin import *
import numpy as np



mesh = RectangleMesh(Point(0, -.3), Point(1, 1.3), 100, 160 )

V = VectorFunctionSpace(mesh,"CG",1)
u = interpolate(Expression(("0","x[1]>0.98 ? -(.3/.32)*(x[1]-0.98) :0 "),
                           degree=1),V)

w = interpolate(Expression(("0","x[1]< 0.02 ? -(.3/.32)*(x[1]-0.02) :0 "),
                           degree=1),V)
                         


ALE.move(mesh,u)
ALE.move(mesh,w)
# Define Taylor--Hood function space W
V = VectorElement("Lagrange", triangle, 2)
Q = FiniteElement("Lagrange", triangle, 1)
S = FiniteElement("Lagrange", triangle, 2)

W = FunctionSpace(mesh, MixedElement([V, Q, S]))

# Define Function and TestFunction(s)
w = Function(W)

(v, q, s) = split(TestFunction(W))
x = SpatialCoordinate(mesh)

D  = 30
# ep = 0.1
ep = 0.1 # use for die swell

u_in = Expression(('0', '4*(1 - pow(x[0],2))'), degree=2)

out = 'near(x[1], 1.0)'
FB = 'near(x[0], 1.0)'
cent = 'near(x[0], 0.0)'
inl = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0,1),  inl) # change (0,1) to u_in for die swell
bcu_cent = DirichletBC(W.sub(0).sub(0), (0.0), cent) 
bcu_outflow_u = DirichletBC(W.sub(0), (0,D), out)
bcr_inflow = DirichletBC(W.sub(2), 1,inl)
bcp_inflow =  DirichletBC(W.sub(1), 0,inl)




bcs = [bcu_inflow,  bcu_outflow_u, bcr_inflow, bcu_cent, bcp_inflow] #,fpi, fpo] 
# bcs_swell = [bcu_inflow_swell,  bcu_outflow_u, bcr_inflow]
 
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)
CompiledSubDomain("near(x[1], 1.0)").mark(colors, 1)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall want ds(3) for free boundary
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)
# speed up  convergence
w = interpolate(Expression(('x[1]*0','exp(std::log('+ str(D) + ')*x[1])','0*x[1]','0.5*exp(0*x[1])'), degree=2) ,W)

(u, p, r) = split(w)
sig_r = - p + 2/r * u[0].dx(0)
sig_z = - p + 2*u[1].dx(1)  - 2*x[0]/r*r.dx(1)*u[1].dx(0)
sig_rz = ep*(u[0].dx(1) - x[0]*r.dx(1)/r * u[0].dx(0)) + 1/ep *(1/r * u[1].dx(0))

rm = (- v[0].dx(0)*r*x[0]*sig_r                   \
      + ep*sig_rz*(-x[0]*(r**2 * v[0]).dx(1)   \
      + r.dx(1)*r*(x[0]**2 * v[0]).dx(0) )        \
     - r*v[0]*(-p + 2*u[0]/(r*x[0])) )*dx                      

zm = (- v[1].dx(0)*r*x[0]*sig_rz                  \
      + ep*sig_z*( -x[0]*(r**2 * v[1]).dx(1)    \
      + r.dx(1)*r*(x[0]**2*v[1]).dx(0)) )*dx   

ce = ( q*(r*(u[0]*x[0]).dx(0) + r**2*x[0]*u[1].dx(1) - x[0]**2*r*r.dx(1)*u[1].dx(0) ) ) *dx 

fb = (s*(r.dx(0))*r**2*x[0] )*dx + ( s*(u[0] - u[1]*r.dx(1))*r**2*x[0]  )*ds(3)

F = rm + zm + ce + fb
# Solve problem

solve(F == 0, w, bcs)

# Plot solutions
(u, p,r) = w.split()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 12]

plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15) 

y = np.linspace(0,1,101)

r_0 = np.exp(-0.5*np.log(D)*y)
u_0 = -0.5*r_0*np.log(D)*np.exp(np.log(D)*y)     # at eta = 1
w_0  = np.exp(np.log(D)*y)
p_0 = -np.log(D)*np.exp(np.log(D)*y)
#plt.plot(y, r_0, r--,linewidth=2)
points = [( 1,y_) for y_ in y]  # 2D points
w_line = np.array([u(point) for point in points])
r_line = np.array([r(point) for point in points])
p_line = np.array([p(point) for point in points])



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(y,w_line[:,0],y,u_0)
axs[ 0, 0].set_title('U')
axs[0, 1].plot(y,w_line[:,1],y,w_0)
axs[0, 1].set_title('W')
axs[1, 0].plot(y,r_line,y,r_0)
axs[1,0].legend(['numerical', 'leading order'], loc='upper right')
axs[1, 0].set_title('R')
axs[1, 1].plot(y,p_line,y,p_0)
axs[1, 1].set_title('P')


plt.show()



xy = np.array(mesh.coordinates())


x = xy[:,0]
y = xy[:,1]

uw = np.array([u(pt) for pt in xy])
P_ = np.array([p(pt) for pt in xy])
R_ = np.array([r(pt) for pt in xy])


X = x.reshape(161,101)
Y = y.reshape(161,101)
u_ = uw[:,0]
w_ = uw[:,1]

U = u_.reshape(161,101 )
W = w_.reshape(161,101 )
P = P_.reshape(161,101 )
R = R_.reshape(161,101 )

tableur = X
np.savetxt('x_plug.dat',tableur)

tableur = Y
np.savetxt('y_plug.dat',tableur)

tableur = U
np.savetxt('u_plug.dat',tableur)

tableur = W
np.savetxt('w_plug.dat',tableur)

tableur = R
np.savetxt('r_plug.dat',tableur)

tableur = P
np.savetxt('p_plug.dat',tableur)

