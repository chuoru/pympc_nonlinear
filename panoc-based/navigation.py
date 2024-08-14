import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# Build parametric optimizer
# ------------------------------------
(nu, nx, N, L, ts) = (2, 3, 20, 0.5, 0.1)
(xref, yref, thetaref) = (1, 1, 0)
(q, qtheta, r, qN, qthetaN) = (10, 0.1, 1, 200, 2)

u = cs.SX.sym('u', nu*N)
z0 = cs.SX.sym('z0', nx)

(x, y, theta) = (z0[0], z0[1], z0[2])

cost = 0
for t in range(0, nu*N, nu):
    cost += q*((x-xref)**2 + (y-yref)**2) + qtheta*(theta-thetaref)**2
    u_t = u[t:t+2]
    theta_dot = (1/L) * (u_t[1] * cs.cos(theta) - u_t[0] * cs.sin(theta))
    cost += r * cs.dot(u_t, u_t)
    x += ts * (u_t[0] + L * cs.sin(theta) * theta_dot)
    y += ts * (u_t[1] - L * cs.cos(theta) * theta_dot)
    theta += ts * theta_dot

cost += qN*((x-xref)**2 + (y-yref)**2) + qthetaN*(theta-thetaref)**2

umin = [-3.0] * (nu*N)
umax = [3.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, z0, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("rust")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()

# Use TCP server
# ------------------------------------
mng = og.tcp.OptimizerTcpManager('rust/navigation')
mng.start()

mng.ping()
solution = mng.call([-1.0, 2.0, 0.0], initial_guess=[1.0] * (nu*N))
mng.kill()

print(f"num_outer_iterations: {solution['num_outer_iterations']}")
print(f"num_inner_iterations: {solution['num_inner_iterations']}")
print(f"last_problem_norm_fpr: {solution['last_problem_norm_fpr']}")
print(f"exit_status: {solution['exit_status']}")
print(f"solution: {solution['solution']}")
print(f"lagrange_multipliers: {solution['lagrange_multipliers']}")
print(f"f2_norm: {solution['f2_norm']}")
print(f"solve_time_ms: {solution['solve_time_ms']}")
print(f"penalty: {solution['penalty']}")
# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
u_star = solution['solution']
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]

plt.subplot(211)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(212)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.xlabel('Time')
plt.show()
