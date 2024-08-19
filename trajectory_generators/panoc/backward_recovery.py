#!/usr/bin/env python3
##
# @file backward_recovery.py
#
# @brief Provide the navigation program for the robot.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard library


# External library
import casadi as cs
import opengen as og


class BackwardRecovery:
    """! The class to generate the trajectory for the robot.

    The class provides the method to generate the trajectory for the robot.
    """

    def __init__(self, model, T, dt):
        """! The constructor of the class.
        @param model: The model of the robot.
        @param T: The time to reach the final position.
        @param dt: The time step of the optimization problem.
        @note The constructor initializes the model of the robot, the time to
        reach the final position, and the time step of the optimization
        problem. For instance:
        - self._N: The number of time steps.
        - self._dt: The time step of the optimization problem.
        - self._weights: The weights of the optimization problem.
        [q, qtheta, r, qN, qthetaN]
        - self._penalties: The penalties of the optimization problem.
        [linear_acceleration_penalty, angular_acceleration_penalty]
        """
        self._model = model

        self._N = int(T / dt)

        self._dt = dt

        self._weight_values = [10, 0.1, 0.1, 1, 200, 2]

        self._penalties = [60, 40]

        self._debug = True

        self._name = 'backward_recovery'

    def generate_trajectory(self, initial_position, final_position):
        """! Generate the trajectory for the robot.
        @param initial_position: The initial position of the robot.
        @param final_position: The final position of the robot.
        @return The input of the robot.
        """
        self._final_position = final_position

        self._define_problem()

        u = []

        manager = og.tcp.OptimizerTcpManager(f'{self._name}/optimizer')

        manager.start()

        manager.ping()

        solution = manager.call([
            initial_position], initial_guess=[1.0] * (self._model.nu*self._N))

        manager.kill()

        u = solution['solution']

        return u

    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def _define_problem(self):
        """! Define the optimization problem.
        """
        u = cs.MX.sym('u', self._model.nu * self._N)

        inital_position = cs.MX.sym('inital_position', self._model.nx)

        x = cs.vertcat(inital_position)

        cost = 0

        for t in range(0, self._model.nu*self._N, self._model.nu):
            u_t = u[t:t + 2]

            cost += self._stage_cost(x, u_t)

            x = self._model.function(x, u_t, self._dt)

        cost += self._terminal_cost(x)

        bounds = self._constraints()

        al_contraints, al_bounds = self._augmented_lagrangian_constraints()

        self._setup_optimizer(
            u, inital_position, cost, bounds, al_contraints, al_bounds)

    def _setup_optimizer(self, u, inital_position, cost, bounds, al_constrains,
                         al_bounds):
        """! Setup the optimizer.
        @param u: The input of the robot.
        @param inital_position: The initial position of the robot.
        @param cost: The cost of the optimization problem.
        @param bounds: The bounds of the optimization problem.
        @param al_constrains: The augmented Lagrangian constraints of the 
        optimization problem.
        @param al_bounds: The augmented Lagrangian bounds of the optimization 
        problem.
        """
        problem = og.builder.Problem(u, inital_position, cost) \
            .with_constraints(bounds) \
            .with_aug_lagrangian_constraints(al_constrains, al_bounds)

        build_config = og.config.BuildConfiguration() \
            .with_build_directory(f"{self._name}") \
            .with_build_mode("debug") \
            .with_tcp_interface_config() \
            .with_build_python_bindings()

        meta = og.config.OptimizerMeta().with_optimizer_name("navigation")

        solver_config = og.config.SolverConfiguration() \
            .with_tolerance(1e-4) \
            .with_initial_tolerance(1e-4) \
            .with_max_outer_iterations(5) \
            .with_delta_tolerance(1e-2) \
            .with_penalty_weight_update_factor(10.0)

        builder = og.builder.OpEnOptimizerBuilder(
            problem, meta, build_config, solver_config) \

        builder.build()

    def _stage_cost(self, x, u):
        """! Define the stage cost of the optimization problem.
        @param x: The state of the robot.
        @param u: The input of the robot.
        """
        cost = self._weight_values[0] * cs.dot(
            x[0:2] - self._final_position[0:2],
            x[0:2] - self._final_position[0:2])

        cost += self._weight_values[1] * cs.dot(
            x[2] - self._final_position[2], x[2] - self._final_position[2])

        cost += self._weight_values[2] * cs.dot(u, u)

        return cost

    def _terminal_cost(self, x):
        """!
        Define the terminal cost of the optimization problem.
        @param x: The state of the robot.
        """
        cost = self._weight_values[3] * cs.dot(
            x[0:2] - self._final_position[0:2],
            x[0:2] - self._final_position[0:2])

        cost += self._weight_values[4] * cs.dot(
            x[2] - self._final_position[2], x[2] - self._final_position[2])

        return cost

    def _constraints(self):
        """! Define the constraints of the optimization problem.
        """
        u_min = [-1.5, -0.5] * self._N

        u_max = [1.5, 0.5] * self._N

        bounds = og.constraints.Rectangle(
            u_min, u_max)

        return bounds

    def _augmented_lagrangian_constraints(
            self, linear_acceleration, angular_acceleration, gamma):
        """! Define the augmented Lagrangian constraints of the optimization 
        problem.
        @param linear_acceleration: The linear acceleration of the robot.
        @param angular_acceleration: The angular acceleration of the robot.
        @param gamma: The angle of the robot.
        """
        lin_acc_min = [-2.5] * self._N

        lin_acc_max = [1] * self._N

        ang_acc_min = [-1.5] * self._N

        ang_acc_max = [1.5] * self._N

        gamma_min = [-0.785] * self._N

        gamma_max = [0.785] * self._N

        al_constrains = cs.vertcat(
            linear_acceleration, angular_acceleration, gamma)

        al_bounds = og.constraints.Rectangle(
            lin_acc_min + ang_acc_min + gamma_min,
            lin_acc_max + ang_acc_max + gamma_max)

        return al_constrains, al_bounds
