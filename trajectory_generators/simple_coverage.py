#!/usr/bin/env python3
##
# @file simple_coverage.py
#
# @brief Provide the simple coverage trajectory generation using 
# cross-tracking error.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard library
import numpy as np

# External library
import casadi.casadi as cs


class SimpleCoverage:
    """! The class to generate the simple coverage trajectory.
    
    The class to generate the simple coverage trajectory using cross-tracking
    error.
    """
    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
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
        [q, qtheta, r, qN, qthetaN, qCTE]
        """
        self._model = model

        self._optimizer = cs.Opti()

        self._optimizer.solver('ipopt', {'print_time': False},
                               {'max_iter': 3000, 'acceptable_iter': 1000,
                                'print_level': 0, 'tol': 0.0001,
                                'acceptable_tol': 0.1, 'mu_init': 0.05,
                                'warm_start_init_point': 'yes'})

        self._N = int(T / dt)

        self._dt = dt

        self._weight_values = [10, 0.1, 1, 200, 2, 100]

        self._eps = 1e-16

    def generate_trajectory(self, initial_position, reference_paths):
        """! The function to generate the trajectory of the robot.
        @param initial_position<list>: The initial position of the robot.
        @param reference_paths<list>: The reference paths of the robot.
        """
        u = []

        self._reference_paths = self._optimizer.parameter(self._model.nx,
                                                          len(reference_paths))

        self._define_problem()

        self._optimizer.set_initial(self._u, np.zeros(
            (self._model.nu, self._N)))

        return u

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _define_problem(self):
        """! The function to define the optimization problem.
        """
        self._u = self._optimizer.variable(self._model.nu, self._N)

        self._weights = self._optimizer.parameter(len(self._weight_values))

        self._weights = self._weight_values

        self._inital_position = self._optimizer.parameter(self._model.nx)

        self._final_position = self._reference_paths[:, -1]

        x = cs.horzcat(self._inital_position)

        cost = 0

        for t in range(self._N):
            cost += self._stage_cost(
                x, self._u[:, t], self._reference_paths[:, t])

            cost += self._cross_track_cost(x, self._reference_paths)

            x = self._model.function(x, self._u[:, t], self._dt)

            self._constraints(self._u[:, t])

        cost += self._terminal_cost(x, self._final_position)

        self._optimizer.minimize(cost)

    def _cross_track_cost(self, x, reference_paths):
        """! The function to calculate the cross-track error.
        @param x: The current state of the robot.
        @param reference_paths: The reference paths of the robot.
        """
        current_waypoint = reference_paths[:, 0]

        distances = cs.SX.ones(1)

        for index, waypoint in enumerate(reference_paths):
            if index == 0:
                continue

            previous_waypoint = current_waypoint

            current_waypoint = waypoint

            line_segment = current_waypoint - previous_waypoint

            t_hat = cs.dot(x - previous_waypoint, line_segment) / (cs.dot(
                line_segment, line_segment) + self._eps)
    
            t_star = cs.fmax(0, cs.fmin(1, t_hat))

            projection = previous_waypoint + t_star * line_segment

            cross_track_error = cs.norm_2(x - projection)

            distances = cs.horzcat(distances, cross_track_error)

        cost = cs.mmin(distances[1:])**self._weights[5]

        return cost

    def _stage_cost(self, x, u, final_position):
        """! The stage cost of the optimization problem.
        @param x: The state of the robot.
        @param u: The input of the robot.
        @param final_position: The final position of the robot.
        """
        position_error = x - final_position

        cost = cs.dot(self._weights[:3], cs.power(position_error, 2))

        cost += self._weights[3] * cs.dot(u, u)

        return cost

    def _terminal_cost(self, x, final_position):
        """! The terminal cost of the optimization problem.
        @param x: The state of the robot.
        @param final_position: The final position of the robot.
        """
        position_error = x - final_position

        cost = self._weights[4] * cs.dot(position_error, position_error)

        return cost

    def _constraints(self, u):
        """! The constraints of the optimization problem.
        @param u: The input of the robot.
        """
        self._optimizer.subject_to(u[0]**2 < self._model.velocity_max**2)

        self._optimizer.subject_to(u[1]**2 < self._model.velocity_max**2)