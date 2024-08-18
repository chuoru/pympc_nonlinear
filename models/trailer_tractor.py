#!/usr/bin/env python3
##
# @file trailer_tractor.py
#
# @brief Provide the trailer tractor model for the robot.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/16.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# Standard library
import numpy as np

# External library
import casadi.casadi as cs


class TrailerTractor:
    # [x, y, theta]
    nx = 4

    # [v, w]
    nu = 2

    def __init__(self, wheel_base):
        """! The constructor of the class.
        @param wheel_base The wheel base of the robot.
        """
        self.wheel_base = wheel_base

        self.length_back = 1.0

        self.length_front = 1.0

        self.velocity_max = 1.0

    def function(self, state, input, dt):
        """! The function to calculate the next state of the robot.
        @param state The current state of the robot.
        @param input The input of the robot.
        @param dt The time step of the robot.
        """
        theta, gamma = state

        v, w = input

        dxdt = v * cs.cos(theta) * cs.cos(
            gamma) - self.length_back * cs.cos(gamma) * w

        dydt = v * cs.sin(theta) * cs.cos(
            gamma) - self.length_back * cs.sin(gamma) * w

        dthetadt = v * cs.sin(gamma) / self.length_back - w * (
            self.length_back/self.length_front) * cs.cos(gamma)

        dgammatdt = dthetadt - w

        dfdt = np.array([dxdt, dydt, dthetadt, dgammatdt])

        next_state = state + dfdt * dt

        return next_state
