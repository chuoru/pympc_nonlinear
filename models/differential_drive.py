#!/usr/bin/env python3
##
# @file differential_drive.py
#
# @brief Provide the differential drive model for the robot.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/18.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# External library
import casadi.casadi as cs


class DifferentialDrive:
    # [x, y, theta]
    nx = 3

    # [v, w]
    nu = 2

    def __init__(self, wheel_base):
        self.wheel_base = wheel_base

        self.velocity_max = 3.0

    def function(self, state, input, dt):
        v = input[0]

        w = input[1]

        dfdt = cs.vertcat(cs.cos(state[2]) * v, cs.sin(state[2]) * v, w)

        next_state = state + dfdt * dt

        return next_state
