#!/usr/bin/env python3
##
# @file run.py
#
# @brief Provide executation of the program.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# External library

# Internal library
from visualizers.plotter import Plotter
from simulators.time_stepping import TimeStepping
from models.differential_drive import DifferentialDrive
from trajectory_generators.simple_p2p import SimpleP2P


def main():
    T = 4

    dt = 0.1

    model = DifferentialDrive(0.1)

    trajectory_generator = SimpleP2P(model, T, dt)

    simulator = TimeStepping(model, T, dt)

    visualizer = Plotter(simulator)

    inital_position = [0, 0, 0]

    final_position = [1, 1, 0]

    u = trajectory_generator.generate_trajectory(
        inital_position, final_position)

    simulator.run(inital_position, u)

    visualizer.plot()


if __name__ == '__main__':
    main()
