import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock
import os

from sl1m.generic_solver import solve_L1_combinatorial, solve_MIP
from sl1m.problem_definition import Problem
from sl1m.stand_alone_scenarios.surfaces.stair_surfaces import quadruped_surfaces_gait as surfaces
from sl1m.stand_alone_scenarios.surfaces.stair_surfaces import scene

import sl1m.tools.plot_tools as plot

USE_COM = True
GAIT = [np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), np.array([0, 1, 1, 1])]

paths = [os.environ["INSTALL_HPP_DIR"] + "/share/anymal-rbprm/com_inequalities/feet_quasi_flat/anymal_",
         os.environ["INSTALL_HPP_DIR"] + "/share/anymal-rbprm/relative_effector_positions/anymal_"]
limbs = ['RHleg', 'RFleg', 'LHleg', 'LFleg']
others = ['RH_ADAPTER_TO_FOOT', 'RF_ADAPTER_TO_FOOT', 'LH_ADAPTER_TO_FOOT', 'LF_ADAPTER_TO_FOOT']
suffix = "_effector_frame_quasi_static_upscale.obj"
offsets = {'RFleg': [0.373, -0.264, -0.47], 'LFleg': [0.373, 0.264, -0.47],
           'RHleg': [-0.373, -0.264, -0.47], 'LHleg': [-0.373, 0.264, -0.47]}

if __name__ == '__main__':
    t_init = clock()
    R = [np.identity(3)] * len(surfaces)
    t_1 = clock()

    q_init = [-0.5, 0., 0.47]

    initial_contacts = [np.array(q_init) + offsets[limb] for limb in limbs]
    t_2 = clock()

    pb = Problem(limb_names=limbs, other_names=others, constraint_paths=paths, suffix_com=suffix)
    pb.generate_problem(R, surfaces, GAIT, initial_contacts, q_init)
    t_3 = clock()

    result = solve_MIP(pb, com=USE_COM)
    t_end = clock()

    print(result)
    print("Optimized number of steps:              ", pb.n_phases)
    print("Total time is:                          ", 1000. * (t_end-t_init))
    print("Computing the surfaces takes            ", 1000. * (t_1 - t_init))
    print("Computing the initial contacts takes    ", 1000. * (t_2 - t_1))
    print("Generating the problem dictionary takes ", 1000. * (t_3 - t_2))
    print("Solving the problem takes               ", 1000. * (t_end - t_3))
    print("The LP and QP optimizations take        ", result.time)

    ax = plot.draw_scene(scene)
    if(result.success):
        plot.plot_planner_result(result.coms, result.all_feet_pos, ax=ax, show=True)
    else:
        plt.show()
