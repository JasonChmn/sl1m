
import numpy as np
from sl1m.planner_biped import BipedPlanner
from sl1m.planner_generic import Planner
from sl1m.solver import call_MIP_solver, Solvers, solve_MIP_gurobi_cost, solve_test
from sl1m.fix_sparsity import fix_sparsity_combinatorial, fix_sparsity_combinatorial_gait, optimize_sparse_L1, generate_fixed_sparsity_problems_gait
from sl1m.problem_data import ProblemData

SOLVER = Solvers.GUROBI #Solvers.CVXPY #Solvers.GUROBI

def print_alphas(alphas, start, end):
    print("===== Print ",start," to ",end," alphas")
    nb_fixed = 0
    for i,alphas_phase in enumerate(alphas[start:end]):
        print(" - ",i," => ",alphas_phase)
        if alphas_phase[0] is not None:
            for alphas in alphas_phase[0]:
                if alphas<0.00000001:
                    nb_fixed += 1
        else:
            nb_fixed += 1
    print("nb_fixed: ",nb_fixed)
    return None

def clipAlphasMin0(alphas):
    alphas_clipped = []
    for alphas_phase in alphas:
        alphas_phase_clipped = []
        if alphas_phase[0] is None:
            alphas_phase_clipped.append([None])
            alphas_clipped.append(alphas_phase_clipped)
        else:
            for alpha in alphas_phase[0]:
                if alpha<0.000000000000001:
                    alphas_phase_clipped.append(0)
                else:
                    alphas_phase_clipped.append(alpha)
            alphas_clipped.append(alphas_phase_clipped)
    return alphas_clipped

# 6 MODES, default is 0:
#   0 = SL1M (slack continuous + cost norm L1)
#   1 = MIP original (slack integer + constraints on nb_surfaces)
#   1 = MIP original relaxed (slack continuous + constraints on nb_surfaces)                ==> This is a relaxation of 1
#   3 = MIP L1_V1 (slack integer + cost norm L1)                                            ==> The relaxation of this is SL1M
#   4 = MIP L1_V2 (slack integer + cost norm L1 + constraints on nb_surfaces)
#   5 = MIP L1_V2 relaxed (slack continuous + cost norm L1 + constraints on nb_surfaces)    ==> This is a relaxation of 4
NAMES = {   0:"SL1M (slack continuous + cost norm L1)",
            1:"MIP original (slack integer + constraints on nb_surfaces)",
            2:"MIP original relaxed (slack continuous + constraints on nb_surfaces)",
            3:"MIP L1_V1 (slack integer + cost norm L1)",                                   # Takes for ever, it will test ALL the combinatorics because he doesn't know the min
            4:"MIP L1_V2 (slack integer + cost norm L1 + constraints on nb_surfaces)",
            5:"MIP L1_V2 relaxed (slack continuous + cost norm L1 + constraints on nb_surfaces)",    
        }

def is_sparsity_fixed(alphas):
    surfaces_decided = True
    for alphas_phase in alphas:
        phase_has_0 = False
        if alphas_phase[0] is None:
            phase_has_0 = True
        else:
            for alpha in alphas_phase[0]:
                if alpha<0.0000000001:
                    phase_has_0 = True
                    break
        if not phase_has_0:
            surfaces_decided = False
            break
    return surfaces_decided

def get_sum_alphas(alphas):
    total = 0
    for alphas_phase in alphas:
        if alphas_phase[0] is None:
            pass
        else:
            for alpha in alphas_phase[0]:
                total += alpha
    return total

def solve_feasibility_problem(pb, lp_solver=SOLVER, qp_solver=SOLVER, com=True, mode_solve=0, return_first_solution=False):
    #print("\n\n********************************************")
    print("solve_feasibility_problem => ",NAMES[mode_solve])
    # Initialization of the problem
    planner = Planner(mip=True, com=com)
    G, h, C, d = planner.convert_pb_to_LP(pb)
    slack_selection_vector = planner.alphas
    P, q = None, None # No cost, we solve a feasibility problem
    # Solve
    USE_RELAX = False
    if mode_solve==0:
        # SL1M (slack continuous + cost norm L1)
        SET_CONSTR_NB_SURF=0
        MODE_L1=1
        CONTINUOUS=1
    elif mode_solve==1:
        # MIP original (slack integer + constraints on nb_surfaces)
        SET_CONSTR_NB_SURF=1 
        MODE_L1=0 
        CONTINUOUS=0
    elif mode_solve==2:
        # MIP original relaxed (slack continuous + constraints on nb_surfaces)
        SET_CONSTR_NB_SURF=1 
        MODE_L1=0
        CONTINUOUS=1
    elif mode_solve==3:
        # MIP L1_V1 (slack integer + cost norm L1) => The relaxation of this is SL1M
        SET_CONSTR_NB_SURF=0 
        MODE_L1=1
        CONTINUOUS=0
        USE_RELAX = False
        #print("NOT REALLY 3, TO CHANGE, THIS IS SUPPOSED TO BE LIKE SL1M WITH THE RELAX FUNCTION OF GUROBI")
        # OK CONFIRMED, THIS IS THE SAME THAN SL1M
    elif mode_solve==4:
        # MIP L1_V2 (slack integer + cost norm L1 + constraints on nb_surfaces)
        SET_CONSTR_NB_SURF=1
        MODE_L1=1
        CONTINUOUS=0
    elif mode_solve==5:
        # MIP L1_V2 relaxed (slack continuous + cost norm L1 + constraints on nb_surfaces)
        SET_CONSTR_NB_SURF=1
        MODE_L1=1
        CONTINUOUS=1
    time = 0.
    result = solve_test(slack_selection_vector,G,h,C,d, SET_CONSTR_NB_SURF=SET_CONSTR_NB_SURF, MODE_L1=MODE_L1, CONTINUOUS=CONTINUOUS, USE_RELAX=USE_RELAX)
    #print("solve_feasibility_problem, first solve: ",result.success)
    #result_relax = solve_test(slack_selection_vector,G,h,C,d, SET_CONSTR_NB_SURF=SET_CONSTR_NB_SURF, MODE_L1=MODE_L1, CONTINUOUS=False, USE_RELAX=True)
    if return_first_solution and mode_solve==0:
        coms, moving_foot_pos, all_feet_pos = planner.get_result(result.x)
        alphas = planner.get_alphas(result.x)
        print_alphas(alphas,0,len(alphas))
        surface_indices = planner.selected_surfaces(alphas)
        print("                          => Relaxation result of SL1M returned directly.")
        return ProblemData(True, time, coms, moving_foot_pos, all_feet_pos, surface_indices)

    if result is not None and result.success:
        alphas = planner.get_alphas(result.x)
        #print_alphas(alphas,0,len(alphas))
        #print("=== TEST CLIP")
        #alphas_clipped = clipAlphasMin0(alphas)
        #print_alphas(alphas_clipped,0,len(alphas_clipped))
        #input("...")
        """
        print("===== RELAXED")
        alphas_relaxed = planner.get_alphas(result_relax.x)
        print_alphas(alphas_relaxed,0,len(alphas))
        input("...")
        result = result_relax
        alphas = alphas_relaxed
        """
        #print_alphas(alphas,0,len(alphas))
        surfaces_decided = is_sparsity_fixed(alphas) or (not CONTINUOUS)
        time += result.time
        #print("Result success: ",result.success)
        print("Surfaces decided: ",surfaces_decided)
        if not surfaces_decided:
            #print("RESULT =====> surfaces_decided: ",surfaces_decided)
            #print("Total sum alphas: ",get_sum_alphas(alphas))
            #input("...")
            print("FIXING SPARSITY...")
            pbs = generate_fixed_sparsity_problems_gait(pb, alphas)
            NB_EXPLO_COMB = 5#len(pbs) # Usually 1 or 2 should be enough. It's almost never more than that, and more would mean than SL1M failed with a bad initial guess.
            for i in range(0,NB_EXPLO_COMB): # X comb max
                fixed_pb, combination = pbs[i]
                print(i," / ",NB_EXPLO_COMB," ==> Test combination: ",combination)
                G, h, C, d = planner.convert_pb_to_LP(fixed_pb)
                slack_selection_vector = planner.alphas
                result_fixed = solve_test(slack_selection_vector,G,h,C,d, SET_CONSTR_NB_SURF=SET_CONSTR_NB_SURF, MODE_L1=MODE_L1, CONTINUOUS=CONTINUOUS)
                time += result_fixed.time
                alphas = planner.get_alphas(result_fixed.x)
                #print_alphas(alphas,0,len(alphas)) # Only None
                #surfaces_decided = is_sparsity_fixed(alphas)
                #print("SURFACES DECIDED => ",surfaces_decided)
                print("  ==> result_fixed.success: ",result_fixed.success)
                if result_fixed.success:
                    #print("*********")
                    #print("** SUCESS")
                    #print("*********")
                    coms, moving_foot_pos, all_feet_pos = planner.get_result(result_fixed.x)
                    surface_indices = planner.selected_surfaces(alphas)
                    print("                          => Successful result in comb")
                    return ProblemData(True, time, coms, moving_foot_pos, all_feet_pos, surface_indices)
        else:
            coms, moving_foot_pos, all_feet_pos = planner.get_result(result.x)
            surface_indices = planner.selected_surfaces(alphas)
            print("                          => Successful result not comb")
            return ProblemData(True, time, coms, moving_foot_pos, all_feet_pos, surface_indices)
    print("                          => Fail")
    return ProblemData(False, time)

# ----------------------- L1 -----------------------------------------------------------------------


def solve_L1_combinatorial(pb, lp_solver=SOLVER, qp_solver=SOLVER, costs=None, com=True):
    """
    Solve the problem by first chosing the surfaces with a L1 norm minimization problem handling the
    combinatorial if necesary, and then optimizing the feet positions with a QP
    @param pb problem to solve
    @surfaces surfaces to choose from
    @lp_solver solver to use for the LP
    @qp_solver solver to use for the QP
    @costs cost dictionary specifying the cost functions to use and their parameters
    @return ProblemData storing the result
    """
    planner = Planner(mip=False, com=com)
    sparsity_fixed, pb, surface_indices, t, pb_data = fix_sparsity_combinatorial_gait(planner, pb, lp_solver)
    if sparsity_fixed and costs is None:
        print("SL1M solved and no cost. Time : ",pb_data.time/1000.)
        pass
    elif sparsity_fixed:
        print("  Surfaces selected     : ",surface_indices)
        pb_data = optimize_sparse_L1(planner, pb, costs, qp_solver, lp_solver)
        pb_data.surface_indices = surface_indices
    else:
        return ProblemData(False, t)
    pb_data.time += t
    return pb_data


def solve_L1_combinatorial_biped(pb, lp_solver=SOLVER, qp_solver=SOLVER, costs=None):
    """
    Solve the problem for a biped by first chosing the surfaces with a L1 norm minimization problem
    handling the combinatorial if necesary, and then optimizing the feet positions with a QP
    @param pb problem to solve
    @surfaces surfaces to choose from
    @lp_solver solver to use for the LP
    @qp_solver solver to use for the QP
    @costs cost dictionary specifying the cost functions to use and their parameters
    @return ProblemData storing the result
    """
    planner = BipedPlanner()
    sparsity_fixed, pb, surface_indices, t = fix_sparsity_combinatorial(planner, pb, lp_solver)
    if sparsity_fixed:
        pb_data = optimize_sparse_L1(planner, pb, costs, qp_solver, lp_solver)
        pb_data.surface_indices = surface_indices
    else:
        return ProblemData(False, t)
    pb_data.time += t
    return pb_data



# ----------------------- MIP -----------------------------------------------------------------------

def solve_MIP(pb, costs=None, solver=SOLVER, com=False):
    """
    Solve the problem with a MIP solver
    @param pb problem to solve
    @surfaces surfaces to choose from
    @costs cost dictionary specifying the cost functions to use and their parameters
    @solver MIP solver to use
    @return ProblemData storing the result
    """
    print("solve_MIP with costs: ",costs)
    planner = Planner(mip=True, com=com)
    G, h, C, d = planner.convert_pb_to_LP(pb)
    slack_selection_vector = planner.alphas
    P, q = None, None

    # If no combinatorial call directly a QP
    if solver == Solvers.CVXPY and np.linalg.norm(slack_selection_vector) < 1:
        return optimize_sparse_L1(planner, pb, costs)

    if costs != None:
        P, q = planner.compute_costs(costs)
    result = call_MIP_solver(slack_selection_vector, P, q, G, h, C, d, solver=solver)

    if costs != None and solver == Solvers.CVXPY:
        alphas = planner.get_alphas(result.x)
        selected_surfaces = planner.selected_surfaces(alphas)
        for i, phase in enumerate(pb.phaseData):
            for j in range(len(phase.n_surfaces)):
                phase.S[j] = [phase.S[j][selected_surfaces[i][j]]]
                phase.n_surfaces[j] = 1
        return optimize_sparse_L1(planner, pb, costs, QP_SOLVER=Solvers.CVXPY, LP_SOLVER=Solvers.CVXPY)

    if result.success:
        alphas = planner.get_alphas(result.x)
        coms, moving_foot_pos, all_feet_pos = planner.get_result(result.x)
        surface_indices = planner.selected_surfaces(alphas)
        return ProblemData(True, result.time, coms, moving_foot_pos, all_feet_pos, surface_indices)
    return ProblemData(False, result.time)


def solve_MIP_biped(pb, costs={}, solver=SOLVER):
    """
    Solve the problem with a MIP solver for a biped
    @param pb problem to solve
    @surfaces surfaces to choose from
    @costs cost dictionary specifying the cost functions to use and their parameters
    @solver MIP solver to use
    @return ProblemData storing the result
    """
    planner = BipedPlanner(mip=True)
    G, h, C, d = planner.convert_pb_to_LP(pb)
    slack_selection_vector = planner.alphas

    if costs != None:
        P, q = planner.compute_costs(costs)
        result = solve_MIP_gurobi_cost(slack_selection_vector, P, q, G, h, C, d)
    else:
        result = call_MIP_solver(slack_selection_vector, G, h, C, d, solver=solver)

    if result.success:
        alphas = planner.get_alphas(result.x)
        coms, moving_foot_pos, all_feet_pos = planner.get_result(result.x)
        surface_indices = planner.selected_surfaces(alphas)
        return ProblemData(True, result.time, coms, moving_foot_pos, all_feet_pos, surface_indices)
    return ProblemData(False, result.time)
