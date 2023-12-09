from SimSyncRegularized import SimSyncReg
import numpy as np
from scipy.stats import chi2
import sys
sys.path.append('/content/drive/MyDrive/SIM-Sync-MONO/TEASER-plusplus/build/python/teaserpp_python')
import teaserpp_python

def TEASER_SimSync(N, edges, pointclouds, scale_gt = None, reg_lambda=1):

    #######################################
    ### step 1: edge-wise weight detect ###
    #######################################

    inliers = []
    weights = []
    start_idx = 0
    for idx in range(len(edges)):

        NumberMeasurements = pointclouds[idx].shape[1]
        dof = 3
        MeasurementNoiseStd = 0.1

        # Solve with TEASER
        print("Solving with:")
        print("  - TEASER")

        # Populating the parameters
        src = pointclouds[idx][3:6,:]
        dst = pointclouds[idx][0:3,:]

        # Thresholds: MeasurementNoiseStd is approximated
        
        epsilon_square = chi2.ppf(0.9999, dof) * (MeasurementNoiseStd ** 2)
        epsilon = np.sqrt(epsilon_square)

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = epsilon
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(src, dst)

        solution = solver.getSolution()

        print("=====================================")
        print("          TEASER++ Results           ")
        print("=====================================")

        print("Estimated rotation: ")
        print(solution.rotation)

        print("Estimated translation: ")
        print(solution.translation)

        print("Estimated scaling: ")
        print(solution.scale)

        residues = solution.scale * solution.rotation @ src + solution.translation.reshape(3,1) - dst
        # residues = solution.rotation @ src + solution.translation.reshape(3,1) - dst
        residuals = np.linalg.norm(residues, axis=0)
        residuals = residuals/solver_params.noise_bound
        residuals = residuals**2
        weights_edge = residuals < 1

        pred_inliers = np.where(weights_edge == True)[0]
        
        inliers = np.append(inliers, start_idx + pred_inliers)
        start_idx += NumberMeasurements
        weights = np.append(weights, weights_edge)


    #######################################
    ####### step 2: weighted SIMSync ######
    #######################################
    # Solvers


    results = {}
    results['gnc'] = {}
    results['gnc']['algname'] = 'TEASER+SIM-Sync'

    # solution = SimSync(N, edges, pointclouds,scale_gt, Weights=weights)
    # scale_gt = np.ones((N,1))
    solution = SimSyncReg(N, edges, pointclouds,scale_gt, Weights=weights, reg_lambda=reg_lambda)

    return solution, weights

