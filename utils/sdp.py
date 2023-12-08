# Xihang Yu
# 05/20/2023

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Perform eigenvalue decomposition and sorting
def sorteig(A, order='descend'):
    D1, V1 = np.linalg.eig(A)
    
    if order == 'ascend':
        idxsort = np.argsort(D1)
    else:
        idxsort = np.argsort(-D1)
    
    D = D1[idxsort].real
    V = V1[:, idxsort].real
    
    return V, D

def isrot(a, display=False):
    rows, cols = a.shape

    if rows != 3:
        if display:
            print('Matrix has not 3 rows')
        return False

    if cols != 3:
        if display:
            print('Matrix has not 3 columns')
        return False

    if np.linalg.norm(np.dot(a, a.T) - np.eye(3)) > 1E-10:
        if display:
            print('Matrix is not orthonormal, i.e. ||(R''R-I)|| > 1E-10')
        return False

    if np.linalg.det(a) < 0:
        if display:
            print('Matrix determinant is -1')
        return False

    return True

def project2SO3(M):
    if M.shape != (3, 3):
        raise ValueError('project2SO3 requires a 3x3 matrix as input')

    U, S, V = svd(M)
    R = U @ V
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ V

    if not isrot(R):
        raise ValueError('project2SO3 failed to produce a valid rotation')

    return R

def getTranslationError(t_gt, t_est):
    t_gt = np.reshape(t_gt, (-1,))
    t_est = np.reshape(t_est, (-1,))
    tranError = np.linalg.norm(t_gt - t_est)
    return tranError

def getAngularError(R_gt, R_est):
    rotError = np.abs(np.arccos((np.trace(np.transpose(R_gt) @ R_est) - 1) / 2))
    rotError = np.rad2deg(rotError)
    return rotError

def printErr(problem, solution):
    R_err = 0
    for i in range(problem['N']):
        R_err += getAngularError(problem['R_gt'][3*i:3*(i+1), :], solution['R_est'][3*i:3*(i+1), :])
    avg_R_err = R_err / problem['N']
    
    t_err = 0
    for i in range(problem['N']):
        t_err += getTranslationError(problem['t_gt'][:, i], solution['t_est'][:, i])
    avg_t_err = t_err / problem['N']
    
    print(f"{problem['type']} using {solution['type']}: avg_R_err = {avg_R_err}[deg], avg_t_err = {avg_t_err}.")

def visCameraTraj(problem, solution):
    # Visualize the camera trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through camera poses
    for i in range(problem['N']):

        # Extract the camera pose components
        position = problem['t_gt'][:, i]
        orientation = problem['R_gt'][3*i:3*(i+1), :]

        for j in range(3):
            ax.quiver(position[0], position[1], position[2], orientation[0, j], orientation[1, j], orientation[2, j],
                    color='b', length=problem['translationDistance']/10, normalize=True)

        # estimated camera pose
        position = solution['t_est'][:, i]
        orientation = solution["R_est"][3*i:3*(i+1), :]

        for j in range(3):
            ax.quiver(position[0], position[1], position[2], orientation[0, j], orientation[1, j], orientation[2, j],
                    color='r', length=problem['translationDistance']/10, normalize=True)
    

    # Set axis limits
    ax.set_xlim(-problem['translationDistance'], problem['translationDistance'])
    ax.set_ylim(-problem['translationDistance'], problem['translationDistance'])
    ax.set_zlim(-problem['translationDistance'], problem['translationDistance'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    from matplotlib.patches import Patch
    # Set the title with colored words
    title = 'Camera Poses Visualization'
    blue_patch = Patch(color='blue', label='GT')
    red_patch = Patch(color='red', label='Est')
    ax.set_title(title, loc='center', fontweight='bold')

    # Add legend with colored patches
    ax.legend(handles=[blue_patch, red_patch], loc='upper right')
    plt.grid(True)
    ax.set_box_aspect([1, 1, 1])
    # plt.axis('equal')
    plt.show()

