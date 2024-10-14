# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA) visualizing with arrays
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from solution import solution
import time
import massCalculation
import gConstant
import gField
import move
import pandas as pd

# Load dataset
df = pd.read_csv('house_price.csv', usecols=['area', 'rooms', 'price'])

# Clean DataFrame
df['area'] = df['area']*0.01  # ปรับให้ใกล้เคียงหลักสิบ
df['price'] = df['price']*0.0001
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
target_array = df.to_numpy()

# Objective function
def arrayObjectiveFunction(agent_pos, target_array):
    if target_array.ndim == 1:
        target_array = target_array.reshape(1, -1)

    agent_area, agent_rooms, agent_price = agent_pos
    target_area = target_array[:, 0]
    target_rooms = target_array[:, 1]
    target_price = target_array[:, 2]

    distances = np.sqrt((target_area - agent_area) ** 2 + (target_rooms - agent_rooms) ** 2)
    min_price = target_price[np.argmin(distances)]

    return abs(agent_price - min_price)

# GSA function with array-based objective function and visualization
def GSA_visualize_array(objf, target_array, lb, ub, dim, PopSize, iters):
    ElitistCheck = 1
    Rpower = 1 
    
    s = solution()
    
    # Initializations
    vel = np.random.uniform(-1, 1, (PopSize, dim)) * 0.1 * (ub - lb)
    fit = np.zeros(PopSize)
    M = np.zeros(PopSize)
    gBest = np.zeros(dim)
    gBestScore = float("inf")
    
    pos = np.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    convergence_curve = np.zeros(iters)

    print("GSA is optimizing an array-based objective function")    
    timerStart = time.time() 
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    if dim == 2 or dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if dim == 3 else fig.add_subplot(111)

    for l in range(iters):
        for i in range(PopSize):
            pos[i, :] = np.clip(pos[i, :], lb, ub)

            fitness = objf(pos[i, :], target_array)
            fit[i] = fitness

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        M = massCalculation.massCalculation(fit, PopSize, M)
        G = gConstant.gConstant(l, iters)
        acc = gField.gField(PopSize, dim, pos, M, l, iters, G, ElitistCheck, Rpower)
        pos, vel = move.move(PopSize, dim, pos, vel, acc)
        
        convergence_curve[l] = gBestScore

        # Visualization
        if dim == 2 or dim == 3:
            ax.clear()
            if dim == 3:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], label='Agents')
                ax.scatter(gBest[0], gBest[1], gBest[2], color='r', label='Best Solution', s=100)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            else:
                ax.scatter(pos[:, 0], pos[:, 1], label='Agents')
                ax.scatter(gBest[0], gBest[1], color='r', label='Best Solution', s=100)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            
            plt.title(f'Iteration: {l + 1}')
            plt.pause(0.1)

        print(f"Iteration {l+1}: Best Score = {gBestScore}, Best Position = {gBest}")

    print("\nFinal Result:")
    print(f"Best Score: {gBestScore}")
    print(f"Best Position: {gBest}")
    plt.show()

    timerEnd = time.time()  
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.Algorithm = "GSA"
    s.objectivefunc = "ArrayObjectiveFunction"
    s.best = gBest
    s.bestScore = gBestScore

    return s

# Running GSA with array-based objective function and visualization
if __name__ == "__main__":
    lb = np.array([df['area'].min(), df['rooms'].min(), df['price'].min()])
    ub = np.array([df['area'].max(), df['rooms'].max(), df['price'].max()]) * 0.9
    dim = 3
    PopSize = 50
    iters = 100

    result = GSA_visualize_array(arrayObjectiveFunction, target_array, lb, ub, dim, PopSize, iters)
    
    print("\nBest Solution found by GSA:")
    print(f"Best Score: {result.bestScore}")
    print(f"Best Position: {result.best}")
