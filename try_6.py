import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kinematics:
    def __init__(self):
        self.Table = []
        self.Matrices = []
        self.positions = []

    def Transform_matrix(self, theta, d, a, alfa):
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alfa), np.sin(alfa)*np.sin(theta), a*np.cos(theta)],
                      [np.sin(theta), np.cos(alfa)*np.cos(theta), -np.sin(alfa)*np.cos(theta), a*np.sin(theta)],
                      [0, np.sin(alfa), np.cos(alfa), d],
                      [0, 0, 0, 1]])
        return T

    def add_values(self, dh_parameters):
        self.Table.append(dh_parameters)

    def show_table(self, angles_in_degrees=True):
        header = ["Theta", "d", "a", "Alfa"]
        print("+---------" * len(header) + "+")
        print("|", end=" ")
        for h in header:
            print(f"{h:8}", end=" | ")
        print()
        print("+---------" * len(header) + "+")
        for row in self.Table:
            print("|", end=" ")
            for i, value in enumerate(row):
                # Convert angles to degrees
                if angles_in_degrees and (i == 0 or i == 3):
                    value = np.degrees(value)
                print(f"{value:8.3f} |", end=" ")
            print()
        print("+---------" * len(header) + "+")

    def get_transformed_values(self):
        self.Matrices = [self.Transform_matrix(*row) for row in self.Table]

    def geting_positions(self):
        self.positions = [[0, 0, 0]]
        T = np.eye(4)
        for matrix in self.Matrices:
            T = T @ matrix
            self.positions.append(T[:3, 3])
        self.positions = np.array(self.positions)

    def plotting_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], c='r', marker='o')
        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], linestyle='-', color='b')
        for i, pos in enumerate(self.positions):
            ax.text(pos[0], pos[1], pos[2], f'Joint {i}', color='black', fontsize=8, ha='right')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        n = 450
        ax.set_xlim([-n, n])
        ax.set_ylim([-n, n])
        ax.set_zlim([-n, n])
        ax.quiver(0, 0, 0, 4, 0, 0, color='g', label='X-axis')
        ax.quiver(0, 0, 0, 0, 4, 0, color='b', label='Y-axis')
        ax.quiver(0, 0, 0, 0, 0, 4, color='y', label='Z-axis')
        ax.view_init(elev=30, azim=-75)
        plt.legend()
        plt.show()

    def inverse_kinematics_optimization(self, target_position):
        def objective_function(params):
            # Unpack the parameters to optimize
            a, theta1, theta2, theta3, theta4 = params
            
            # Update only the specified DH parameters in the table
            self.Table[0][3] = a       # Modify link length a
            self.Table[1][0] = theta1 # Modify joint angle theta1
            self.Table[2][0] = theta2 # Modify joint angle theta2
            self.Table[3][0] = theta3 # Modify joint angle theta3
            self.Table[4][1] = theta4 # Modify offset d
            
            # Update transformation matrices based on the new DH table
            self.get_transformed_values()
            
            # Calculate the forward kinematics to get the end-effector position
            end_f_matrix = np.linalg.multi_dot(self.Matrices)
            end_effector_position = end_f_matrix[:3, 3]
            
            # Compute the error as the Euclidean distance to the target position
            error = np.linalg.norm(end_effector_position - target_position)
            return error

        # Initial guesses for the parameters to optimize
        initial_guess = [
            self.Table[0][3],  # Initial value of 'a'
            self.Table[1][0],  # Initial value of theta1
            self.Table[2][0],  # Initial value of theta2
            self.Table[3][0],  # Initial value of theta3
            self.Table[4][1]   # Initial value of theta4
        ]
        
        # Define bounds for the parameters
        bounds = [
            (-np.pi, np.pi ),          # Bounds for 'a'
            (-np.pi, np.pi),  # Bounds for theta1
            (-np.pi, np.pi),  # Bounds for theta2
            (-np.pi, np.pi),  # Bounds for theta3
            (-1000, 1000)         # Bounds for theta4
        ]
        
        # Perform optimization using SciPy's minimize
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', tol=1e-6, bounds=bounds)
        
        # Display results
        if result.success:
            optimized_params = result.x
            print("Optimization Successful!")
            print(f"Optimized Parameters: {optimized_params}")
            
            # Update the DH table with the optimized parameters
            self.Table[0][3] = optimized_params[0]
            self.Table[1][0] = optimized_params[1]
            self.Table[2][0] = optimized_params[2]
            self.Table[3][0] = optimized_params[3]
            self.Table[4][1] = optimized_params[4]
            
            # Recompute positions and plot the results
            self.get_transformed_values()
            self.geting_positions()
            self.plotting_points()
            self.show_table()

            # Calculate and print the final end-effector coordinates
            end_f_matrix = np.linalg.multi_dot(self.Matrices)
            final_end_effector_position = end_f_matrix[:3, 3]
            print(f"Final End-Effector Position: [{final_end_effector_position[0]:.2f}, {final_end_effector_position[1]:.2f}, {final_end_effector_position[2]:.2f}]")


        else:
            print("Optimization failed.")

# Example usage:
# Target end-effector position
target_position = np.array([-80, 450, 0])

k = Kinematics()

# Define DH parameters
k.add_values([0, 0, 70, 0])                # [theta, d, a, alpha]
k.add_values([0, 0, 200, 0])
k.add_values([np.radians(-90), 0, 250, 0])
k.add_values([np.radians(0), 0, 0, np.radians(90)])
k.add_values([np.radians(0), 0, 0, 0])

k.get_transformed_values()
k.inverse_kinematics_optimization(target_position)
