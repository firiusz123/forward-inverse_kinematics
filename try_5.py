import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Kinematics:
    def __init__(self):
        self.Table = []  # To store DH parameters
        self.Matrices = []  # To store individual transformation matrices
        self.Positions = []  # To store positions of each joint

    def add_values(self, data):
        """Add a row of DH parameters to the table."""
        self.Table.append(data)

    def show_table(self):
        """Display the DH parameter table."""
        print("DH Table:")
        print(np.array(self.Table))

    def get_transformed_values(self):
        """Calculate the transformation matrices for each row in the DH table."""
        self.Matrices = []
        for i in self.Table:
            theta, d, a, alpha = i
            matrix = np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            self.Matrices.append(matrix)

    def geting_positions(self):
        """Calculate and store the positions of each joint."""
        self.Positions = [np.array([0, 0, 0])]  # Starting position
        T = np.eye(4)  # Initialize transformation matrix
        for matrix in self.Matrices:
            T = T @ matrix
            self.Positions.append(T[:3, 3])  # Extract position (x, y, z)

    def plotting_points(self):
        """Plot the positions of the joints."""
        positions = np.array(self.Positions)
        plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], marker='o', label='Joints')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Robot Arm Configuration")
        plt.legend()
        plt.grid()
        plt.show()

    def get_end_effector_position(self):
        """Calculate and return the end-effector position based on the current DH table."""
        T = np.eye(4)
        for matrix in self.Matrices:
            T = T @ matrix
        end_effector_position = T[:3, 3]  # Extract the position (x, y, z)
        print(f"End-Effector Position: X={end_effector_position[0]:.3f}, "
              f"Y={end_effector_position[1]:.3f}, Z={end_effector_position[2]:.3f}")
        return end_effector_position

    def inverse_kinematics_optimization(self, target_position):
        """Perform inverse kinematics optimization to achieve a target position."""
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
            
            # Display the final end-effector position
            self.get_end_effector_position()
        else:
            print("Optimization failed.")

# Example usage:
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
