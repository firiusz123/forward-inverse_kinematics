import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kinematics:
    def __init__(self):
        self.Table = []
        self.Matrices = []
        self.positions_matrices = []
        self.positions = []
        self.optimazing_mask=[]
        self.indexes_to_optimize = []


    def Transform_matrix(self, theta, d, a, alfa):
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alfa), np.sin(alfa)*np.sin(theta), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alfa), -np.sin(theta)*np.cos(alfa), a*np.sin(theta)],
            [0, np.sin(alfa), np.cos(alfa), d],
            [0, 0, 0, 1]
        ])
        return T

    def add_values(self, dh_parameters):
        self.Table.append(dh_parameters)
    def add_optimazing_mask(self,mask_optimize):
        self.optimazing_mask.append(mask_optimize)

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
                if angles_in_degrees and (i == 0 or i == 3):
                    value = np.degrees(value)
                print(f"{value:8.3f} |", end=" ")
            print()
        print("+---------" * len(header) + "+")
        
    def get_transformed_values(self):
        k = []
        for i in self.Table:
            T = self.Transform_matrix(i[0], i[1], i[2], i[3])
            k.append(T)
        self.Matrices = k
        return k
    
    def get_positions_matrices(self):
        self.positions_matrices = [np.eye(4)]  # Start with the identity matrix
        for matrix in self.Matrices:
            matrix_multiply = np.dot(self.positions_matrices[-1], matrix)
            #print(matrix_multiply)
            self.positions_matrices.append(matrix_multiply)

    def get_positions(self):
        self.positions = []
        for i in self.positions_matrices:
            self.positions.append(i[:3, 3])
        self.positions = np.array(self.positions)
        
    def plot_mechanism(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        z = self.positions[:, 2]
        
        # Plot points in red
        ax.scatter(x, y, z, color='red' , s=100)
        
        # Plot lines in blue
        ax.plot(x, y, z, color='blue')
        #ax.set_xlim([-10, 10])
        #ax.set_ylim([-10, 10])
        #ax.set_zlim([-10, 10])
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
    def get_optimazing_mask(self):
        for i in range(len(self.optimazing_mask)):
            for j in range(len(self.optimazing_mask[i])):
                #print(i , j)
                if self.optimazing_mask[i][j] == 1:
                    self.indexes_to_optimize.append([i,j])

        #print((self.indexes_to_optimize))
    def get_params_to_optimize(self):
        params = []
        for row,column in self.indexes_to_optimize:
            params.append(self.Table[row][column])
            
        return params
    
    def swap_parameters(self,values):
        if len(values) != len(self.indexes_to_optimize):
            print("amount of parameters is not equal to amount of indexes of parameters , internal error ")
        else:
            index = 0
            for row,column in self.indexes_to_optimize:
                self.Table[row][column] =values[index]
                index = index + 1 
        #kinematics.show_table()
    def objective_function(self, parameters, target_position):
        self.swap_parameters(parameters)
        self.get_transformed_values()
        self.get_positions_matrices()
        self.get_positions()
        end_effector_position = self.positions[-1]
        loss = np.sum(np.square(end_effector_position - target_position))  # Mean Squared Error
        return loss

    def inverse(self, target_position,method='L-BFGS-B', num_epochs=100, learning_rate=0.01):
       kinematics.get_optimazing_mask()
       params_to_optimize = kinematics.get_params_to_optimize()
       result = minimize(kinematics.objective_function, params_to_optimize, args=(target_position,),method=method) 
       kinematics.swap_parameters(result.x)
       #kinematics.plot_mechanism()
       kinematics.show_table()
       return result.x, result
    
        

            
# Example usage
kinematics = Kinematics()

# Add DH parameters for each joint (theta, d, a, alfa)
kinematics.add_values([0, 0, 5, 0])
kinematics.add_optimazing_mask([0 , 0 , 1 , 0])
kinematics.add_values([0, 3, 0, 0])
kinematics.add_optimazing_mask([0 , 1 , 0 , 0])
kinematics.add_values([0, 20, 0, 0])
kinematics.add_optimazing_mask([0 , 1 , 0 , 0])
#kinematics.add_values([np.radians(45), 0, 5, 0])
#kinematics.add_optimazing_mask([1 , 0 , 0 , 0])
#kinematics.add_values([np.radians(45), 0, 5, 0])
#kinematics.add_optimazing_mask([1 , 0 , 0 , 0])
#kinematics.add_values([np.radians(30), 0, 5, 0])
#kinematics.add_optimazing_mask([1 , 0 , 1, 0])
#kinematics.add_values([np.radians(45), 0, 2, np.radians(0)])
#kinematics.add_values([np.radians(60), 1, 1, np.radians(-45)])

# Show DH parameters table
kinematics.show_table()

kinematics.get_optimazing_mask()

kinematics.get_transformed_values()

kinematics.get_positions_matrices()
kinematics.get_positions()
#kinematics.plot_mechanism()

kinematics.inverse([5,0,20])
