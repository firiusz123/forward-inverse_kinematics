import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kinematics:
    def __init__(self):
        self.Table = []
        self.Matrices = []
        self.positions = []
        self.target_position=[]
        self.mask= []
        self.indexes_to_optimize=[]
        self.list_of_parameters = []

       #####################    BASIC FUNCTIONS ################################### 

    def Transform_matrix(self, theta, d, a, alfa):
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alfa), np.sin(alfa)*np.sin(theta), a*np.cos(theta)],
                       [np.sin(theta), np.cos(alfa)*np.cos(theta), -np.sin(alfa)*np.cos(theta), a*np.sin(theta)],
                       [0, np.sin(alfa), np.cos(alfa), d],
                       [0, 0, 0, 1]])
        return T

    def add_values(self, dh_parameters):
        self.Table.append(dh_parameters)
    def set_target_position(self,target_position):
        self.target_position = target_position
    def add_mask(self,mask_setup):
        self.mask.append(mask_setup)
    def show_mask(self):
        print(np.array(self.mask))
    def get_indexes_to_optimize(self):
        for i in range(len(self.mask)):
            #print(self.mask[i])
            for j in range(len(self.mask[i])):
                #print(self.mask[i][j])
                if self.mask[i][j] == 1 :
                    self.indexes_to_optimize.append([i,j])
        print(self.indexes_to_optimize)

    def get_variables_to_optimize(self):
        self.list_of_parameters=[]
        for i in self.indexes_to_optimize:
            self.list_of_parameters.append(self.Table[i[0]][i[1]])
        print(self.list_of_parameters)

    def change_variables_to_optimize(self,D1_array):
        self.list_of_parameters=[]
        for i in D1_array:
            self.list_of_parameters.append(self.Table[i[0]][i[1]])
            print(self.list_of_parameters)

   
    def swap_table_variables(self):
        k = 0 
        for i in (self.indexes_to_optimize):
            self.Table[i[0]][i[1]] = self.list_of_parameters[k]
            #print('those are i and j',[i[0],  i[1]])
            #print("this is k " ,k)
            #print('the value to be put into the table ' ,   self.list_of_parameters[k])
            k=k+1
            
        

    def show_table(self, angles_in_degrees=True):
        header = [" Theta", "d", "a", "Alfa"]
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
                if angles_in_degrees and i == 0 or i ==3:
                    value = np.degrees(value)
                print(f"{value:8.3f} |", end=" ")
            print()
        print("+---------" * len(header) + "+")

    def get_transformed_values(self):
        k = []
        for i in self.Table:
            T = Kinematics.Transform_matrix(self, i[0], i[1], i[2], i[3])
            k.append(T)
        self.Matrices = k

    def geting_positions(self):
        self.positions = []
        m = np.array(self.Matrices)
        self.positions.append([0, 0, 0])
        self.positions.append(m[0][:3, 3])
        for i in range(len(m)-1):
            result = np.linalg.multi_dot(m[0:i+2])
            print(result[:3,3])
            self.positions.append(result[:3, 3])
 #####################    BASIC FUNCTIONS ################################### 





    def plotting_points(self):
        positions_array = np.array(self.positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], c='r', marker='o')
        ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], linestyle='-', color='b')
        for i, pos in enumerate(positions_array):
            ax.text(pos[0], pos[1], pos[2], f'Joint {i+1}', color='black', fontsize=8, ha='right')

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
    def inverse_kinematics_optimization(self):
        def objective_function(self):
            k.get_indexes_to_optimize()
            k.get_variables_to_optimize()
            k.swap_table_variables()
            self.get_transformed_values()
            end_f_matrix=np.linalg.multi_dot(self.Matrices)
            end_effector_position = end_f_matrix[:3, 3]
            error = np.linalg.norm(end_effector_position - self.target_position)
            print(error)
        objective_function(self)
""" 
    def inverse_kinematics_optimization(self):
        k.get_indexes_to_optimize()
        k.get_variables_to_optimize(self.indexes_to_optimize)
        def objective_function(params):
            self.list_of_parameters = params
            Kinematics.swap_table_variables(self)
            self.get_transformed_values()
            end_f_matrix=np.linalg.multi_dot(self.Matrices)
            end_effector_position = end_f_matrix[:3, 3]
            error = np.linalg.norm(end_effector_position - self.target_position)
            print(error.shape , 'kurwa shape ')
            return error
        #bounds = [(0, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        initial_guess = self.list_of_parameters
        bounds = [(None, None), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', tol=1e-4 , bounds=bounds)
        print(result)
        if result.success:
            optimized_params = result.x
            self.indexes_to_optimize = optimized_params
            swap_table_variables(self.indexes_to_optimize)
            self.get_transformed_values()
            self.geting_positions()
            self.plotting_points()
            self.show_table()
        else:
            print("Optimization failed.") """


# Example usage:
# Target end-effector position
target_position = np.array([300, 100, 200] )


k = Kinematics()

# Define DH parameters
k.add_values([0, 100, 0, 0])
k.add_mask(  [0 , 1 , 0 , 0])

k.add_values([0, 0, 100, np.radians(90)])
k.add_mask(  [0 , 0 , 0 ,0])

k.add_values([np.radians(0), 0, 100, 0])
k.add_mask([1 , 0 , 0 , 0])

k.add_values([np.radians(0), 0, 100, 0])
k.add_mask([1 , 0 , 0 , 0])

k.add_values([np.radians(0), 0, 100, 0])
k.add_mask([1 , 0 , 0 , 0])

#k.show_mask()





k.set_target_position(target_position)

k.get_transformed_values()
k.inverse_kinematics_optimization()
