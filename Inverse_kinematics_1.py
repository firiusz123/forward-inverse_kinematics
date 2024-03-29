import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kinematics:
    def __init__(self):
        self.Table = []
        self.Matrices = []
        self.positions=[]

    def Transform_matrix(self,theta , d , a , alfa):
      T =np.array([[np.cos(theta), -np.sin(theta)*np.cos(alfa), np.sin(alfa)*np.sin(theta), a*np.cos(theta)],
            [np.sin(theta), np.cos(alfa)*np.cos(theta), -np.sin(alfa)*np.cos(theta), a*np.sin(theta)],
            [0, np.sin(alfa), np.cos(alfa), d],
            [0, 0 ,0 ,1]])
      return T

    def add_values(self, dh_parameters):
        self.Table.append(dh_parameters)
    def show_table(self):
        header = ["Theta", "d", "a", "Alfa"]
        print("+---------" * len(header) + "+")
        print("|", end=" ")
        for h in header:
            print(f"{h:8}", end=" | ")
        print()
        print("+---------" * len(header) + "+")

        for row in self.Table:
            print("|", end=" ")
            for value in row:
                print(f"{value:8.3f} |", end=" ")
            print()
        print("+---------" * len(header) + "+")
    def get_transformed_values(self):
        k = []
        for i in self.Table:
            T = Kinematics.Transform_matrix(self,i[0] , i[1] , i[2] , i[3])
            k.append(T)
        self.Matrices = k
        """ for i in k:
            print(i) """
    def geting_positions(self):
        self.positions=[]
        m = np.array(self.Matrices)
        self.positions = []
        self.positions.append([0 , 0 , 0 ])
        self.positions.append(m[0][:3,3])
        for i in range(len(m)-1):
            result = np.linalg.multi_dot(m[0:i+2])
            self.positions.append(result[:3,3])
        for i in self.positions:
            print(i)        
        
    def plotting_points(self):
        positions_array = np.array(self.positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        ax.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], c='r', marker='o')
        # Plot lines connecting the points
        ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], linestyle='-', color='b')
        for i, pos in enumerate(positions_array):
            ax.text(pos[0], pos[1], pos[2], f'Joint {i+1}', color='black', fontsize=8, ha='right')
        # Set labels for axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        n = 20
        ax.set_xlim([-n, n])
        ax.set_ylim([-n, n])
        ax.set_zlim([-n, n])
        ax.quiver(0, 0, 0, 4, 0, 0, color='g', label='X-axis')
        ax.quiver(0, 0, 0, 0, 4, 0, color='b', label='Y-axis')
        ax.quiver(0, 0, 0, 0, 0, 4, color='y', label='Z-axis')
        ax.view_init(elev=30, azim=-75)
        plt.legend()
        # Show the plot
        plt.show()  
######################### end of forward  inverse BEGINS #################
    def objective_fggunction(self,*parameters) :
         pass  
    def inverse_kinematics_optimization(self, target_position):
        def objective_function(params):
            a, theta1, theta2, theta3 = params
            self.Table[0][1] = a
            self.Table[2][0] = theta1
            self.Table[3][0] = theta2
            self.Table[4][0] = theta3
            self.get_transformed_values()
            end_effector_position = self.Matrices[-1][:3, 3]
            error = np.linalg.norm(end_effector_position - target_position)
            return error

        initial_guess = [self.Table[0][1], self.Table[2][0], self.Table[3][0], self.Table[4][0]]
        result = minimize(objective_function, initial_guess, method='SLSQP')

        if result.success:
            optimized_params = result.x
            self.Table[0][1], self.Table[2][0], self.Table[3][0], self.Table[4][0] = optimized_params
            self.get_transformed_values()
            self.geting_positions()
            self.plotting_points()
            self.show_table()
        else:
            print("Optimization failed.")




#Parameters:
a = 4
theta1 = np.radians(45)
theta2 = np.radians(-0)
theta3 = np.radians(0)

# Target end-effector position
target_position = np.array([4, 0, 22])

# Create Kinematics object
k = Kinematics()

# Define DH parameters
k.add_values([0, a, 0, 0])
k.add_values([0, 0, 4, np.radians(90)])
k.add_values([theta1, 0, 6, 0])
k.add_values([theta2, 0, 6, 0])
k.add_values([theta3, 0, 6, 0])

# Forward kinematics to initialize Matrices
k.get_transformed_values()

# Use inverse kinematics with optimization to achieve the target position
k.inverse_kinematics_optimization(target_position)
 

