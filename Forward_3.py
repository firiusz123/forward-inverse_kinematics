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
            print(np.round(i,5))        
        
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
        n = 500
        ax.set_xlim([-n, n])
        ax.set_ylim([-n, n])
        ax.set_zlim([-n, n])
        ax.quiver(0, 0, 0, 4, 0, 0, color='g', label='X-axis')
        ax.quiver(0, 0, 0, 0, 4, 0, color='b', label='Y-axis')
        ax.quiver(0, 0, 0, 0, 0, 4, color='y', label='Z-axis')
        ax.view_init(elev=30, azim=-75)
        plt.legend()
        
        plt.show()          




#Parameters:
a = 100 
theta1 = np.radians(45)
theta2 = np.radians(45)
theta3 = np.radians(-90)
k = Kinematics()
k.add_values([0, a, 0, 0])
k.add_values([0, 0, 100, np.radians(60)])
k.add_values([theta1 , 0 , 120 , 0])
k.add_values([theta2 , 0 , 120 , 0])
k.add_values([theta3 , 0 , 140 , 0])
k.get_transformed_values()
k.geting_positions()
k.plotting_points()
k.show_table()
 

