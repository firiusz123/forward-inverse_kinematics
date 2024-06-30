import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kinematics:
    def __init__(self):
        self.Table = []
        self.Matrices = []
        self.positions_matrices = []
        self.positions = []
        self.optimazing_mask = []
        self.indexes_to_optimize = []

    def Transform_matrix(self, theta, d, a, alfa):
        T = torch.tensor([
            [torch.cos(theta), -torch.sin(theta) * torch.cos(alfa), torch.sin(alfa) * torch.sin(theta), a * torch.cos(theta)],
            [torch.sin(theta), torch.cos(theta) * torch.cos(alfa), -torch.sin(alfa) * torch.cos(alfa), a * torch.sin(theta)],
            [0, torch.sin(alfa), torch.cos(alfa), d],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        return T

    def add_values(self, dh_parameters):
        self.Table.append(dh_parameters)

    def add_optimazing_mask(self, mask_optimize):
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
                    value = torch.degrees(value).item()
                print(f"{value:8.3f} |", end=" ")
            print()
        print("+---------" * len(header) + "+")

    def get_transformed_values(self):
        self.Matrices = [self.Transform_matrix(*params) for params in self.Table]
        return self.Matrices

    def get_positions_matrices(self):
        self.positions_matrices = [torch.eye(4, dtype=torch.float32)]  # Start with the identity matrix
        for matrix in self.Matrices:
            matrix_multiply = torch.mm(self.positions_matrices[-1], matrix)
            self.positions_matrices.append(matrix_multiply)

    def get_positions(self):
        self.positions = [mat[:3, 3] for mat in self.positions_matrices]
        self.positions = torch.stack(self.positions)

    def plot_mechanism(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = self.positions[:, 0].numpy()
        y = self.positions[:, 1].numpy()
        z = self.positions[:, 2].numpy()

        # Plot points in red
        ax.scatter(x, y, z, color='red', s=100)

        # Plot lines in blue
        ax.plot(x, y, z, color='blue')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def get_optimazing_mask(self):
        for i in range(len(self.optimazing_mask)):
            for j in range(len(self.optimazing_mask[i])):
                if self.optimazing_mask[i][j] == 1:
                    self.indexes_to_optimize.append([i, j])

    def get_params_to_optimize(self):
        params = []
        for row, column in self.indexes_to_optimize:
            params.append(self.Table[row][column])
        return params

    def swap_parameters(self, values):
        if len(values) != len(self.indexes_to_optimize):
            print("Amount of parameters is not equal to amount of indexes of parameters, internal error")
        else:
            index = 0
            for row, column in self.indexes_to_optimize:
                self.Table[row][column] = values[index]
                index += 1

    def objective_function(self, parameters, target_position):
        self.swap_parameters(parameters)
        self.get_transformed_values()
        self.get_positions_matrices()
        self.get_positions()
        end_effector_position = self.positions[-1]
        loss = torch.sum((end_effector_position - target_position) ** 2)  # Mean Squared Error
        return loss

    def inverse(self, target_position, num_epochs=100, learning_rate=0.01):
        # Convert parameters to torch tensors
        params = self.get_params_to_optimize()
        parameters_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=True)

        # Define the optimizer
        optimizer = optim.SGD([parameters_tensor], lr=learning_rate)

        # Optimization loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.objective_function(parameters_tensor, target_position)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Update the Table with the optimized parameters
        self.swap_parameters(parameters_tensor.detach().numpy())

# Example usage
kinematics = Kinematics()

# Add DH parameters for each joint (theta, d, a, alfa)
kinematics.add_values([0, 0, 5, 0])
kinematics.add_optimazing_mask([0, 0, 1, 0])
kinematics.add_values([0, 3, 0, 0])
kinematics.add_optimazing_mask([0, 1, 0, 0])

# Get the optimization mask
kinematics.get_optimazing_mask()

# Get the transformed values and positions
kinematics.get_transformed_values()
kinematics.get_positions_matrices()
kinematics.get_positions()

# Define the target position for the end effector
target_position = torch.tensor([10.0, 5.0, 3.0], dtype=torch.float32)

# Perform the optimization
kinematics.inverse(target_position)

# Show the updated DH parameters
kinematics.show_table()

# Plot the mechanism
kinematics.plot_mechanism()
