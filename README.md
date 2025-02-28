The main idea of project is to make forward kinematics and inverse kinematics using denavit hartenberg method of calculating the displacment and rotation of parts.


# Example usage:
# Target end-effector position

adding the desired position of end effector
target_position = np.array([-80, 450, 0] )



k = Kinematics()

# Define DH parameters
here you can add parameters as follows z theta alfa x  
  z - displament in z axis
  theta - rotaation in z axis 
  alfa - rotation in x axis
  x - displecment in x axis 

  
k.add_values([0, 0, 70, 0])
k.add_values([0, 0, 200, 0])
k.add_values([-np.pi, 0, 250, 0])
k.add_values([np.radians(0), 0, 0, np.pi])
k.add_values([np.radians(0), 0, 0, 0])


k.get_transformed_values()

k.inverse_kinematics_optimization(target_position)
