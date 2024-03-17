import matplotlib.pyplot as plt
import numpy as np
import time

def update_graph():
    # Generate some data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    
    # Clear previous plot
    plt.clf()
    
    # Plot the updated data
    plt.plot(x, y, color='b')
    plt.title('Updated Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Show the plot
    plt.show()

def update_graph_new():
    # Generate some new data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    
    # Clear previous plot
    plt.clf()
    
    # Plot the updated data
    plt.plot(x, y, color='r')  # Using a different color for the new data
    plt.title('Updated Graph (New)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Show the plot
    plt.show()

# Invoke the first function to update the graph
update_graph()

# Pause for 2 seconds
time.sleep(2)

# Invoke the second function to update the graph with different data
update_graph_new()
