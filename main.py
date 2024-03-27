import numpy as np

import matplotlib.pyplot as plt

def draw_graph(m, b):
    x = np.linspace(-10, 10, 100)
    y = m * x + b

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of y = mx + b')
    plt.grid(True)
    plt.show()

# Example usage
m = 2
b = 3
draw_graph(m, b)