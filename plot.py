"""
This script provides a function for creating a plot of the training loss during model training.
The plot function generates a line plot, where the x-axis represents the number of training steps,
and the y-axis represents the training loss. The plot is saved as a PNG file in a specified directory.
"""


import matplotlib.pyplot as plt
import os
from typing import List


def plot(points: List[float], step_size: int, plots_dir: str = "plots", model_name: str = "model"):
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    plt.figure()
    plt.plot(list(range(step_size, (1 + len(points)) * step_size, step_size)), points)
    plt.savefig("plots/" + model_name + ".png")
    plt.close()
