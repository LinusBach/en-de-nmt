import matplotlib.pyplot as plt
import os


def plot(points, step_size, plots_dir="plots", model_name="model"):
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    plt.figure()
    plt.plot(list(range(step_size, (1 + len(points)) * step_size, step_size)), points)
    plt.savefig("plots/" + model_name + ".png")
    plt.close()
