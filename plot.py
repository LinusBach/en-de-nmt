import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# plt.switch_backend('agg')


def show_plot(points, model_name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("plots/" + model_name + "loss.png")
