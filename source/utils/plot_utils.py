import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_graph(vector,
                   label="NoName",
                   color="blue",
                   title="Title",
                   x_label = 'Frames',
                   y_label = "y_labels",
                   figsize=(20,5)):

    fig = plt.figure(figsize=figsize)
    x = np.arange(0, len(vector))
    plt.plot(x, vector, label=label, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.ylim(y_val_min, y_val_max)
    fig.canvas.draw()
    plot_image = fig.canvas.renderer._renderer
    image_numpy = np.array(plot_image)
    plt.clf()
    plt.close()
    return image_numpy


def generate_graph_overwrite(vector1, #gt
                             vector2, #pd
                             title="Title",
                             x_label = 'Frames',
                             y_label = "y_labels",
                             figsize=(20,5)):

    fig = plt.figure(figsize=figsize)
    x = np.arange(0, len(vector1))
    plt.plot(x, vector1, label="GroundTruth", color="red")
    plt.plot(x, vector2, label="Generated", color="blue")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.ylim(y_val_min, y_val_max)
    fig.canvas.draw()
    plot_image = fig.canvas.renderer._renderer
    image_numpy = np.array(plot_image)
    plt.clf()
    plt.close()
    return image_numpy