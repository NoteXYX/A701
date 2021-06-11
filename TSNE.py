import matplotlib.pyplot as plt


def plot_with_labels(clusterJson, filename):
    plt.figure(figsize=(15, 15))  # in inches
    color_list = ['b', 'y', 'g', 'c', 'm']
    clusters = clusterJson['clusters']
    for label in list(clusters.keys()):
        for vec in clusters[label]:
            x, y = vec
            plt.scatter(x, y, c=color_list[label])
    centers = clusterJson['centers']
    for label in list(centers.keys()):
        x, y = centers[label]
        plt.scatter(x, y, c='r')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)

