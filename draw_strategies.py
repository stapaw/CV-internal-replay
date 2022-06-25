from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from models.plr_strategy import get_plr_frequencies

sns.set_style("whitegrid")


def get_weight_counts(layer_sizes, n_classes):
    return [first * second + second for first, second in
            zip(layer_sizes, layer_sizes[1:] + [n_classes])]


def plot_strategies(strategies, layer_sizes, n_classes, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    weight_counts = get_weight_counts(layer_sizes, n_classes)

    data = []
    for strategy in strategies:
        update_frequencies = get_plr_frequencies(strategy, weight_counts=weight_counts,
                                                 layer_sizes=layer_sizes)
        for i, frequency in enumerate(update_frequencies):
            data.append({
                "frequency": frequency,
                "layer": i,
                "strategy": strategy
            })
    data = pd.DataFrame(data)

    plt.cla()
    plt.clf()
    plt.figure()
    barplot = sns.barplot(x="layer", y="frequency", data=data, hue="strategy")
    barplot.set_title(title)
    barplot.set_xlabel("Layer index")
    barplot.set_ylabel("Update frequency")
    barplot.set_ylim(0, 1.)
    barplot.get_figure().savefig(str(output_path))


if __name__ == "__main__":
    layer_sizes_4 = [1024, 500, 500, 500]
    layer_sizes_3 = [1024, 2000, 2000]
    n_classes = 100
    strategies = ["basic", "cumulative_weights", "total_weights", "input_size", "layer_idx"]

    plot_strategies(strategies, layer_sizes_3, 100, "strategy_plots/3.png",
                    "Update frequencies for CIFAR100 with 3 layers.")
    plot_strategies(strategies, layer_sizes_4, 100, "strategy_plots/4.png",
                    "Update frequencies for CIFAR100 with 4 layers.")
