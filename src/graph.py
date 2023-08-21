import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


def graph(PATH):
    # Load the data
    data = pd.read_csv(PATH)
    data['時刻'] = pd.to_datetime(data['時刻'])

    # Initialize the plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)

    # Plot 体表温度
    axes[0].plot(data['時刻'], data['体表温度'], label='体表温度', color='blue')
    axes[0].set_ylabel('体表温度')
    axes[0].legend(loc='upper right')

    # Plot 体動
    axes[1].plot(data['時刻'], data['体動'], label='体動', color='green')
    axes[1].set_ylabel('体動')
    axes[1].legend(loc='upper right')

    # Plot 脈周期[ms]
    axes[2].plot(data['時刻'], data['脈周期[ms]'], label='脈周期[ms]', color='red')
    axes[2].set_ylabel('脈周期[ms]')
    axes[2].set_xlabel('時刻')
    axes[2].legend(loc='upper right')

    # Shade regions based on ラベル values
    labels = data['ラベル'].unique()
    colors = ['grey', 'yellow', 'orange', 'purple']

    for i, ax in enumerate(axes):
        for j, label in enumerate(labels):
            mask = data['ラベル'] == label
            ax.fill_between(data['時刻'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask, color=colors[j], alpha=0.2, label=f'ラベル {label}')

    # Remove duplicate legends
    handles, labels = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set title and show plot
    plt.suptitle('Graphs of 体表温度, 体動, and 脈周期[ms] against 時刻', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
