import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = {
    'Conv2_winning': pd.read_csv('results/Conv2_dropoutFalse_winning_ticket.csv'),
    'Conv2_random': pd.read_csv('results/Conv2_dropoutFalse_random_reinit.csv'),
    'Conv4_winning': pd.read_csv('results/Conv4_dropoutFalse_winning_ticket.csv'),
    'Conv4_random': pd.read_csv('results/Conv4_dropoutFalse_random_reinit.csv'),
    'Conv6_winning': pd.read_csv('results/Conv6_dropoutFalse_winning_ticket.csv'),
    'Conv6_random': pd.read_csv('results/Conv6_dropoutFalse_random_reinit.csv')
}

# Plotting scale to somewhat evenly plot the datapoints
def forward(x):
    return np.power(x, 0.5)  # (x, 0.8)
# Inverse to get back to original
def inverse(x):
    return np.power(x, 2.0)

# Plotting (not the devious kind)
fig = plt.figure(figsize=(8, 12))
gs = fig.add_gridspec(3, 1, hspace=0.2) # 2,2

# Adaptive colors and markers
colors = {'Conv2': "#cf2114", 'Conv4': "#a11ca1", 'Conv6': "#2074f1"}
markers = {'winning': 'D', 'random': 'x'}
msizes = {'winning': 6, 'random': 6}
linestyles = {'winning': '-', 'random': '--'}

# Axes for the plots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Plot configurations
plot_configs = [
    (ax1, 'test_accuracy', 'Test accuracy at $t_\mathrm{ES}$ (in %)'),
    (ax2, 'train_accuracy', 'Train accuracy at $t_\mathrm{ES}$ (in %)'),
    (ax3, 'early_stop_iteration', '$t_\mathrm{ES}$ (iterations)')
]

# Loop through each plot
first_run = True
for ax, y_col, ylabel in plot_configs:
    # Loop through each data 
    for model in ['Conv2', 'Conv4', 'Conv6']:
        for exp_type in ['winning', 'random']:
            key = f'{model}_{exp_type}'
            if key in data:
                df = data[key]
                label = f'{model} {"Winning ticket" if exp_type == "winning" else "Random-reinit."}'
                
                if ax == ax1:
                    # ax.set_yticks([62, 66, 70, 74, 78, 82])
                    ax.set_ylim(58, 83)
                elif ax == ax2:
                    ax.set_ylim(67, 97)
                elif ax == ax3:
                    ax.set_yticklabels(['2.5K', '5K', '7.5K', '10K', '12.5K', '15K', '17.5K', '20K'])

                if exp_type == 'winning':
                    ax.plot(df['remaining_percent'], df[y_col],
                           marker=markers[exp_type], color=colors[model], label=label if first_run else "",
                           linestyle=linestyles[exp_type], linewidth=2, markersize=msizes[exp_type],
                           markeredgewidth=1.0, markeredgecolor='black')
                else:
                    ax.plot(df['remaining_percent'], df[y_col],
                           marker=markers[exp_type], color=colors[model],
                           label=label if first_run else "",
                           linestyle=linestyles[exp_type], linewidth=2, markersize=msizes[exp_type], alpha=0.7)
    first_run = False  # Only label once for legend
    ax.tick_params(which='major', axis='both',
                    direction='in', length=4, width=.5)
    ax.set_xlabel('Weights remaining (in %)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), frameon=False, fontsize=11, ncols=3)
    ax.grid(alpha=0.3)
    ax.set_xscale('function', functions=(forward, inverse))
    ax.set_xticks([100, 80, 60, 40, 20, 5])
    ax.invert_xaxis()
    ax.set_xlim(100, 3)

plt.savefig('lottery_ticket_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved")