import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils.functions import funcs_to_str, get_functions


def plot_2d_by_func(axs, ode_func, betas):
    t = np.linspace(0, 10, 100)
    IC = np.linspace(0.9, 1.8, 10)

    for x in IC:
        X0 = [x, x]
        Xs = solve_ivp(ode_func, (t[0], t[-1]), X0, args=betas, t_eval=t, method='BDF').y.T
        axs.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{str(round(X0[0], 2))},{str(round(X0[1], 2))}]")
    axs.set_title("Target Phase-space by IC")
    axs.set_xlabel("Preys")
    axs.set_ylabel("Predators")
    axs.legend()


def plot_2d_by_y(axs, x0, ys, labels):
    for y, label in zip(ys, labels):
        axs.plot(y[:, 0], y[:, 1], "-", label=label)

    axs.plot(x0[0], x0[1], 'ro')
    axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0] + 0.5, x0[1] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=1))

    x_min, x_max = ys[0][:, 0].min(), ys[0][:, 0].max()
    y_min, y_max = ys[0][:, 1].min(), 2  # ys[0][:, 1].max()
    axs.set_xlim(x_min - x_max * 0.1, x_max * 1.1)
    axs.set_ylim(y_min - y_max * 0.1, y_max * 1.1)

    axs.set_title("Best Estimate")
    axs.set_xlabel("Preys")
    axs.set_ylabel("Predators")
    # axs.legend()


def plot_3d_estimates(axs, t, x0, ys, results, labels, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # Use a colormap for consistent colors

    for result, label, color in zip(results, labels, colors):
        axs.plot(t, result[:, 0], "-", label=label, alpha=0.8, color=color)
        axs.plot(t, result[:, 1], "-", alpha=0.8, color=color)
        axs.plot(t, result[:, 2], "-", alpha=0.8, color=color)

    axs.plot(t, ys[:, 0], "-", label="SIR", color='black')
    axs.plot(t, ys[:, 1], "-", label="SIR", color='black')
    axs.plot(t, ys[:, 2], "-", label="SIR", color='black')
    axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0], x0[1]))

    x_min, x_max = t.min(), t.max()
    y_min = ys.min()
    y_max = ys.max()

    y_padding = 0.1 * (y_max - y_min)  # 10% of the range
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min - y_padding, y_max + y_padding)

    axs.set_title(title)
    axs.set_xlabel("Time (t)")
    axs.set_ylabel("SIR Populations")
    # axs.legend(loc='upper right')


def plot_2d_estimates_by_y(axs, x0, ys, results, labels, title):
    for result, label in zip(results, labels):
        axs.plot(result[:, 0], result[:, 1], "-", label=label, alpha=0.8)

    axs.plot(ys[:, 0], ys[:, 1], "-", label="Lotka-Volterra", color='black')
    axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0], x0[1]))

    x_min, x_max = ys[:, 0].min(), ys[:, 0].max()
    y_min, y_max = ys[:, 1].min(), ys[:, 1].max()
    axs.set_xlim(x_min - x_max * 0.1, x_max * 1.1)
    axs.set_ylim(y_min - y_max * 0.1, y_max * 1.1)

    axs.set_title(title)
    axs.set_xlabel("Preys")
    axs.set_ylabel("Predators")
    # axs.legend(loc='upper right')


def plot_min_loss_by_iteration(axs, results):
    sorted_results = sorted(results, key=lambda result: len(funcs_to_str(get_functions(result['param']['f0ps']))))

    for result in sorted_results:
        axs.plot(np.arange(len(result['min_loss'])) + 1, result['min_loss'],
                 label=funcs_to_str(get_functions(result['param']['f0ps'])), linestyle='-', alpha=0.8)
    axs.plot([], [], label='SIR (Noisy)', color='black', linestyle='-', linewidth=2)

    # Configure the plot
    axs.set_xlabel("Generation")
    axs.set_ylabel("Minimum Loss")
    axs.set_title("Minimum Loss Over Generations")

    # lines, labels = axs.get_legend_handles_labels()
    # # Ensure "Lotka-Volterra" appears last
    # sorted_indices = sorted(range(len(labels)), key=lambda i: (labels[i] == 'Lotka-Volterra', len(labels[i])))
    # sorted_lines = [lines[i] for i in sorted_indices]
    # sorted_labels = [labels[i] for i in sorted_indices]

    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend(loc='upper right')


def plot_avg_loss_by_iteration(axs, results):
    for result in results:
        axs.plot(np.arange(len(result['avg_loss'])) + 1, result['avg_loss'],
                 label=funcs_to_str(get_functions(result['param']['f0ps'])), linestyle='-', alpha=0.8)

    axs.set_xlabel("Generation")
    axs.set_ylabel("Average Loss")
    # axs.tick_params(axis='y', labelcolor='blue')
    axs.set_title("Average Loss Over Generations")
    lines, labels = axs.get_legend_handles_labels()
    sorted_indices = sorted(range(len(labels)), key=lambda i: len(labels[i]))
    sorted_lines = [lines[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    # axs.legend(sorted_lines, sorted_labels, loc='upper right')


def plot_time_per_generation(axs, results):
    for history in results:
        if 'time_records' in history[1][0]:
            axs.plot(np.arange(len(history[1][0]['time_records'])), history[1][0]['time_records'], marker='o',
                     label=funcs_to_str(get_functions(history[1][0]['f0ps'])), linestyle='-', alpha=0.8)

    axs.set_xlabel("Generation #")
    axs.set_ylabel("Time (s)")
    axs.set_title("Time per generation")
    lines, labels = axs.get_legend_handles_labels()
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend(lines, labels, loc='upper right')


def plot_invalid_counts_by_iteration(axs, results):
    for result in results:
        axs.plot(np.arange(len(result['invalid'])), result['invalid'], marker='o',
                 label=funcs_to_str(get_functions(result['param']['f0ps'])), linestyle='-', alpha=0.8)

    axs.set_xlabel("Generation")
    axs.set_ylabel("Invalid Count")
    axs.set_title("Invalid Count Over Generations")
    lines, labels = axs.get_legend_handles_labels()
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend(lines, labels, loc='upper right')


def plot_loss_by_iteration(axs, min_loss, avg_loss):
    axs.plot(np.arange(len(min_loss)), min_loss, marker='o', label='Minimum Loss', linestyle='-', alpha=0.8)
    axs.set_xlabel("Generation")
    axs.set_ylabel("Min Loss", color='blue')
    axs.tick_params(axis='y', labelcolor='blue')

    ax2 = axs.twinx()
    ax2.plot(np.arange(len(avg_loss)), avg_loss, marker='s', label='Average Loss', linestyle='--', alpha=0.8,
             color='red')
    ax2.set_ylabel("Average Loss", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    axs.set_title("Loss Over Generations")
    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend(lines + lines2, labels + labels2, loc='upper right')


def plot_invalid_by_iteration(axs, invalid):
    axs.plot(range(len(invalid)), invalid, marker='o', linestyle='-', color='r', alpha=0.8, label="Invalid Systems")
    axs.set_title("Invalid Systems Over Generations")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Number of Invalid Entries")
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend()


def plot_3d_by_y(axs, t, y_target, y_best, labels):
    axs.plot(t, y_target[:, 0], label='Target Data (Variable 1)')  # 'ro'
    axs.plot(t, y_target[:, 1], label='Target Data (Variable 2)')  # 'ro'
    axs.plot(t, y_target[:, 2], label='Target Data (Variable 3)')  # 'ro'
    for y, label in zip(y_best, labels):
        axs.plot(t, y[:, 0], label=label + ' (Variable 1)')  # 'ro'
        axs.plot(t, y[:, 1], label=label + ' (Variable 2)')  # 'ro'
        axs.plot(t, y[:, 2], label=label + ' (Variable 3)')  # 'ro'

    # axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0] + 0.5, x0[1] + 0.5),
    #              arrowprops=dict(facecolor='black', shrink=1))

    axs.set_title('Time Series Plot For each variable')
    axs.set_xlabel('Time')
    axs.set_ylabel('Value')
    axs.legend()


#FOLLOWING: FOR LORENZ


def plot_lorenz_2d_by_func(axs, ode_func, betas):
    t = np.linspace(0, 10, 100)
    IC = np.linspace(0.9, 1.8, 10)

    for x in IC:
        X0 = [x, x, x]  # Starting point for 3D Lorenz system
        Xs = solve_ivp(ode_func, (t[0], t[-1]), X0, args=betas, t_eval=t, method='BDF').y.T
        axs.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{round(X0[0], 2)},{round(X0[1], 2)},{round(X0[2], 2)}]")
    axs.set_title("Lorenz Phase-space by IC")
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    axs.legend()

def plot_lorenz_2d_by_y(axs, x0, ys, labels):
    for y, label in zip(ys, labels):
        axs.plot(y[:, 0], y[:, 1], "-", label=label)

    axs.plot(x0[0], x0[1], 'ro')
    axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0] + 0.5, x0[1] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=1))

    x_min, x_max = ys[0][:, 0].min(), ys[0][:, 0].max()
    y_min, y_max = ys[0][:, 1].min(), ys[0][:, 1].max()
    axs.set_xlim(x_min - x_max * 0.1, x_max * 1.1)
    axs.set_ylim(y_min - y_max * 0.1, y_max * 1.1)

    axs.set_title("Best Estimate for Lorenz System")
    axs.set_xlabel("X")
    axs.set_ylabel("Y")

def plot_lorenz_3d_estimates(axs, t, x0, ys, results, labels, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # Use a colormap for consistent colors

    for result, label, color in zip(results, labels, colors):
        axs.plot(t, result[:, 0], "-", label=label, alpha=0.8, color=color)
        axs.plot(t, result[:, 1], "-", alpha=0.8, color=color)
        axs.plot(t, result[:, 2], "-", alpha=0.8, color=color)

    axs.plot(t, ys[:, 0], "-", label="Lorenz Solution", color='black')
    axs.plot(t, ys[:, 1], "-", color='black')
    axs.plot(t, ys[:, 2], "-", color='black')
    axs.annotate(f'IC:[{round(x0[0], 1)},{round(x0[1], 1)}]', xy=(x0[0], x0[1]), xytext=(x0[0], x0[1]))

    x_min, x_max = t.min(), t.max()
    y_min = ys.min()
    y_max = ys.max()

    y_padding = 0.1 * (y_max - y_min)  # 10% of the range
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min - y_padding, y_max + y_padding)

    axs.set_title(title)
    axs.set_xlabel("Time (t)")
    axs.set_ylabel("Lorenz States (X, Y, Z)")

def plot_lorenz_loss_by_iteration(axs, results):
    sorted_results = sorted(results, key=lambda result: len(funcs_to_str(get_functions(result['param']['f0ps']))))

    for result in sorted_results:
        axs.plot(np.arange(len(result['min_loss'])) + 1, result['min_loss'],
                 label=funcs_to_str(get_functions(result['param']['f0ps'])), linestyle='-', alpha=0.8)
    axs.plot([], [], label='Lorenz (Noisy)', color='black', linestyle='-', linewidth=2)

    axs.set_xlabel("Generation")
    axs.set_ylabel("Minimum Loss")
    axs.set_title("Minimum Loss Over Generations for Lorenz")

def plot_lorenz_invalid_by_iteration(axs, results):
    sorted_results = sorted(results, key=lambda result: len(funcs_to_str(get_functions(result['param']['f0ps']))))

    for result in sorted_results:
        axs.plot(np.arange(len(result['invalid'])) + 1, result['invalid'],
                 label=funcs_to_str(get_functions(result['param']['f0ps'])), linestyle='-', alpha=0.8)
    axs.plot([], [], label='Lorenz (Noisy)', color='black', linestyle='-', linewidth=2)

    axs.set_xlabel("Generation")
    axs.set_ylabel("Invalid Solutions")
    axs.set_title("Invalid Solutions Over Generations for Lorenz")

def plot_lorenz_results(history, t, x0, best, y_target):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Phase space plot for Lorenz system
    plot_lorenz_2d_by_func(axs[0, 0], best[2], best[0].x)
    plot_lorenz_2d_by_y(axs[0, 1], x0, [y_target], ["Noisy Target"])

    # Time-series plot for Lorenz system
    plot_lorenz_3d_estimates(axs[1, 0], t, x0, y_target, [best[2]], ["Best Estimate"], "Lorenz System Estimates")

    # Loss and invalidity plots
    plot_lorenz_loss_by_iteration(axs[1, 1], history)
    plot_lorenz_invalid_by_iteration(axs[1, 1], history)

    plt.tight_layout()
    plt.show()
