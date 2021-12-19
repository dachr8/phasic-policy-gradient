from .graph_util import plot_experiment, switch_to_outer_plot
from .constants import ENV_NAMES, NAME_TO_CASE, HARD_GAME_RANGES
import matplotlib
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize_and_reduce', dest='normalize_and_reduce', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='ppg')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--n_run', default=1)
    parser.add_argument('--single_env_name', default="coinrun")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    n_run = args.n_run
    single_env_name = args.single_env_name

    main_pcg_sample_entry(experiment_name, args.normalize_and_reduce, n_run, single_env_name)

    plt.tight_layout()

    if args.save:
        suffix = '-mean' if args.normalize_and_reduce else '-' + single_env_name
        plt.savefig(f'tmp/{experiment_name}{suffix}.pdf')
    else:
        plt.show()


def main_pcg_sample_entry(experiment_name, normalize_and_reduce, n_run, single_env_name):
    params = {
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 24,
        'legend.fontsize': 18,
        'figure.figsize': [9, 9]
    }
    matplotlib.rcParams.update(params)

    kwargs = {
        'smoothing': .9,
        'x_scale': 4 * 256 * 64 / 1e6,  # num_workers * num_steps_per_rollout * num_envs_per_worker / graph_scaling
        'single_env_name': single_env_name
    }

    normalization_ranges = HARD_GAME_RANGES

    y_label = 'Score'
    x_label = 'Timesteps (M)'

    if experiment_name == 'ppo':
        kwargs['csv_file_groups'] = [[f'ppo-run{x}' for x in range(n_run)]]
    elif experiment_name == 'ppg':
        kwargs['csv_file_groups'] = [[f'ppg-run{x}' for x in range(n_run)]]
    elif experiment_name == 'e_pi':
        kwargs['csv_file_groups'] = [[f'e-pi-{x}'] for x in [1, 2, 3, 4, 5, 6]]
        kwargs['labels'] = [f"$E_\\pi$ = {x}" for x in [1, 2, 3, 4, 5, 6]]
    elif experiment_name == 'e_aux':
        kwargs['csv_file_groups'] = [[f'e-aux-{x}'] for x in [1, 2, 3, 6, 9]]
        kwargs['labels'] = ["$E_{aux}$ = " + str(x) for x in [1, 2, 3, 6, 9]]
    elif experiment_name == 'n_pi':
        kwargs['csv_file_groups'] = [[f'n-pi-{x}'] for x in [2, 4, 8, 16, 32]]
        kwargs['labels'] = ["$N_\pi$ = " + str(x) for x in [2, 4, 8, 16, 32]]
    elif experiment_name == 'ppgkl':
        kwargs['csv_file_groups'] = [[f'ppgkl-run{x}' for x in range(n_run)]]
    elif experiment_name == 'ppg_single_network':
        kwargs['csv_file_groups'] = [[f'ppgsingle-run{x}' for x in range(n_run)]]
    else:
        assert False, f"experiment_name {experiment_name} is invalid"

    # We throw out the first few datapoints to give the episodic reward buffers time to fill up
    # Otherwise, there could be a short-episode bias
    kwargs['first_valid'] = 10

    if normalize_and_reduce:
        kwargs['normalization_ranges'] = normalization_ranges
        y_label = 'Mean Normalized Score'

    fig, axarr = plot_experiment(**kwargs)

    if normalize_and_reduce or single_env_name:
        axarr.set_xlabel(x_label, labelpad=20)
        axarr.set_ylabel(y_label, labelpad=20)
    else:
        ax0 = switch_to_outer_plot(fig)
        ax0.set_xlabel(x_label, labelpad=40)
        ax0.set_ylabel(y_label, labelpad=35)


if __name__ == '__main__':
    main()
