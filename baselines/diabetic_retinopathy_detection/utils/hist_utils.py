import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict
import os
from copy import deepcopy

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from baselines.diabetic_retinopathy_detection.utils import fast_load_dataset_to_model_results

lib = os.path.join(os.path.dirname(__file__), "../../..")
if lib not in sys.path:
    sys.path.insert(0, lib)

from baselines.diabetic_retinopathy_detection.utils.plot_utils import grid_plot_wrapper

tio = tf.io.gfile


ID_SINGLE_MODELS = [
    ('deterministic', 1, True, 'indomain', '5'),
    ('dropout', 1, False, 'indomain', '5'),
    ('fsvi', 1, False, 'indomain', '5'),
    ('rank1', 1, False, 'indomain', '5'),
    ('vi', 1, False, 'indomain', '5'),
    ('radial', 1, False, 'indomain', '5')
]

ID_ENSEMBLE_MODELS = [
    ('deterministic', 3, True, 'indomain', '5'),
    ('dropout', 3, False, 'indomain', '5'),
    ('fsvi', 3, False, 'indomain', '5'),
    ('rank1', 3, False, 'indomain', '5'),
    ('vi', 3, False, 'indomain', '5'),
    ('radial', 3, False, 'indomain', '5')
]


def get_name_to_severity_mapping(files, name_col="image", level_col="level"):
    d = {}
    for file_path in files:
        with tio.GFile(file_path) as f:
            df = pd.read_csv(f)
            assert df[name_col].unique().shape[0] == df.shape[0]
            to_update = {df[name_col].iloc[i]: df[level_col].iloc[i] for i in range(df.shape[0])}
            assert set(d.keys()).intersection(set(to_update.keys())) == set()
            d.update(to_update)
    return d


def partition_names(names, name_to_severity):
    severities = [name_to_severity[n.decode()] for n in names]
    partitions = defaultdict(list)
    for i, severity in enumerate(severities):
        partitions[severity].append(i)
    return partitions


def get_name_to_label_mapping() -> Dict[str, int]:
    # loading name to label mapping
    train_labels = "gs://ub-data/retinopathy/downloads/manual/trainLabels.csv"
    valid_and_test_labels = "gs://ub-data/retinopathy/downloads/manual/validationAndTestLabels.csv"
    name_to_severity = get_name_to_severity_mapping([train_labels, valid_and_test_labels], name_col="image", level_col="level")
    aptos_labels = "gs://ub-data/aptos/metadata.csv"
    aptos_mapping = get_name_to_severity_mapping([aptos_labels], name_col="id_code", level_col="diagnosis")
    assert len(aptos_mapping) == 3662
    name_to_severity.update(aptos_mapping)
    return name_to_severity


from matplotlib.ticker import PercentFormatter


def plot_result_grid(result, name_to_severity, i=0, bins=50, normalized=False, axes=None, alpha=1.0,
                     ncols=2, fontsize=7, model="", percentage=True):
    partitions = partition_names(result["names"][i], name_to_severity=name_to_severity)
    severities = sorted(partitions.keys())

    def _plot(ax, j):
        severity = severities[j]
        indices = partitions[severity]
        vals = result["y_pred_entropy"][i][indices]
        weights = np.ones(len(vals)) / len(vals) if percentage else None
        ax.hist(vals, bins=bins, density=normalized, alpha=alpha, weights=weights)
        ax.set_title(f"severity={severity}, {model}",
                     fontsize=fontsize)
        if percentage:
            ax.yaxis.set_major_formatter(PercentFormatter(1))

    kwargs = lambda j: {"j": j}
    axes = grid_plot_wrapper(_plot, n_plots=len(partitions), ncols=ncols,
                             get_kwargs=kwargs, axes=axes)
    return axes


def get_model_joint_results(dataset_to_model_results, model, datasets=["in_domain_test", "ood_test"]):
    fields = ["names", "y_pred_entropy", "accuracy_arr", "y_epistemic_uncert"]
    result = {}
    for field in fields:
        l = deepcopy(dataset_to_model_results[datasets[0]][model][field])
        for dataset in datasets[1:]:
            for i in range(len(l)):
                l[i] = np.concatenate([l[i], dataset_to_model_results[dataset][model][field][i]])
        result[field] = l
    return result


def get_sequential_colors(n, cmap_name="rainbow"):
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(v) for v in np.linspace(0, 1, n)]
    return colors


def plot_hist(ax, vals, percentage=True, kde=False, **kwargs):
    weights = np.ones(len(vals)) / len(vals) if percentage else None
    if kwargs.get("bins") == "custom":
        kwargs['bins'] = custom_number_bins(len(vals))
        # print(f"set bins to {kwargs['bins']}")
    hist = ax.hist(vals, weights=weights, **kwargs)
    # pdb.set_trace()
    if kde:
        add_kde(hist=hist, ax=ax, vals=vals)


def custom_number_bins(n):
    if n > 1000:
        return 40
    if n > 500:
        return 30
    return 15


DISPLAY_MODEL_NAME = {
    "deterministic": r"\textsc{MAP} (Deterministic)",
    "dropout": r"\textsc{MC DROPOUT}",
    "fsvi": r"\textsc{FSVI}",
    "rank1": r"\textsc{RANK}-1",
    "vi": r"\textsc{MFVI}",
    "radial": r"\textsc{RADIAL}-\textsc{MFVI}",
}


def add_kde(hist, ax, vals, n_kde=500):
    """
    https://github.com/timrudner/function-space-variational-inference/blob/master/plotting/plotting_entropy.py
    page 22 http://timrudner.com/papers/Rethinking_Function-Space_Variational_Inference_in_Bayesian_Neural_Networks/Rudner2021_Rethinking_Function-Space_Variational_Inference_in_Bayesian_Neural_Networks.pdf
    """
    _, bins, patches = hist
    color = patches[0].get_facecolor()
    alpha = patches[0].get_alpha()
    plotting_thresholds = np.linspace(bins[0], bins[-1], n_kde)

    def _fit(x):
        if x.ndim == 1:
            x = x[:, None]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(x)
        return np.exp(kde.score_samples(plotting_thresholds[:, None]))

    ax.plot(plotting_thresholds, _fit(vals), color=color, alpha=alpha)


def plot_result_grid_sep_accuracy(result, name_to_severity, field="y_pred_entropy",
                                  i=0, bins=50, normalized=False, axes=None, alpha=1.0,
                                  ncols=2, fontsize=7, model="", percentage=True, **kwargs):
    partitions = partition_names(result["names"][i], name_to_severity=name_to_severity)
    severities = sorted(partitions.keys())

    correct_indices = set([j for j, v in enumerate(result["accuracy_arr"][i]) if v])

    def _plot(ax, j):
        severity = severities[j]
        indices = partitions[severity]

        _correct = [ind for ind in indices if ind in correct_indices]
        _incorrect = [ind for ind in indices if ind not in correct_indices]

        correct_vals = result[field][i][_correct]
        incorrect_vals = result[field][i][_incorrect]

        plot_hist(ax, correct_vals, percentage, density=normalized, alpha=alpha, color="blue", bins=bins, **kwargs)
        plot_hist(ax, incorrect_vals, percentage, density=normalized, alpha=alpha, color="red", bins=bins, **kwargs)

        # ax.set_title(f"{model}, {len(correct_vals)}, {(len(incorrect_vals))}",
        #              fontsize=fontsize)
        ax.set_title(f"{DISPLAY_MODEL_NAME[model]}",
                     fontsize=fontsize)
        if percentage:
            ax.yaxis.set_major_formatter(PercentFormatter(1))

    get_kwargs = lambda j: {"j": j}
    axes = grid_plot_wrapper(_plot, n_plots=len(partitions), ncols=ncols,
                             get_kwargs=get_kwargs, axes=axes)
    return axes


METRIC_TO_YLIM_UP = {
    "y_pred_entropy": 0.726,
    "y_epistemic_uncert": 0.2,
}


def show_several_models(dataset_to_model_results, models, name_to_severity,
                        datasets=["in_domain_test", "ood_test"], figsize=(9, 6), normalized=False,
                        height_clip=0.05, bins=40, percentage=True,
                        sep_accuracy=False, field="y_pred_entropy",
                        **kwargs):
    """
    Before running this function, make sure to run the following at command line:
        # sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng
        # sudo apt-get install cm-super
        # pip install latex
    """
    plt.rc('font', family='serif')

    plt.rcParams['text.usetex'] = True
    if percentage and normalized:
        raise ValueError("At most one should be set to True for percentage and normalized")
    results = {m: get_model_joint_results(dataset_to_model_results, m, datasets) for m in models}

    fig, axes = plt.subplots(nrows=5, ncols=len(models), sharex=True, sharey=True,
                             figsize=figsize)
    for i, m in enumerate(models):
        model_axes = axes[:, i:i + 1] if axes.ndim == 2 else axes[:, None]
        result = results[m]
        if sep_accuracy:
            plot_result_grid_sep_accuracy(result, name_to_severity,
                                          i=i, normalized=normalized, axes=model_axes,
                                          field=field,
                                          ncols=1,
                                          model=m[0], bins=bins, percentage=percentage, **kwargs)
        else:
            plot_result_grid(result, name_to_severity,
                             i=i, normalized=normalized, axes=model_axes, ncols=1,
                             model=m[0], bins=bins, percentage=percentage, **kwargs)

    ax = axes[0][-1]
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y0 + height_clip * (y1 - y0))

    x0, x1 = ax.get_xlim()
    ax.set_xlim(-0.05, METRIC_TO_YLIM_UP.get(field, x1))

    for ax in axes[:, 0]:
        if normalized:
            ax.set_ylabel("Density")
            ax.set_yticks([])
            ax.set_yticks([], minor=True)
        else:
            ax.set_ylabel("Percentage")

    for i, ax in enumerate(axes[:, -1]):
        ax2 = ax.twinx()
        ax2.set_ylabel(i, rotation=0, labelpad=10)
        ax2.set_yticks([])
        ax2.set_yticks([], minor=True)

    for axs in axes[1:]:
        for ax in axs:
            ax.set_title("")
    plt.tight_layout()
    return fig


def save_figure(fig, path: Path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    default = {"bbox_inches": "tight"}
    fig.savefig(path, dpi=fig.dpi, **default)


def plot_histograms_rebuttal(plot_dir):
    plot_dir = Path(plot_dir)
    # load results
    results_dir = "gs://drd-final-results-multi-seeds/severity"
    severity_dataset_to_model_results = fast_load_dataset_to_model_results(results_dir)
    results_dir = "gs://drd-final-results-multi-seeds/aptos"
    aptos_dataset_to_model_results = fast_load_dataset_to_model_results(results_dir)

    # load mapping from name to label
    name_to_severity = get_name_to_label_mapping()

    fig = show_several_models(
        severity_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_SINGLE_MODELS,
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        height_clip=0.05,
        sep_accuracy=True,
        alpha=0.5,
        kde=True,
    )
    save_figure(fig, plot_dir / "severity.pdf")

    fig = show_several_models(
        severity_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_ENSEMBLE_MODELS,
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        height_clip=0.05,
        sep_accuracy=True,
        alpha=0.5,
        kde=True,
    )
    save_figure(fig, plot_dir / "severity_ensemble.pdf")

    fig = show_several_models(
        aptos_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_SINGLE_MODELS,
        datasets=["in_domain_test"],
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        kde=True,
        height_clip=0.05,
        sep_accuracy=True,
        alpha=0.5,
    )
    save_figure(fig, plot_dir / "aptos_indomain.pdf")

    fig = show_several_models(
        aptos_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_SINGLE_MODELS,
        datasets=["ood_test"],
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        kde=True,
        height_clip=0.02,
        sep_accuracy=True,
        alpha=0.5,
    )
    save_figure(fig, plot_dir / "aptos_ood.pdf")

    fig = show_several_models(
        aptos_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_ENSEMBLE_MODELS,
        datasets=["in_domain_test"],
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        kde=True,
        height_clip=0.05,
        sep_accuracy=True,
        alpha=0.5,
    )
    save_figure(fig, plot_dir / "aptos_indomain_ensemble.pdf")

    fig = show_several_models(
        aptos_dataset_to_model_results,
        name_to_severity=name_to_severity,
        models=ID_ENSEMBLE_MODELS,
        datasets=["ood_test"],
        figsize=(9, 6),
        bins="custom",
        normalized=True,
        percentage=False,
        kde=True,
        height_clip=0.01,
        sep_accuracy=True,
        alpha=0.5,
    )
    save_figure(fig, plot_dir / "aptos_ood_ensemble.pdf")
