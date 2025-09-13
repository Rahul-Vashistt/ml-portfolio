"""
A decision-region plotting utility: robust, pretty, and flexible.

Features:
- Works with numpy arrays or pandas DataFrames
- Supports fitting on: full data, explicit (X_fit,y_fit), or separate X_train/y_train passed as args
- Optional evaluation dataset, or defaults to data used to fit
- Title includes algorithm name and accuracy (or custom metric)
- Uses crisp-filled regions for discrete classes and optional probability heatmap overlay
- Handles sklearn Pipelines gracefully (extracts final estimator name)
- Clean legends, configurable markers/colors, and publication-ready layout

Usage example (at bottom) shows how to run on iris.

"""

from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, Tuple, Union
import warnings


def _get_estimator_name(estimator):
    """Try to extract a human-friendly estimator name (handles Pipelines)."""
    try:
        # sklearn Pipeline has attribute steps = [(name, est), ...]
        if hasattr(estimator, "steps"):
            last = estimator.steps[-1][1]
            return last.__class__.__name__
    except Exception:
        pass
    return estimator.__class__.__name__


def plot_decision_regions(
    estimator,
    X,
    y,
    features: Tuple[int, int] = (0, 1),
    fit_on: Union[str, Tuple] = "full",
    X_train: Optional[object] = None,
    y_train: Optional[object] = None,
    eval_on: Optional[Tuple[object, object]] = None,
    grid_step: float = 0.02,
    pad_ratio: float = 0.10,
    proba_overlay: bool = False,
    cmap=None,
    markers: Optional[list] = None,
    figsize: Tuple[int, int] = (8, 6),
    title_prefix: Optional[str] = None,     # kept but not used in mlxtend-style title
    show_feature_names: bool = True,
    metric_fn = accuracy_score,
    show: bool = True,
    ax = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    Robust 2D decision-region plotter.

    estimator : sklearn-like estimator or Pipeline (unfitted or fitted)
    X, y : full dataset (pandas DataFrame/Series or numpy arrays)
    features : pair of column indices to visualize
    fit_on : 'full', 'train', or (X_fit, y_fit) tuple. If 'train', you must pass X_train/y_train.
    eval_on : optional (X_eval, y_eval) pair to compute score shown in title. If None, uses the data estimator was fit on.
    proba_overlay : If True and estimator has predict_proba, draws a subtle probability heatmap (useful for binary)
    cmap : matplotlib colormap or None (defaults to tab10/Tab20 as needed)
    metric_fn : function(y_true, y_pred) -> float (defaults to accuracy_score)

    Returns matplotlib axis with the plot.
    """

    # --- create/clear axis up front so notebook runs are idempotent ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        ax.cla()

    # --- Normalize inputs and feature names (respect show_feature_names) ---
    if hasattr(X, "iloc"):
        X_arr = X.iloc[:, list(features)].values
        if show_feature_names:
            feat_names = [X.columns[features[0]], X.columns[features[1]]]
        else:
            feat_names = [f"Feature {features[0]}", f"Feature {features[1]}"]
    else:
        X_arr = np.asarray(X)[:, list(features)]
        feat_names = [f"Feature {features[0]}", f"Feature {features[1]}"]

    y_arr = np.asarray(y)

    # Clone estimator so we don't mutate user's object
    est = clone(estimator)

    # Fit est according to fit_on
    if fit_on == "full":
        X_fit, y_fit = X_arr, y_arr
        est.fit(X_fit, y_fit)
    elif fit_on == "train":
        if X_train is None or y_train is None:
            raise ValueError("When fit_on='train' you must provide X_train and y_train arguments")
        if hasattr(X_train, "iloc"):
            X_fit = X_train.iloc[:, list(features)].values
        else:
            X_fit = np.asarray(X_train)[:, list(features)]
        y_fit = np.asarray(y_train)
        est.fit(X_fit, y_fit)
    elif isinstance(fit_on, tuple) and len(fit_on) == 2:
        X_fit_raw, y_fit_raw = fit_on
        if hasattr(X_fit_raw, "iloc"):
            X_fit = X_fit_raw.iloc[:, list(features)].values
        else:
            X_fit = np.asarray(X_fit_raw)[:, list(features)]
        y_fit = np.asarray(y_fit_raw)
        est.fit(X_fit, y_fit)
    else:
        raise ValueError("fit_on must be 'full', 'train', or a (X_fit, y_fit) tuple")

    # Decide evaluation set for scoring
    if eval_on is None:
        X_eval, y_eval = X_fit, y_fit
        eval_name = "fit-data"
    else:
        X_eval_raw, y_eval_raw = eval_on
        if hasattr(X_eval_raw, "iloc"):
            X_eval = X_eval_raw.iloc[:, list(features)].values
        else:
            X_eval = np.asarray(X_eval_raw)[:, list(features)]
        y_eval = np.asarray(y_eval_raw)
        eval_name = "eval-data"

    # Build meshgrid around X_arr (the full data passed to function) for consistent axes
    x_min, x_max = X_arr[:, 0].min(), X_arr[:, 0].max()
    y_min, y_max = X_arr[:, 1].min(), X_arr[:, 1].max()
    x_pad = (x_max - x_min) * pad_ratio if (x_max - x_min) != 0 else 1.0
    y_pad = (y_max - y_min) * pad_ratio if (y_max - y_min) != 0 else 1.0

    xx_range = np.arange(x_min - x_pad, x_max + x_pad + grid_step, grid_step)
    yy_range = np.arange(y_min - y_pad, y_max + y_pad + grid_step, grid_step)
    xx, yy = np.meshgrid(xx_range, yy_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict (class labels) on grid
    try:
        Z = est.predict(grid_points)
    except Exception as e:
        raise RuntimeError(f"Estimator failed to predict on grid points: {e}")
    Z = np.asarray(Z).reshape(xx.shape)

    # probability overlay
    proba_grid = None
    if proba_overlay:
        if hasattr(est, "predict_proba"):
            try:
                P = est.predict_proba(grid_points)
                # For binary/multiclass, we'll visualize max class probability
                proba_grid = np.max(P, axis=1).reshape(xx.shape)
            except Exception:
                warnings.warn("predict_proba failed; continuing without probability overlay")
                proba_grid = None
        else:
            warnings.warn("Estimator has no predict_proba; skipping proba_overlay")
            proba_grid = None

    # Prepare colors and colormap
    classes = np.unique(y_arr)
    n_classes = len(classes)
    if cmap is None:
        # choose tab10 for up to 10 classes, otherwise tab20
        cmap_base = plt.cm.tab10 if n_classes <= 10 else plt.cm.tab20
    else:
        cmap_base = cmap
    colors = [cmap_base(i) for i in range(n_classes)]
    cmap_listed = matplotlib.colors.ListedColormap(colors)

    # --- responsive sizing: compute scale factor from figsize relative to base (8x6) ---
    try:
        base_area = 8 * 6
        area = float(figsize[0]) * float(figsize[1])
        scale = (area / base_area) ** 0.5  # sqrt(area ratio)
    except Exception:
        scale = 1.0

    title_font = max(8, 13 * scale)
    label_font = max(6, 12 * scale)
    legend_font = max(6, 10 * scale)
    marker_size = max(8, int(round(70 * scale)))
    edge_linewidth = max(0.3, 0.6 * scale)
    contour_lw = max(0.3, 0.6 * scale)

    # Crisp discrete regions: use shifted levels (works well for integer labels)
    try:
        # Map Z values into indices 0..n_classes-1 consistently
        class_to_idx = {c: i for i, c in enumerate(classes)}
        Z_idx = np.vectorize(class_to_idx.get)(Z)
        levels = np.arange(-0.5, n_classes + 0.5, 1.0)
        cf = ax.contourf(xx, yy, Z_idx, levels=levels, cmap=cmap_listed, alpha=0.25)
        # draw black boundaries between classes (mlxtend-style)
        ax.contour(xx, yy, Z_idx, levels=levels, colors='k', linewidths=contour_lw, alpha=0.6)
    except Exception:
        # fallback: direct contourf (and attempt to draw boundaries)
        cf = ax.contourf(xx, yy, Z, cmap=cmap_listed, alpha=0.25)
        try:
            ax.contour(xx, yy, Z, colors='k', linewidths=contour_lw, alpha=0.6)
        except Exception:
            pass

    # If probability overlay provided, show as subtle imshow overlay
    if proba_grid is not None:
        ax.imshow(
            proba_grid,
            extent=(xx_range.min(), xx_range.max(), yy_range.min(), yy_range.max()),
            origin='lower',
            cmap='Greys',
            alpha=0.12,
            aspect='auto'
        )

    # Scatter points with clear edge and nice marker cycle
    if markers is None:
        markers = ["o", "s", "^", "P", "*", "D", "v", "X", "h", "+"]

    handles = []
    for i, cls in enumerate(classes):
        mask = (y_arr == cls)
        sc = ax.scatter(
            X_arr[mask, 0], X_arr[mask, 1],
            marker=markers[i % len(markers)],
            s=marker_size,
            facecolor=[colors[i]],
            edgecolor='k',
            linewidth=edge_linewidth,
            label=str(cls)
        )
        handles.append(sc)

    # axes labels (respect show_feature_names)
    if show_feature_names:
        ax.set_xlabel(feat_names[0], fontsize=label_font)
        ax.set_ylabel(feat_names[1], fontsize=label_font)
    else:
        ax.set_xlabel(f"Feature {features[0]}", fontsize=label_font)
        ax.set_ylabel(f"Feature {features[1]}", fontsize=label_font)

    # -----------------------
    # Title: estimator name + score in mlxtend-like style
    # -----------------------
    estimator_name = _get_estimator_name(estimator)
    try:
        y_pred = est.predict(X_eval)
        score_val = metric_fn(y_eval, y_pred)
    except Exception:
        score_val = None

    # format metric label
    metric_label = metric_fn.__name__.replace('_', ' ').title()
    title_score = ""
    if score_val is not None:
        # if metric returns between 0 and 1, display as percentage like mlxtend
        try:
            if 0.0 <= float(score_val) <= 1.0:
                title_score = f"{metric_label}: {float(score_val) * 100.0:.2f}%"
            else:
                title_score = f"{metric_label}: {float(score_val):.4f}"
        except Exception:
            title_score = f"{metric_label}: {score_val}"

    # final title: "EstimatorName | Accuracy: 82.00%"
    if title_score:
        ax.set_title(f"{estimator_name} | {title_score}", fontsize=title_font, weight='semibold')
    else:
        ax.set_title(f"{estimator_name}", fontsize=title_font, weight='semibold')

    # Legend (scaled) — use markerscale instead of touching internals
    markerscale = max(0.6, scale * 0.9)  # adjust how big legend markers appear relative to plot markers
    ax.legend(
        handles=handles,
        title='class',
        loc='best',
        fontsize=legend_font,
        title_fontsize=legend_font,
        markerscale=markerscale,
        frameon=True
    )

    # Clean layout
    ax.set_xlim(xx_range.min(), xx_range.max())
    ax.set_ylim(yy_range.min(), yy_range.max())
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(max(0.6, 0.8 * scale))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    return ax


# ---------------------------
# Example quick demo (Iris) -- runs if user executes this script directly
# ---------------------------
if __name__ == "__main__":
    # lightweight demo using sklearn's iris dataset
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        data = load_iris()
        X = data['data']
        y = data['target']
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        ax = plot_decision_regions(
            clf, X, y, features=(2,3),
            grid_step=0.02,
            proba_overlay=False,
            figsize=(9,7)  # <--- example custom size
        )
    except Exception as e:
        print('Demo failed:', e)
