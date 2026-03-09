import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_subject_metrics(metrics_dict, subject_prefix="", colormap=None, color=["#1f77b4", "#ff7f0e", "#2ca02c"]):
    """
    Plot per-subject metrics from a dict of lists.

    Parameters
    ----------
    metrics_dict : dict
        Keys are metric names (e.g. "AUC", "Acc", "F1"),
        values are lists/arrays of metric values for subjects 1..N.
        All lists must have the same length.
    subject_prefix : str
        Optional prefix for subject labels.
    colormaps examples:
      viridis — modern, clear, professional
      cividis — similar, but more colorblind-friendly
      Set2 — softer presentation colors
      tab10 — strong distinct category colors
    custom caolors examples:
      color=["navy", "orange", "green"]
    """
    if not metrics_dict:
        raise ValueError("metrics_dict is empty.")

    # Convert to arrays and validate lengths
    metric_names = list(metrics_dict.keys())
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metrics_dict.items()}

    lengths = [len(v) for v in metric_arrays.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"All metric lists must have the same length, got lengths={lengths}")

    n_subjects = lengths[0]

    # Build subject labels
    subject_labels = [f"{subject_prefix}{i}" for i in range(1, n_subjects + 1)]

    # Build dataframe
    df = pd.DataFrame({"Subjects": subject_labels})
    for metric_name, values in metric_arrays.items():
        df[metric_name] = values

    # Add mean row
    mean_row = {"Subjects": "mean"}
    for metric_name, values in metric_arrays.items():
        mean_row[metric_name] = values.mean()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    if not colormap:
      df.plot(
          x="Subjects",
          y=metric_names,
          kind="bar",
          ax=ax,
          color=color
      )
    else:
      df.plot(
          x="Subjects",
          y=metric_names,
          kind="bar",
          ax=ax,
          colormap=colormap
      )


    ax.set_ylabel("Metric", fontsize=18)
    ax.set_xlabel("Subjects", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=18, rotation=0)
    ax.tick_params(axis='y', labelsize=18)

    ax2 = ax.twinx()
    ax2.set_ylabel("Accuracy (%)", fontsize=18)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelsize=16)
    
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.5, 0.4))

    plt.tight_layout()
    plt.show()

    return df
