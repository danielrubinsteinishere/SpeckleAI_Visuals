from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve)
import matplotlib as plt
import numpy as np


def display_multiclass_cm_with_percents(cm, class_names, cmap='viridis'):
  '''
  cmap="Blues" is a good alternative
  '''
  n_classes = cm.shape[0]
  if class_names is None:
      class_names = [f"class_{i}" for i in range(n_classes)]

  cm_pct = cm / cm.sum(axis=1, keepdims=True)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=class_names)
  disp.plot(xticks_rotation=45, include_values=False, cmap=cmap)

  high_color = "black"
  low_color = "white"
  if "Blues" in cmap:
      high_color = "white"
      low_color = "black"
  elif "viridis_r" in cmap:
      high_color = "purple"
      low_color = "yellow"

  for i in range(n_classes):
      for j in range(n_classes):
          pct = cm_pct[i, j] * 100
          color = low_color if cm_pct[i, j] < 0.5 else high_color
          disp.ax_.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                        color=color, fontsize=11)

def get_multiclass_cm_with_percents(proba_test, y_test_c, ypt):
  n_classes = proba_test.shape[1]
  cm = confusion_matrix(y_test_c, ypt, labels=list(range(n_classes)))
  return cm
    
def multiclass_cm_with_percents(proba_test, y_test_c, ypt, class_names, cmap='virids'):
  ''' 
  cmap="Blues" is a good alternative
  '''
  n_classes = proba_test.shape[1]
  if class_names is None:
      class_names = [f"class_{i}" for i in range(n_classes)]
  #print(classification_report(y_test_c, ypt, target_names=class_names))
  cm = get_multiclass_cm_with_percents(proba_test, y_test_c, ypt)
  display_multiclass_cm_with_percents(cm, class_names, cmap=cmap)
  return cm



def plot_confusion_matrix_percent(
    cm,
    class_names,
    figsize=None,
    cmap="Blues",
    font_size=12,
    rotation=45,
    show_counts=False
):
    """
    Plot confusion matrix with row-wise percentages.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix.
    class_names : list[str]
        Labels in the same order as cm.
    figsize : tuple or None
        Figure size. If None, chosen automatically.
    cmap : str
        Matplotlib colormap.
    font_size : int
        Font size for labels and annotations.
    rotation : int
        Rotation for x tick labels.
    show_counts : bool
        If True, show 'xx.x%\\n(count)' in each cell.
        Otherwise show only percentages.
    """
    cm = np.asarray(cm)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square")
    if len(class_names) != cm.shape[0]:
        raise ValueError("len(class_names) must match cm dimensions")

    n_classes = cm.shape[0]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    ) * 100.0

    if figsize is None:
        side = max(10, min(18, 0.75 * n_classes + 4))
        figsize = (side, side)

    fig, ax = plt.pyplot.subplots(figsize=figsize, dpi=200)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Percent", rotation=270, labelpad=15, fontsize=font_size)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, fontsize=font_size)
    ax.set_yticklabels(class_names, fontsize=font_size)
    ax.set_xlabel("Predicted label", fontsize=font_size + 1)
    ax.set_ylabel("True label", fontsize=font_size + 1)
    ax.set_title("Confusion Matrix (%)", fontsize=font_size + 2)

    plt.pyplot.setp(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")

    threshold = 50.0
    for i in range(n_classes):
        for j in range(n_classes):
            if show_counts:
                text = f"{cm_pct[i, j]:.1f}%\n({int(cm[i, j])})"
            else:
                text = f"{cm_pct[i, j]:.1f}%"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_pct[i, j] > threshold else "black",
                fontsize=max(6, font_size - max(0, n_classes - 10) // 2)
            )

    fig.tight_layout()
    plt.pyplot.show()
