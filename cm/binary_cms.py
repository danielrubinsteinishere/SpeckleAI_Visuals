import numpy as np
import matplotlib.pyplot as plt

def test_cm_percents_sum_to_200(cm_percents, tol=1e-6):
    """
    Verifies that each confusion matrix in cm_percents sums to 200%.
    Assumes row-normalized percentages (each row sums to 100%).
    """
    for idx, cm_pct in enumerate(cm_percents):
        total = np.sum(cm_pct) * 100.0  # convert from [0,1] to %
        if not np.isclose(total, 200.0, atol=tol):
            raise AssertionError(
                f"cm_percents[{idx}] sums to {total:.2f}%, expected 200%"
            )
    print("✅ All confusion matrices sum to 200% (row-normalized).")



''' multiple binary confusion matruces per plot '''
'''
sample usage:
import numpy as np
cms = [np.array([[500,  0],
              [ 2, 498]])]
# order: 
#[[TN, FP],
# [FN, TP]]

create_image_with_multiple_binary_confusion_matrices(cms=cms, captions=None,n_cols=1,
                    class_names=["English", "Swedish"], color_map='Blues') # A single cm will be created
The number of created cms are determined by cms list.
'''
def create_image_with_multiple_binary_confusion_matrices(cms, captions=None, n_cols=3, class_names=None, color_map=None):
    '''
    color_map parameter:
    Tested color maps:
    'Blues','viridis_r','BuPu','Reds','Greens','Purples','Oranges','plasma_r','cividis_r','magma_r',
    'Greys','bone_r','binary','copper_r', 'gray_r', 'gist_grey_r, 'gist_yarg', 'gist_yerg', 'gist_gray_r',
    'gist_heat_r', 'pink_r', 'PuRd','PuBu','OrRd','CMRmap_r','YlGnBu', 'afmhot_r'
    yellow/orange: 'Wistia', 'autumn_r'
    green/yellow: 'summer_r' 
    blue/green: 'winter_r'  
    '''
    cm_percents = [cm / cm.sum(axis=1, keepdims=True) for cm in cms]
    test_cm_percents_sum_to_200(cm_percents)

    n_matrices = len(cms)
    n_rows = int(np.ceil(n_matrices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Make axes always indexable as 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    if class_names is None:
        class_names = ["0", "1"]

    for idx, (cm, cm_pct) in enumerate(zip(cms, cm_percents)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if color_map is None:
          color_map = 'Blues'

        # Display the confusion matrix (percent values for color scale)
        cax = ax.matshow(cm_pct, cmap=color_map, vmin=0, vmax=1)

        # Title
        if captions is not None:
            ax.set_title(captions[idx], pad=10, fontsize=18)
        else:
            ax.set_title(f'Subject{idx + 1} (test)', pad=10, fontsize=22)

        # Axis labels
        ax.set_xlabel('Predicted', fontsize=18)
        ax.set_ylabel('True', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Annotate each cell with raw value and percent value
        for (i, j), raw_val in np.ndenumerate(cm):
            pct_val = round(cm_pct[i, j] * 100.0, 2)
            annotation = f'{raw_val}\n{pct_val}%'
            color = "white" if pct_val > 50 else "black"
            ax.text(
                j, i, annotation,
                va='center', ha='center',
                color=color, fontsize=20, fontweight='bold'
            )

        # Colorbar per subplot
        #fig.colorbar(cax, ax=ax)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.ax.tick_params(labelsize=12)   # colorbar numbers

    # Hide unused subplots
    for idx in range(n_matrices, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def create_image_with_mean_binary_confusion_matrix(cms, captions=None,
                    class_names=None, color_map='Blues'):
  cms_mean = []
  cms_mean = [np.round(np.mean(np.stack(cms), axis=0)).astype(int)]
  create_image_with_multiple_binary_confusion_matrices(cms=cms_mean, captions=None, n_cols=1,
                      class_names=class_names, color_map=color_map)



def plot_binary_confusion_matrix_from_cm(
    cm,
    class_names=("class_0", "class_1"),
    normalize="true",
    cmap="Blues",
    title="Confusion Matrix (%)",
    show_counts=True,
    font_size=16,
):
    """
    Plot a 2x2 confusion matrix where color intensity is based on percentages.
    """
    cm = np.asarray(cm, dtype=float)

    if cm.shape != (2, 2):
        raise ValueError(f"Expected cm shape (2, 2), got {cm.shape}")

    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
    elif normalize == "all":
        total = cm.sum()
        cm_pct = (cm / total) * 100.0 if total > 0 else np.zeros_like(cm)
    else:
        raise ValueError("normalize must be 'true' or 'all'")

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    ax.set_xlabel("Predicted label", fontsize=font_size)
    ax.set_ylabel("True label", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 50.0
    for i in range(2):
        for j in range(2):
            text = f"{cm_pct[i, j]:.1f}%"
            if show_counts:
                text = f"{int(cm[i, j])}\n({cm_pct[i, j]:.1f}%)"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_pct[i, j] >= thresh else "black",
                fontsize=font_size
            )

    plt.tight_layout()
    plt.show()

    return cm_pct
