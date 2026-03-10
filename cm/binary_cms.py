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
