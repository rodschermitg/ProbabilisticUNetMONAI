import os

import matplotlib
import matplotlib.pyplot as plt

try:
    import config
except ModuleNotFoundError:
    from src import config


def concat_logs(dict_):
    """
    Loss/metric values in train_logs are stored in separate lists for each
    fold. Use this function to concat all lists in the train_logs dict
    into a single list.
    """
    return [value for logs in dict_.values() for value in logs]


def create_log_plots(
    y_list,
    fold,
    epoch,
    labels,
    output_path,
    best_fold=None,
    best_epoch=None,
    best_dice=None,
    title=None,
    y_label=None,
    log_y_scale=False
):
    """
    Creates and saves plots to log training progress.

    Args:
        y_list (list of lists): Contains lists of y-axis values for each line
            plot.
        fold (int): Current fold.
        epoch (int): Current epoch.
        labels (list of str): List of labels for each line plot.
        output_path (str): File path to save the generated plot.
        best_fold (int, optional): Best fold corresponding to best_dice value.
        best_epoch (int, optional): Best epoch corresponding to best_dice
            value.
        best_dice (float, optional): Best dice value.
        title (str, optional): Title of the plot.
        y_label (str, optional): Label for the y-axis.
        log_y_scale (bool, optional): Set to 'True' to enable logarithmic scale
            on y-axis.
    """

    matplotlib.use("Agg")
    plt.style.use("ggplot")
    plt.figure()
    for y, label in zip(y_list, labels):
        # length of y varies depending on whether it contains train or val data
        if len(y) == fold*config.EPOCHS+epoch+1:  # train
            x = list(range(0, len(y)))
        else:  # val
            x = list(range(
                config.VAL_INTERVAL-1,
                len(y)*config.VAL_INTERVAL,
                config.VAL_INTERVAL
            ))
        plt.plot(x, y, label=label)

    if all((best_fold, best_epoch, best_dice)):
        plt.plot(
            best_fold*config.EPOCHS+best_epoch,
            best_dice,
            "x",
            markersize=10,
            markeredgewidth=3
        )

    # add vertical lines representing each fold
    for fold_line in range(
        config.EPOCHS-1,
        (fold+1)*config.EPOCHS-1,
        config.EPOCHS
    ):
        plt.axvline(x=fold_line, linestyle="dotted", color="grey")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    if log_y_scale:
        plt.yscale("log")
    plt.legend(loc="upper left")
    plt.savefig(output_path)
    plt.close()


def create_slice_plots(
    images,
    slice_dim=0,
    title=None,
    labels=None,
    num_slices=10
):
    total_slices = images.shape[slice_dim+1]
    slice_stride = total_slices // num_slices
    num_plots = images.shape[0]

    fig, ax = plt.subplots(num_slices, num_plots)
    for row in range(num_slices):
        for col in range(num_plots):
            ax[row, col].axis("off")

            slice_idx = (row + 1) * slice_stride
            slice_idx = min(slice_idx, total_slices-1)  # avoid IndexError

            if slice_dim == 0:
                ax[row, col].imshow(images[col, slice_idx, :, :])
            elif slice_dim == 1:
                ax[row, col].imshow(images[col, :, slice_idx, :])
            elif slice_dim == 2:
                ax[row, col].imshow(images[col, :, :, slice_idx])

            if row == 0 and labels is not None:
                ax[row, col].set_title(labels[col])

    fig.suptitle(title)
    plt.show()


def get_patient_name(file_path):
    start_idx = file_path.find("Patient")
    end_idx = file_path.find(os.sep, start_idx)
    patient_name = file_path[start_idx:end_idx]

    return patient_name
