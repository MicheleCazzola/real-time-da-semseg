import os
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(*losses, labels, title, y_label, best_epoch=None, num_ticks=5, save_to=None, show=False):
    
    plt.figure(figsize=(10, 6))
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    
    if best_epoch is not None:
        plt.axvline(x=best_epoch-1, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")
    
    plt.xlabel("Epoch")
    
    period = max(1, len(losses[0]) // num_ticks)
    plt.xticks(
        ticks=[0] + list(np.arange(period-1, len(losses[0]), period)),
        labels=[1] + list(np.arange(period, len(losses[0]) + 1, period))
    )
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()
    else:
        plt.close()
    
def plot_results(dir_path, labels, main_losses=None, mean_ious=None, ious_per_class=None, train_losses=None, show=False):

    train_ious = list(np.array(ious_per_class["train_ious"]).T) if "train_ious" in ious_per_class else None
    val_ious = list(np.array(ious_per_class["val_ious"]).T) if "val_ious" in ious_per_class else None
    
    best_epoch = np.argmax(mean_ious["val_mious"]) + 1 if mean_ious is not None else None
    
    # Training/Validation Losses
    if main_losses is not None:
        plot_losses(
            main_losses["train_losses"],
            main_losses["val_losses"],
            labels=["Training", "Validation"],
            title="Loss",
            y_label="Loss",
            best_epoch=best_epoch,
            num_ticks=5,
            save_to=os.path.join(dir_path, "losses.png"),
            show=show
        )
    
    # Training specific losses
    if train_losses is not None:
        for name, group in train_losses.items():
            plot_losses(
                *group.values(),
                labels=[label.replace("train_losses_", "").replace("_", " ") for label in group.keys()],
                title=f"Training Losses: {name.title()}",
                y_label="Loss",
                best_epoch=best_epoch,
                num_ticks=5,
                save_to=os.path.join(dir_path, f"train_losses_{name}.png"),
                show=show
            )
    
    # mIoU
    if mean_ious is not None:
        plot_losses(
            mean_ious["train_mious"],
            mean_ious["val_mious"],
            labels=["Training", "Validation"],
            title="Mean Intersection over Union (mIoU)",
            y_label="mIoU (%)",
            best_epoch=best_epoch,
            num_ticks=5,
            save_to=os.path.join(dir_path, "mious.png"),
            show=show
        )
    
    # IoU per class (train)
    if train_ious is not None:
        plot_losses(
            *train_ious,
            labels=[label for label in labels] + [f"{label} (Val)" for label in labels],
            title="Training Intersection over Union (IoU) per Class",
            y_label="IoU (%)",
            best_epoch=best_epoch,
            num_ticks=5,
            save_to=os.path.join(dir_path, "train_ious_per_class.png"),
            show=show
        )
    
    # IoU per class (val)
    if val_ious is not None:
        plot_losses(
            *val_ious,
            labels=[label for label in labels],
            title="Validation Intersection over Union (IoU) per Class",
            y_label="IoU (%)",
            best_epoch=best_epoch,
            num_ticks=5,
            save_to=os.path.join(dir_path, "val_ious_per_class.png"),
            show=show
        )