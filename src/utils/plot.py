import logging
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.variables import categories

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
    
def plot_results(dir_path, id2label, main_losses, mean_ious, ious_per_class, train_losses=None, show=False):

    train_ious = list(np.array(ious_per_class["train_ious"]).T)
    val_ious = list(np.array(ious_per_class["val_ious"]).T)
    
    best_epoch = np.argmax(mean_ious["val_mious"]) + 1
    
    # Training/Validation Losses
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
    plot_losses(
        *train_ious,
        labels=[label for label in id2label] + [f"{label} (Val)" for label in id2label],
        title="Training Intersection over Union (IoU) per Class",
        y_label="IoU (%)",
        best_epoch=best_epoch,
        num_ticks=5,
        save_to=os.path.join(dir_path, "train_ious_per_class.png"),
        show=show
    )
    
    # IoU per class (val)
    plot_losses(
        *val_ious,
        labels=[label for label in id2label],
        title="Validation Intersection over Union (IoU) per Class",
        y_label="IoU (%)",
        best_epoch=best_epoch,
        num_ticks=5,
        save_to=os.path.join(dir_path, "val_ious_per_class.png"),
        show=show
    )

### Image visualization
def plot_tensor_mask(mask_tensor, categories):

    categories = dict(sorted(categories.items(), key=lambda item: item[1][0]))

    # Convert mask tensor to numpy array
    mask_array = mask_tensor.squeeze().numpy()

    # Create a colored mask image
    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for i, (label, (value, color)) in enumerate(categories.items()):
        mask = mask_array == i
        colored_mask[mask] = color

    # Display the colored mask
    plt.figure(figsize=(8, 5))
    plt.imshow(colored_mask)
    plt.axis("off")

    # Create a legend
    legend_patches = [mpatches.Patch(color=np.array(color)/255, label=label) for label, (_, color) in categories.items()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plot_class_distribution(urban_classes, rural_classes):
    colors = [np.array(color)/255 for _, color in sorted(categories.values())]

    wedges, texts, autotexts = plt.pie([v.cpu().numpy() for v in urban_classes.values()], labels=urban_classes.keys(), colors=colors, autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.1, startangle=90)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    plt.title('Urban Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

    logging.info("urban_percentage = ", [float(autotext.get_text().strip('%')) for autotext in autotexts])

    wedges, texts, autotexts= plt.pie([v.cpu().numpy() for v in rural_classes.values()], labels=rural_classes.keys(), colors=colors, autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.1, startangle=90)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    plt.title("Rural dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

    logging.info("rural_percentage = ", [float(autotext.get_text().strip('%')) for autotext in autotexts])