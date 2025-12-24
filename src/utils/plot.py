import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.variables import categories

### Plot losses and mious
def plot_losses_mious(train_losses, eval_losses, miou_scores, num_epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(eval_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(range(0, num_epochs), range(1, num_epochs + 1))
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid()

    ax2.plot(miou_scores, label='mIoU')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('mIoU')
    ax2.set_xticks(range(0, num_epochs), range(1, num_epochs + 1))
    ax2.set_title('mIoU')
    ax2.legend()
    ax2.grid()

    plt.show()
    
    
def plot_mious_per_category(miou_scores, num_epochs):
    plt.figure(figsize=(10, 6))
    for class_name, miou_values in miou_scores.items():
        plt.plot(range(num_epochs), miou_values, label=class_name)

    plt.xlabel('Epoch')
    plt.ylabel('mIoU (%)')
    plt.xticks(range(0, num_epochs), range(1, num_epochs + 1))
    plt.title('mIoU per Class over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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

    print("urban_percentage = ", [float(autotext.get_text().strip('%')) for autotext in autotexts])

    wedges, texts, autotexts= plt.pie([v.cpu().numpy() for v in rural_classes.values()], labels=rural_classes.keys(), colors=colors, autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.1, startangle=90)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    plt.title("Rural dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("rural_percentage = ", [float(autotext.get_text().strip('%')) for autotext in autotexts])