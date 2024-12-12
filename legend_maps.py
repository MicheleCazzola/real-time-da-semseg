import matplotlib.patches as mpatches
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Apri l'immagine
with open("/Users/marcodeluca/Downloads/Val/Rural/masks_png/2522.png", "rb") as f:
    img = Image.open(f)
    imgtrans = torchvision.transforms.ToTensor()(img)

# Definisci i colori personalizzati e le categorie
categories = {
    'BARREN': (0.003921568859368563, (159, 129, 183)),       # lilla
    'AGRICULTURE': (0.027450980618596077, (255, 195, 128)),  # arancione
    'BUILDING': (0.007843137718737125, (255, 0, 0)),         # Rosso
    'WATER': (0.01568627543747425, (0, 0, 255)),             # Blu
    'ROAD': (0.0117647061124444, (255, 255, 0)),             # Giallo
    'BG': (0.019607843831181526, (255, 255, 255)),           # bianco
    'FOREST': (0.0235294122248888, (0, 255, 0))              # Verde
}

# Crea una mappa di colori
cmap = mcolors.ListedColormap(
    [color for _, color in sorted(categories.values())])

# Converti il tensore in un array numpy
img_array = imgtrans.numpy().squeeze()

# Crea un'immagine vuota con 3 canali (RGB)
colored_img = np.zeros(
    (img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

# Applica i colori personalizzati
for label, (value, color) in categories.items():
    mask = img_array == value
    colored_img[mask] = color

# Visualizza l'immagine colorata
print(set(imgtrans.flatten().tolist()))

plt.figure(figsize=(8, 5))

plt.imshow(colored_img)
plt.axis("off")

# Crea la legenda
legend_patches = [mpatches.Patch(color=np.array(color)/255, label=label)
                  for label, (_, color) in categories.items()]

plt.legend(handles=legend_patches, bbox_to_anchor=(
    1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
