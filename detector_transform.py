import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

filename = ".\Data\Scan_031_323719586.h5"
dataset_path = "/CCD/Image"

with h5py.File(filename, "r") as h5:
    image = h5[dataset_path][...]

print(type(image))
print(image.shape)
print(image.dtype)
print(np.min(image), np.max(image))

image = np.rot90(image, k=-1)

plt.figure()

plt.imshow(
    image,
    origin="lower",
    cmap="gray",
    norm=LogNorm(
        vmin=max(1, np.nanpercentile(image, 1)),
        vmax=np.nanpercentile(image, 99.9),
    ),
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.colorbar(label="Intensity")
plt.show()