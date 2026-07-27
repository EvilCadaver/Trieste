import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

dataset_path = "/CCD/Image"

scanname = ".\Data\FeRh_A04\Scan_061\\rawdata\Scan_061_326273873.h5"
BGname = ".\Data\FeRh_A04\Scan_061_NoProbe\\rawdata\Scan_061_NoProbe_326262948.h5"

with h5py.File(scanname, "r") as h5:
    image_scan = h5[dataset_path][...]

with h5py.File(BGname, "r") as h5:
    image_BG = h5[dataset_path][...]

image = image_scan - image_BG

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