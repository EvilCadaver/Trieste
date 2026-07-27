import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

dataset_path = "/CCD/Image"

fileScan = ".\Data\FeRh_A04\Scan_061\\rawdata\Scan_061_326273873.h5"
fileDark = ".\Data\FeRh_A04\Scan_061_NoProbe\\rawdata\Scan_061_NoProbe_326262948.h5"
fileNP = ".\Data\FeRh_A04\Scan_061_NoProbe\\rawdata\Scan_061_NoProbe_326262948.h5"
fileOP = ".\Data\FeRh_A04\Scan_061_OnlyProbe\\rawdata\Scan_061_OnlyProbe_326264983.h5"

with h5py.File(fileScan, "r") as h5:
    imageScan = h5[dataset_path][...]

with h5py.File(fileDark, "r") as h5:
    imageDark = h5[dataset_path][...]

with h5py.File(fileNP, "r") as h5:
    imageNP = h5[dataset_path][...]

with h5py.File(fileOP, "r") as h5:
    imageOP = h5[dataset_path][...]

image = imageScan - imageDark - imageNP - imageOP

areaBG = image[400:440,100:150]

levelBG = np.median(image)

image = image.astype(np.float64) - levelBG

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
    # norm=LogNorm(
    #     vmin=max(1, np.nanpercentile(image, 1)),
    #     vmax=np.nanpercentile(image, 99.9),
    # ),
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.colorbar(label="Intensity")
plt.show()