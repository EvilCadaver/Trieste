import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

data_folder = Path(r"C:/git/Trieste/Data/FeRh_A04")

fileScan = (data_folder / "Scan_061" / "rawdata" / "Scan_061_326273873.h5")
fileDark = (data_folder / "Scan_061_Dark" / "rawdata" / "Scan_061_Dark_326266957.h5")
fileNP = (data_folder / "Scan_061_NoProbe" / "rawdata" / "Scan_061_NoProbe_326262948.h5")
fileOP = (data_folder / "Scan_061_OnlyProbe" / "rawdata" / "Scan_061_OnlyProbe_326264983.h5")

dataset_path = "/CCD/Image"

with h5py.File(fileScan, "r") as h5:
    imageScan = h5[dataset_path][...]
    imageScan = np.rot90(imageScan, 2)

with h5py.File(fileDark, "r") as h5:
    imageDark = h5[dataset_path][...]
    imageDark = np.rot90(imageDark, 2)

with h5py.File(fileNP, "r") as h5:
    imageNP = h5[dataset_path][...]
    imageNP = np.rot90(imageNP, 2)

with h5py.File(fileOP, "r") as h5:
    imageOP = h5[dataset_path][...]
    imageOP = np.rot90(imageOP, 2)

roiBG = np.s_[400:440, 100:150] # ROI of the background assumption

## Normalisation factors to imageScan for backgrounds
normDark = np.sum(imageScan[roiBG])/np.sum(imageDark[roiBG])

DESImg = imageScan - normNp * imageNP

# normNP = np.sum(imageScan[roiBG])/np.sum(imageNP[roiBG])
# normOP = np.sum(imageScan[roiBG])/np.sum(imageOP[roiBG])

levelBG = np.median(DESImg[roiBG])

image = DESImg.astype(np.float64) - levelBG

print(type(image))
print(image.shape)
print(image.dtype)
print(np.min(image), np.max(image))

plt.figure()

plt.imshow(
    image.T,
    origin="upper",
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