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
    imageScan = np.rot90(imageScan.T, 2)

with h5py.File(fileDark, "r") as h5:
    imageDark = h5[dataset_path][...]
    imageDark = np.rot90(imageDark.T, 2)

with h5py.File(fileNP, "r") as h5:
    imageNoProbe = h5[dataset_path][...]
    imageNoProbe = np.rot90(imageNoProbe.T, 2)

with h5py.File(fileOP, "r") as h5:
    imageOnlyProbe = h5[dataset_path][...]
    imageOnlyProbe = np.rot90(imageOnlyProbe.T, 2)

roiBG = np.s_[400:440, 160:210] # ROI of the background assumption

maskBeamStop = np.ones_like(imageScan, dtype=bool)
maskBeamStop[379:647,0:222] = False
maskBeamStop[511:1024,29:120] = False
maskBeamStop[roiBG] = True

plt.imshow(imageScan[roiBG], origin="upper", interpolation="none")
plt.xlabel("x / column")
plt.ylabel("y / row")
plt.show()

normNoProbe = np.sum(imageScan[roiBG])/np.sum(imageNoProbe[roiBG])

DESImg = (imageScan - normNoProbe * imageNoProbe)
print("DESImg data type",DESImg.dtype)

normDark = np.sum(imageOnlyProbe[roiBG])/np.sum(imageDark[roiBG])

DESImgUP = (imageOnlyProbe - normDark * imageDark)
print("DESImgUP data type",DESImgUP.dtype)

normScan = np.sum(imageScan * maskBeamStop)
normNoProbe = np.sum(imageOnlyProbe * maskBeamStop)

image = (DESImg/normScan - DESImgUP/normNoProbe) * maskBeamStop

levelBG = np.median(image[roiBG])

image = image.astype(np.float64) - levelBG

print(type(image))
print(image.shape)
print(image.dtype)
print(np.min(image), np.max(image))

plt.figure()

plt.imshow(
    image,
    origin="upper",
    cmap="RdBu",
    # norm=LogNorm(
    #     vmin=max(1, np.nanpercentile(image, 1)),
    #     vmax=np.nanpercentile(image, 99.9),
    # ),
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.colorbar(label="Intensity")
plt.show()