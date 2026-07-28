import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

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

NBinn = 2
SizePixel = NBinn * 13.5e-6     #Pixel edge length, m
Lambda = 23.5e-9                #Wavelength, m
CY1 = 1024/NBinn - 1            #Reflection centre Y, pixels
CX1 = 100/NBinn - 1             #Reflection centre X, pixels
RCCD = 67e-3                    #Distance to detector's CY1, CX1, m
OMEGA = 16.5 /180*np.pi         #Incidence beam angle, rad
CHI = 45 /180*np.pi             #Detector angle, rad

roiBG = np.s_[400:440, 160:210] # ROI of the background assumption

maskBeamStop = np.ones_like(imageScan, dtype=bool)
maskBeamStop[379:647,0:222] = False
maskBeamStop[511:1024,29:120] = False
# maskBeamStop[roiBG] = True

# plt.imshow(imageScan[roiBG], origin="upper", interpolation="none")
# plt.xlabel("x / column")
# plt.ylabel("y / row")
# plt.show()

normNoProbe = np.sum(imageScan[roiBG])/np.sum(imageNoProbe[roiBG])

DESImg = (imageScan - normNoProbe * imageNoProbe)
print("DESImg data type",DESImg.dtype)

normDark = np.sum(imageOnlyProbe[roiBG])/np.sum(imageDark[roiBG])

DESImgUP = (imageOnlyProbe - normDark * imageDark)
print("DESImgUP data type",DESImgUP.dtype)

normScan = np.sum(DESImg * maskBeamStop)
normNoProbe = np.sum(DESImgUP * maskBeamStop)

image = (DESImg/normScan - DESImgUP/normNoProbe) * maskBeamStop

levelBG = np.median(image[roiBG])

image = image.astype(np.float64) - levelBG

valid = image[maskBeamStop & np.isfinite(image)]
# limit = np.percentile(np.abs(valid),90)
vmin = 0.1 * valid.min()
vmax = 0.1 * valid.max()

# print(type(image))
# print(image.shape)
# print(image.dtype)
# print(np.min(image), np.max(image))

plt.figure()

plt.imshow(
    image,
    origin="upper",
    cmap="RdBu",
    # norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    vmin=vmin,
    vmax=vmax,
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.colorbar(label="Intensity")
plt.show()

height, width = image.shape

column_edges = np.arange(width + 1) - 0.5
row_edges = np.arange(height + 1) - 0.5

# Calculate each pixel corner in 3D:
columns, rows = np.meshgrid(column_edges, row_edges)

u = (columns - CX1) * SizePixel
v = (rows - CY1) * SizePixel

# Detector tilted around its column/horizontal axis
x = u
y = v * np.cos(CHI)
z = RCCD - v * np.sin(CHI)

# horizontal and vertical observed angles
angle_x = np.arctan2(x, z)

# Elevation relative to the reflected-beam direction
angle_y = np.arctan2(y, np.sqrt(x**2 + z**2))

angle_x_deg = np.degrees(angle_x)
angle_y_deg = np.degrees(angle_y)

# Plot directly on the nonuniform angular grid:
fig, ax = plt.subplots()

mesh = ax.pcolormesh(
    angle_x_deg,
    angle_y_deg,
    image,
    shading="flat",
    cmap="RdBu",
    vmin=vmin,
    vmax=vmax,
)

ax.invert_yaxis()
ax.set_xlabel(r"$\theta_x$ (degrees)")
ax.set_ylabel(r"$\theta_y$ (degrees)")
fig.colorbar(mesh, ax=ax, label="Intensity")
plt.show()