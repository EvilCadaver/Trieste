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
CHI = -45 /180*np.pi             #Detector angle, rad

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

columns, rows = np.meshgrid(
    np.arange(width), 
    np.arange(height),
)

# Coordinates in the untilted detector plane
x0 = (columns - CX1) * SizePixel
y0 = (rows - CY1) * SizePixel

# Detector rotation around its column direction
x = x0 * np.cos(CHI)
y = y0
z = RCCD - x0 * np.sin(CHI)

# Unit vector toward every detector pixel
ray_length = np.sqrt(x**2 + y**2 + z**2)

sf_x = x / ray_length
sf_y = y / ray_length
sf_z = z / ray_length

# Here x* is along the surface and y* is the outward sample normal:

sin_omega = np.sin(OMEGA)
cos_omega = np.cos(OMEGA)

# Sample basis expressed in detector/laboratory coordinates
e_xstar = np.array([sin_omega, 0.0, -cos_omega])
e_ystar = np.array([ 0.0, -1.0, 0.0])
e_zstar = np.array([cos_omega, 0.0, sin_omega])

# Incident-beam unit vector in laboratory coordinates
si = (-cos_omega * e_xstar - sin_omega * e_zstar)

# Scattering vector

k = 2 * np.pi / Lambda

qx_lab = k * (sf_x - si[0])
qy_lab = k * (sf_y - si[1])
qz_lab = k * (sf_z - si[2])

# Project into the sample reciprocal basis
qx_star = (
    qx_lab * e_xstar[0]
    + qy_lab * e_xstar[1]
    + qz_lab * e_xstar[2]
)

qy_star = (
    qx_lab * e_ystar[0]
    + qy_lab * e_ystar[1]
    + qz_lab * e_ystar[2]
)

# Convert m^-1 to nm^-1
qx_star *= 1e-9
qy_star *= 1e-9

# The first value should be approximately zero and the second approximately expected_qy.
# print(qx_star[int(CY1), int(CX1)])
# print(qy_star[int(CY1), int(CX1)])

# expected_qy = 2 * k * np.sin(OMEGA) * 1e-9
# print("Expected specular Qy*:", expected_qy)

# Rebinning
valid = maskBeamStop & np.isfinite(image)

qx_values = qx_star[valid]
qy_values = qy_star[valid]
intensity_values = image[valid]

qx_min, qx_max = qx_values.min(), qx_values.max()
qy_min, qy_max = qy_values.min(), qy_values.max()

qx_span = qx_max - qx_min
qy_span = qy_max - qy_min

# At most approximately 600 bins along the larger dimension
dq = max(qx_span, qy_span) / 600

nx = int(np.ceil(qx_span / dq))
ny = int(np.ceil(qy_span / dq))

qx_edges = qx_min + np.arange(nx + 1) * dq
qy_edges = qy_min + np.arange(ny + 1) * dq

intensity_sum, _, _ = np.histogram2d(
    qx_values,
    qy_values,
    bins=(qx_edges, qy_edges),
    weights=intensity_values,
)

sample_count, _, _ = np.histogram2d(
    qx_values,
    qy_values,
    bins=(qx_edges, qy_edges),
)

rebinned = np.divide(
    intensity_sum,
    sample_count,
    out=np.full_like(intensity_sum, np.nan),
    where=sample_count > 0,
)

# Plotting

fig, ax = plt.subplots()

mesh = ax.pcolormesh(
    qx_edges,
    qy_edges,
    rebinned.T,
    shading="flat",
    cmap="RdBu",
    vmin=vmin,
    vmax=vmax,
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$Q_x^*$ (nm$^{-1}$)")
ax.set_ylabel(r"$Q_y^*$ (nm$^{-1}$)")
fig.colorbar(mesh, ax=ax, label="Mean intensity")

plt.show()