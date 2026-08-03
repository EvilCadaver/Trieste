import h5py
from fileScan import scanFiles
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

## Scan specific settings
# Data folder location
folderData = r"C:/git/Trieste/Data"
# Sample name / subfolder in folderData
sampleName = "FeRh_A04"
# Scan specific names: 0) Scan name; 1) Data aqusition name; 2) Background 1 acqusition name; 3) Background 2 acquistion name; 4) Background 3 acqusition name
scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]
# Scan number chosen for analysis
scanNo = 31
# File size ratio for scan batches rejection
minimumFileSizeRatio = 0.95
# Verbose reporting on file scan results
verbose = False

## Single file analysis for now
## HDF5 package paths
h5CCDImagePath = "/CCD/Image"
h5DelayPath = "/photon_source/SeedLaser/Delay_line_2"
# Delay zero setting
delayZero = -3096.49
# Single scan index to analyse
dataIndex = 40

## Scan measurement parameters
N_BINN = 2                      #Binning of the detector
PIXEL_SIZE = N_BINN * 13.5e-6   #Pixel edge length, m
LAMBDA = 23.5e-9                #Wavelength, m
CY0 = 511                       #Reflection centre Y, pixels
CX0 = 49                        #Reflection centre X, pixels
DCCD = 67e-3                    #Shortest distance from the incident point on the sample to the detector, m 
ALPHA = 17 /180*np.pi           #Incidence beam angle, rad
OMEGA = 16.5 /180*np.pi         #Angle between scattered beam maximum and the detector normal (positive towards the sample surface), rad
BETA = ALPHA + OMEGA

## Make true for masks allignment
alignMasks = False
roiAllignMasks = np.s_[400:600, 0:220]

# ROI for the background zero substraction
roiBG = np.s_[400:450, 160:210] 

## Masking regions, will be added to the empty mask, add rectangles as np.s_[y0:y1,x0:x1] to the list [ ] structure.
maskBS = [np.s_[379:647,0:222], np.s_[511:1024,29:120]]

## Plot Q space in 3D
Q3D_PLOT = False
# Plot every fourth pixel to keep the interactive 3D plot responsive.
Q3D_STEP = 4

## Q-space plot settings
# Maximum number of reciprocal-space bins along the longer Qx/Qy dimension.
Q_SPACE_BINS_MAX = 512
# Gaussian smoothing, used only for second plot display, set to 0 to disable
Q_SPACE_SMOOTHING_SIGMA = 0  # In Q-space bins

## Calling the folder scanning function
results = scanFiles(
    folderData=folderData,
    sampleName=sampleName,
    scanNames=scanNames,
    scanNo=scanNo,
    verbose=verbose,
    minimumFileSizeRatio=minimumFileSizeRatio,
)

## Unpacking results dictionary
referenceFileSize = results["referenceFileSize"]
minimumFileSize = results["minimumFileSize"]
brokenFiles = results["brokenFiles"]
fileSizes = results["fileSizes"]
allDataFiles = results["allDataFiles"]
dataIndexByFile = results["dataIndexByFile"]
backgroundsForFile = results["backgroundsForFile"]
backgroundNames = results["backgroundNames"]
backgroundGroups = results["backgroundGroups"]
invalidGroups = results["invalidBackgroundGroups"]

## Files scanning short report
if referenceFileSize is None:
    print("No matching HDF5 files found.")
    exit()

print(f"Reference file size: {referenceFileSize:.0f} bytes")
print(f"Minimum accepted size: {minimumFileSize:.0f} bytes")
print(f"Borked files: {len(brokenFiles)}")

for brokenFile in sorted(brokenFiles):
    print(f"  {brokenFile}: {fileSizes[brokenFile]} bytes")

print(
    f"\nBackground timing windows: "
    f"{len(backgroundGroups) + len(invalidGroups)}"
)
print(f"Valid background groups: {len(backgroundGroups)}")
print(f"Invalid background groups: {len(invalidGroups)}")
print(
    "Repaired background groups: "
    f"{sum(bool(group['rejectedBackgrounds']) for group in backgroundGroups)}"
)

## Check for validity
print(f"Chosen scan index: {dataIndex}")
try:
    dataFilePath = results["allDataFiles"][dataIndex]
    
except IndexError:
    print(f"Scan index {dataIndex} does not exist")
    exit()

else:
    if dataFilePath in results["brokenFiles"]:
        print("Scan is broken")
        exit()

    elif dataFilePath in results["filesWithoutBackground"]:
        print("No suitable background set")
        exit()

    else:
        backgrounds = results["backgroundsForFile"].get(dataFilePath)

        if backgrounds is None:
            print("No background assignment found")
        else:
            print("Scan:", dataFilePath)
            print("NoProbe:", backgrounds["NoProbe"])
            print("OnlyProbe:", backgrounds["OnlyProbe"])
            print("Dark:", backgrounds["Dark"])

## Reading data files for 'dataIndex'

with h5py.File(dataFilePath, "r") as h5:
    imageScan = h5[h5CCDImagePath][...]
    imageScan = np.rot90(imageScan.T, 2)
    delayScan = round(- h5[h5DelayPath][...] + delayZero,1)

with h5py.File(backgrounds["Dark"], "r") as h5:
    imageDark = h5[h5CCDImagePath][...]
    imageDark = np.rot90(imageDark.T, 2)

with h5py.File(backgrounds["NoProbe"], "r") as h5:
    imageNoProbe = h5[h5CCDImagePath][...]
    imageNoProbe = np.rot90(imageNoProbe.T, 2)

with h5py.File(backgrounds["OnlyProbe"], "r") as h5:
    imageOnlyProbe = h5[h5CCDImagePath][...]
    imageOnlyProbe = np.rot90(imageOnlyProbe.T, 2)

## BeamStop masking
maskBeamStop = np.ones_like(imageScan, dtype=bool)
for mask in maskBS:
    maskBeamStop[mask] = False
maskBGroi = ~np.ones_like(imageScan, dtype=bool)
maskBGroi[roiBG] = True

## Differential image formulation
normNoProbe = np.sum(imageScan[roiBG])/np.sum(imageNoProbe[roiBG])
imageDifferential = (imageScan - normNoProbe * imageNoProbe)

normDark = np.sum(imageOnlyProbe[roiBG])/np.sum(imageDark[roiBG])
imageDifferentialNoProbe = (imageOnlyProbe - normDark * imageDark)

normScan = np.sum(imageDifferential * maskBeamStop)
normOnlyProbe = np.sum(imageDifferentialNoProbe * maskBeamStop)

image = (imageDifferential/normScan - imageDifferentialNoProbe/normOnlyProbe)

## Masks allignment or image reconstraction for 'dataIndex'
if alignMasks:
    image = image * (maskBeamStop ^ maskBGroi)
    image = image[roiAllignMasks]
    valid = image[maskBGroi[roiAllignMasks] & np.isfinite(image)]
    vmin = valid.min()
    vmax = valid.max()
    
else:
    ## Background levelling
    levelBG = np.median(image[roiBG])
    image = (image - levelBG) * maskBeamStop
    valid = image[maskBeamStop & np.isfinite(image)]
    vmin = 0.1 * valid.min()
    vmax = 0.1 * valid.max()

## Plotting the 'image'
plt.figure()

plt.imshow(
    image,
    origin="upper",
    cmap="RdBu",
    vmin=vmin,
    vmax=vmax,
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
plt.colorbar(label="Intensity")
plt.show(block=False)

if alignMasks:
    exit()

# Getting pixels coordinates
height, width = image.shape
# Coordinates in the detector plane
v, u = np.indices((height, width))
## Sample's coordinate system
# Maximum of the scattered beam
R0 = np.array([DCCD/np.cos(OMEGA)*np.cos(ALPHA), 0.0, -DCCD/np.cos(OMEGA)*np.sin(ALPHA)])
# Detector's normal
N_CCD = np.array([np.cos(BETA), 0.0, -np.sin(BETA)])
# Detector's lines
e_v = np.array([0.0, 1.0, 0.0])
# Detector's columns
# e_u = np.cross(e_v, N_CCD)
e_u = np.array([-np.sin(BETA), 0.0, -np.cos(BETA)])
# Each pixel's coordinate
du = (u - CX0) * PIXEL_SIZE
dv = (v - CY0) * PIXEL_SIZE

R = (
    R0
    + du[..., None] * e_u
    + dv[..., None] * e_v
)

# Scattering vectors' coordinates
S_f = R / np.linalg.norm(R, axis=-1, keepdims=True)
# Incident beam coordinates
S_i = np.array([np.cos(ALPHA), 0.0, np.sin(ALPHA)])
# Reciprocal coordinates of the detector pixcels
Q = 2*np.pi/LAMBDA*(S_f - S_i)

## Plotting q space in 3D if Q3D_PLOT == True
if Q3D_PLOT:
    from matplotlib.colors import TwoSlopeNorm

    q_plot = Q[::Q3D_STEP, ::Q3D_STEP] * 1e-9  # Convert m^-1 to nm^-1
    i_plot = image[::Q3D_STEP, ::Q3D_STEP]
    mask_plot = maskBeamStop[::Q3D_STEP, ::Q3D_STEP]

    valid = (
        mask_plot
        & np.isfinite(i_plot)
        & np.all(np.isfinite(q_plot), axis=-1)
    )

    q_points = q_plot[valid]
    intensity = i_plot[valid]

    # Symmetric colour scale suitable for differential intensity.
    colourLimit = np.percentile(np.abs(intensity), 99)
    norm = TwoSlopeNorm(
        vmin=-colourLimit,
        vcenter=0.0,
        vmax=colourLimit,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    dots = ax.scatter(
        q_points[:, 0],
        q_points[:, 1],
        q_points[:, 2],
        c=intensity,
        cmap="RdBu_r",
        norm=norm,
        s=2,
        linewidths=0,
    )

    ax.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
    ax.set_ylabel(r"$Q_y$ (nm$^{-1}$)")
    ax.set_zlabel(r"$Q_z$ (nm$^{-1}$)")
    ax.set_title(f"Scan {dataIndex}, delay = {delayScan} ps")

    # Preserve the relative scale of the reciprocal-space axes.
    axisRanges = np.ptp(q_points, axis=0)
    ax.set_box_aspect(np.maximum(axisRanges, 1e-12))

    fig.colorbar(dots, ax=ax, pad=0.12, label="Intensity")
    plt.tight_layout()
    plt.show()

## Rebinning and plotting intensity in the Qx/Qy projection
valid = (
    maskBeamStop
    & np.isfinite(image)
    & np.all(np.isfinite(Q[..., :2]), axis=-1)
)

# Q is calculated in m^-1; use nm^-1 for plotting.
qxValues = Q[..., 0][valid] * 1e-9
qyValues = Q[..., 1][valid] * 1e-9
intensityValues = image[valid]

qxMin, qxMax = qxValues.min(), qxValues.max()
qyMin, qyMax = qyValues.min(), qyValues.max()
qxSpan = qxMax - qxMin
qySpan = qyMax - qyMin

if Q_SPACE_BINS_MAX < 1:
    raise ValueError("Q_SPACE_BINS_MAX must be a positive integer")

# Use square reciprocal-space bins. The longer dimension has at most
# Q_SPACE_BINS_MAX bins; the shorter dimension is scaled proportionally.
dq = max(qxSpan, qySpan) / Q_SPACE_BINS_MAX
if not np.isfinite(dq) or dq <= 0:
    raise ValueError("The Qx/Qy projection has no finite reciprocal-space extent")

qxBinCount = min(Q_SPACE_BINS_MAX, max(1, int(np.ceil(qxSpan / dq))))
qyBinCount = min(Q_SPACE_BINS_MAX, max(1, int(np.ceil(qySpan / dq))))
qxEdges = qxMin + np.arange(qxBinCount + 1) * dq
qyEdges = qyMin + np.arange(qyBinCount + 1) * dq

intensitySum, _, _ = np.histogram2d(
    qxValues,
    qyValues,
    bins=(qxEdges, qyEdges),
    weights=intensityValues,
)
pixelCount, _, _ = np.histogram2d(
    qxValues,
    qyValues,
    bins=(qxEdges, qyEdges),
)

# Average the intensities of all detector pixels assigned to each Q-space bin.
intensityQxQy = np.divide(
    intensitySum,
    pixelCount,
    out=np.full_like(intensitySum, np.nan),
    where=pixelCount > 0,
)

finiteIntensity = intensityQxQy[np.isfinite(intensityQxQy)]
if finiteIntensity.size == 0:
    raise ValueError("No detector intensities were assigned to the Qx/Qy grid")

# A symmetric scale represents positive and negative differential intensities
# while reducing the influence of isolated extreme pixels.
colourLimit = np.percentile(np.abs(finiteIntensity), 99)
if colourLimit == 0:
    colourLimit = 1.0

if Q_SPACE_SMOOTHING_SIGMA > 0:
    smoothedSum = gaussian_filter(
        intensitySum,
        sigma=Q_SPACE_SMOOTHING_SIGMA,
    )
    smoothedCount = gaussian_filter(
        pixelCount,
        sigma=Q_SPACE_SMOOTHING_SIGMA,
    )

    intensityQxQySmoothed = np.divide(
        smoothedSum,
        smoothedCount,
        out=np.full_like(smoothedSum, np.nan),
        where=smoothedCount > 0,
    )

    # Do not invent data outside the measured projection.
    intensityQxQySmoothed[pixelCount == 0] = np.nan
else:
    intensityQxQySmoothed = intensityQxQy
    
fig, ax = plt.subplots(figsize=(9, 7))
q_space_plot = ax.pcolormesh(
    qxEdges,
    qyEdges,
    intensityQxQySmoothed.T,
    shading="flat",
    cmap="seismic",
    vmin=-colourLimit,
    vmax=colourLimit,
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
ax.set_ylabel(r"$Q_y$ (nm$^{-1}$)")
ax.set_title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
fig.colorbar(q_space_plot, ax=ax, label="Mean intensity per bin")
fig.tight_layout()
plt.show()

