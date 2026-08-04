from fileScan import scanFiles
from qSpaceMap import createQSpaceMap
import numpy as np
import matplotlib.pyplot as plt

## Scan specific settings
# Data folder location
folderData = r"C:/git/Trieste/Data"
# Sample name / subfolder in folderData
sampleName = "FeRh_A04"
# Scan specific names: 0) Scan name; 1) Data aqusition name; 2) Background 1 acqusition name; 3) Background 2 acquistion name; 4) Background 3 acqusition name
scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]
# Scan number chosen for analysis
scanNo = 68
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
dataIndex = 43

## Scan measurement parameters
N_BINN = 2                      #Binning of the detector
PIXEL_SIZE = N_BINN * 13.5e-6   #Pixel edge length, m
LAMBDA = 23.5e-9                #Wavelength, m
CY0 = 511                       #Reflection centre Y, pixels
CX0 = 49                        #Reflection centre X, pixels
DCCD = 67e-3                    #Shortest distance from the incident point on the sample to the detector, m 
ALPHA = 17 /180*np.pi           #Incidence beam angle, rad
OMEGA = 16.5 /180*np.pi         #Angle between scattered beam maximum and the detector normal (positive towards the sample surface), rad

## Masks allignemnt
# Make true for masks allignment
alignMasks = False
# Limit the region to show during masks allignment
roiAllignMasks = np.s_[400:600, 0:220]

# ROI for the background zero substraction
roiBG = np.s_[400:450, 160:210] 

## Masking regions, will be added to the empty mask, add rectangles as np.s_[y0:y1,x0:x1] to the list [ ] structure.
maskBS = [np.s_[379:647,0:222], np.s_[511:1024,29:120]]

## Q-space plot settings
# Maximum number of reciprocal-space bins along the longer Qx/Qy dimension.
Q_SPACE_BINS_MAX = 512

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

qxCenters, qyCenters, intensityQxQy, delayScan = createQSpaceMap(
    results=results,
    h5CCDImagePath=h5CCDImagePath,
    h5DelayPath=h5DelayPath,
    delayZero=delayZero,
    dataIndex=dataIndex,
    PIXEL_SIZE=PIXEL_SIZE,
    LAMBDA=LAMBDA,
    CY0=CY0,
    CX0=CX0,
    DCCD=DCCD,
    ALPHA=ALPHA,
    OMEGA=OMEGA,
    alignMasks=alignMasks,
    roiAllignMasks=roiAllignMasks,
    roiBG=roiBG,
    maskBS=maskBS,
    Q_SPACE_BINS_MAX=Q_SPACE_BINS_MAX,
)

if alignMasks:
    fig, ax = plt.subplots()
    intensity = intensityQxQy
    heatmap = ax.pcolormesh(
        qxCenters,
        qyCenters,
        intensity,
        shading="nearest",
        cmap="RdBu_r",
    )
    ax.invert_yaxis()
    ax.set_title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(heatmap, ax=ax, label="Mean intensity per bin")
    plt.show()
    exit()

finiteIntensity = intensityQxQy[np.isfinite(intensityQxQy)]
if finiteIntensity.size == 0:
    raise ValueError("No detector intensities were assigned to the Qx/Qy grid")

# A symmetric scale represents positive and negative differential intensities
# while reducing the influence of isolated extreme pixels.
colourLimit = np.percentile(np.abs(finiteIntensity), 99)
if colourLimit == 0:
    colourLimit = 1.0

fig, ax = plt.subplots()

heatmap = ax.pcolormesh(
    qxCenters,
    qyCenters,
    intensityQxQy,
    shading="nearest",
    cmap="seismic",
    vmin=-colourLimit,
    vmax=colourLimit,
)

ax.set_title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(heatmap, ax=ax, label="Mean intensity per bin")
plt.show()