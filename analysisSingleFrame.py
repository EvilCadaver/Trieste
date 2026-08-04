from fileScan import scanFiles
from qSpaceFunctions import createQSpaceMap, createRadialIntensityProfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
CX0 = 15                        #Reflection centre X, pixels
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

## Q-space analysis
# Show angles chosen for integration
plotProfileAngles = True
# Choose the highlight colour
plotProfileAnglesColour = r"green"
# Angle from Qx in deg
ZETA = 45
# Acceptance angle in deg
D_ZETA = 20
# Symmetry
ZETA_SYMMETRY = 4
# Radial step bin
RADIAL_STEP_BIN = 2

## Calling the folder scanning function
results = scanFiles(
    folderData=folderData,
    sampleName=sampleName,
    scanNames=scanNames,
    scanNo=scanNo,
    verbose=verbose,
    minimumFileSizeRatio=minimumFileSizeRatio,
)

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

qxSpan = np.ptp(qxCenters)
qySpan = np.ptp(qyCenters)

longSide = 8.0  # inches

if qxSpan >= qySpan:
    plotWidth = longSide
    plotHeight = longSide * qySpan / qxSpan
else:
    plotHeight = longSide
    plotWidth = longSide * qxSpan / qySpan

# Add some horizontal space for the colour bar.
figQspace, axQspace = plt.subplots(
    figsize=(plotWidth + 1.2, plotHeight),
    layout="constrained",
)

heatmap = axQspace.pcolormesh(
    qxCenters,
    qyCenters,
    intensityQxQy,
    shading="nearest",
    cmap="seismic",
    vmin=-colourLimit,
    vmax=colourLimit,
)

if plotProfileAngles:
    qxGrid, qyGrid = np.meshgrid(qxCenters, qyCenters)

    angleGrid = np.degrees(
        np.arctan2(qyGrid, qxGrid)
    )

    symmetryPeriod = 360.0 / ZETA_SYMMETRY

    angleDifference = (
        (
            angleGrid
            - ZETA
            + symmetryPeriod / 2
        )
        % symmetryPeriod
        - symmetryPeriod / 2
    )

    profileMask = (
        np.abs(angleDifference) <= D_ZETA / 2
    ) & np.isfinite(intensityQxQy)

    acceptedOverlay = np.ma.masked_where(
        ~profileMask,
        np.ones_like(intensityQxQy),
    )

    axQspace.pcolormesh(
        qxCenters,
        qyCenters,
        acceptedOverlay,
        shading="nearest",
        cmap=ListedColormap([plotProfileAnglesColour]),
        alpha=0.25,
    )

axQspace.set_title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
axQspace.set_aspect("equal", adjustable="box")
figQspace.colorbar(heatmap, ax=axQspace, label="Mean intensity per bin")

# Sum intensities in radial shells, restricted to the reference direction and
# all directions related to it by ZETA_SYMMETRY. D_ZETA is the complete
# angular width, so the accepted half-width is D_ZETA/2 on either side.
distance, sumIntensity = createRadialIntensityProfile(
    qxCenters=qxCenters,
    qyCenters=qyCenters,
    intensity=intensityQxQy,
    ZETA=ZETA,
    D_ZETA=D_ZETA,
    ZETA_SYMMETRY=ZETA_SYMMETRY,
    RADIAL_STEP_BIN=RADIAL_STEP_BIN,
)

figProfile, axProfile = plt.subplots()
axProfile.plot(distance, sumIntensity)
axProfile.set_xlabel(r"$|Q|$ (nm$^{-1}$)")
axProfile.set_ylabel("Summed intensity")
axProfile.set_title(
    f"Zeta={ZETA} deg, symmetry={ZETA_SYMMETRY}, "
    f"acceptance={D_ZETA} deg"
)
axProfile.grid(True, alpha=0.3)

def moveFigure(fig, x, y):
    """Move a Matplotlib figure window to screen position (x, y)."""
    window = fig.canvas.manager.window

    if hasattr(window, "move"):          # Qt backend
        window.move(x, y)
    elif hasattr(window, "wm_geometry"):  # Tk backend
        window.wm_geometry(f"+{x}+{y}")
    else:
        print("Current Matplotlib backend does not support window positioning")

moveFigure(figQspace, 20, 50)
moveFigure(figProfile, 600, 50)

plt.show()