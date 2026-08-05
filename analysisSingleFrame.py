from fileScan import scanFiles
from qSpaceFunctions import createQSpaceMap, createRadialIntensityProfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

## Scan specific settings
# Verbose reporting on file scan results
verbose = False
from configs.FeRh_A06 import (
    folderData,
    sampleName,
    scanNames,
    scanNo,
    minimumFileSizeRatio,
    h5CCDImagePath,
    h5DelayPath,
    delayZero,
    dataIndex,
    N_BINN,
    PIXEL_SIZE,
    LAMBDA,
    CY0,
    CX0,
    DCCD,
    ALPHA,
    OMEGA,
    alignMasks,
    roiAllignMasks,
    roiBG,
    maskBS,
    Q_SPACE_BINS_MAX,
    plotProfileAngles,
    plotProfileAnglesColour,
    ZETA,
    D_ZETA,
    ZETA_SYMMETRY,
    RADIAL_STEP_BIN,
)

## Calling the folder scanning function
results = scanFiles(
    folderData=folderData,
    sampleName=sampleName,
    scanNames=scanNames,
    scanNo=scanNo,
    verbose=verbose,
    minimumFileSizeRatio=minimumFileSizeRatio,
)

qxCenters, qyCenters, intensityQxQy, delayScan, imageCCD = createQSpaceMap(
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
    valid = imageCCD[np.isfinite(imageCCD)]
    vmin = 0.1 * valid.min()
    vmax = 0.1 * valid.max()
    figDet, axDet = plt.subplots()

    height, width = imageCCD.shape
    rowSlice, columnSlice = roiAllignMasks
    y0 = 0 if rowSlice.start is None else rowSlice.start
    x0 = 0 if columnSlice.start is None else columnSlice.start
    detectorExtent = (
        x0 -0.5,           # left
        x0 + width -0.5,   # right
        y0 + height -0.5,  # bottom
        y0 -0.5,           # top, because origin="upper"
    )    
    detmap = axDet.imshow(
        imageCCD,
        origin="upper",
        extent=detectorExtent,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    axDet.set_title(f"Scan {scanNo}, data batch {dataIndex}, delay = {delayScan} ps")
    axDet.set_aspect("equal", adjustable="box")
    figDet.colorbar(detmap, ax=axDet, label="Mean intensity per bin")
    plt.show()
    exit()

## Plot detector image
valid = imageCCD[np.isfinite(imageCCD)]
vmin = 0.1 * valid.min()
vmax = 0.1 * valid.max()
figDet, axDet = plt.subplots()
detmap = axDet.imshow(
    imageCCD,
    origin="upper",
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
)
figDet.colorbar(detmap, ax=axDet, label="Mean intensity per bin")
## Detector-array coordinates are (row, column) = (CY0, CX0). axhline and
## axvline span the full axes, so the crosshair remains complete when zooming.
axDet.axhline(CY0, color="lime", linewidth=3.0, linestyle="-.")
axDet.axvline(CX0, color="lime", linewidth=3.0, linestyle="-.")


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
moveFigure(figDet, 1350, 50)

plt.show()
