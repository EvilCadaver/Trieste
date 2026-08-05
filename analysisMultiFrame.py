from fileScan import scanFiles
from qSpaceFunctions import (
    createQSpaceMap,
    createRadialIntensityProfile,
    subtractPolynomialBackground,
)
import csv
from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings


def getSystemListDelimiter():
    """Return the Windows list separator, falling back to a comma."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Control Panel\International",
        ) as internationalKey:
            delimiter, _ = winreg.QueryValueEx(internationalKey, "sList")

        # csv.writer requires a one-character delimiter.
        if isinstance(delimiter, str) and len(delimiter) == 1:
            return delimiter
    except (ImportError, OSError):
        pass

    return ","


def formatSetting(value):
    """Format settings, including NumPy slice masks, for the CSV header."""
    if isinstance(value, slice):
        start = "" if value.start is None else value.start
        stop = "" if value.stop is None else value.stop
        step = "" if value.step is None else f":{value.step}"
        return f"{start}:{stop}{step}"

    if isinstance(value, tuple):
        return "[" + ", ".join(formatSetting(item) for item in value) + "]"

    if isinstance(value, list):
        return "[" + ", ".join(
            repr(item) if isinstance(item, str) else formatSetting(item)
            for item in value
        ) + "]"

    return str(value)


def fileModificationTime(filePath):
    """Return a timezone-aware ISO timestamp for an acquired data file."""
    return datetime.fromtimestamp(
        Path(filePath).stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")


def saveQDelayData(
    outputPath,
    qValues,
    delayValues,
    intensityValues,
    metadata,
):
    """Save the plotted Q-delay matrix as a sorted long-form CSV table."""
    if intensityValues.shape != (qValues.size, delayValues.size):
        raise ValueError(
            "Intensity shape must equal (number of Q values, number of delays)"
        )

    qOrder = np.argsort(qValues, kind="stable")
    delayOrder = np.argsort(delayValues, kind="stable")
    delimiter = getSystemListDelimiter()

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    # Mode "w" deliberately truncates an existing CSV: outputs are always
    # replaced and the analysis never appends to data from an earlier run.
    with outputPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(
            outputFile,
            delimiter=delimiter,
            lineterminator="\n",
        )

        for name, value in metadata:
            writer.writerow([name, formatSetting(value)])

        writer.writerow(["***DATA***"])
        writer.writerow(["Q", "dt", "Intensity"])
        writer.writerow(["nm^-1", "ps", "a.u."])

        # Q is the primary sort key; delay is increasing within every Q value.
        for qIndex in qOrder:
            for delayIndex in delayOrder:
                writer.writerow(
                    [
                        f"{qValues[qIndex]:.12g}",
                        f"{delayValues[delayIndex]:.12g}",
                        f"{intensityValues[qIndex, delayIndex]:.12g}",
                    ]
                )

    return delimiter


def moveFigure(fig, x, y):
    """Move a Matplotlib figure window to screen position (x, y)."""
    window = getattr(fig.canvas.manager, "window", None)

    if window is None:
        return

    if hasattr(window, "move"):          # Qt backend
        window.move(x, y)
    elif hasattr(window, "wm_geometry"):  # Tk backend
        window.wm_geometry(f"+{x}+{y}")
    else:
        print("Current Matplotlib backend does not support window positioning")


def readDelay(dataFilePath, h5DelayPath, delayZero):
    """Read the delay even for scans that will not undergo image analysis."""
    with h5py.File(dataFilePath, "r") as h5:
        delayValue = np.asarray(h5[h5DelayPath][...]).squeeze()

    if delayValue.size != 1:
        raise ValueError(
            f"Expected one delay value in {dataFilePath}, "
            f"received shape {delayValue.shape}"
        )

    return round(float(-delayValue + delayZero), 1)

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
    dataIndexesExcluded,
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
    Q_LOW_CUTOFF,
    Q_HIGH_CUTOFF,
    BACKGROUND_NPOLY
)

## Output files settings
# Output folder
folderOutput = Path(r"./Output_DS")
folderOutput = folderOutput / sampleName
# Duplicate delays handling rule: "keep first", "propagate delay"
handlingDuplicateDelays = "propagate delay"

## Calling the folder scanning function
results = scanFiles(
    folderData=folderData,
    sampleName=sampleName,
    scanNames=scanNames,
    scanNo=scanNo,
    verbose=verbose,
    minimumFileSizeRatio=minimumFileSizeRatio,
)

if alignMasks:
    raise ValueError(
        "Multi-frame profile analysis requires alignMasks=False"
    )

allDataFiles = results["allDataFiles"]
numberOfScans = len(allDataFiles)
excludedIndexes = set(dataIndexesExcluded)

invalidExcludedIndexes = sorted(
    index
    for index in excludedIndexes
    if not isinstance(index, (int, np.integer))
    or isinstance(index, (bool, np.bool_))
    or index < 0
    or index >= numberOfScans
)
if invalidExcludedIndexes:
    raise ValueError(
        "dataIndexesExcluded contains invalid physical scan indexes: "
        f"{invalidExcludedIndexes}"
    )

# Read delays before processing so excluded and broken scans retain their proper
# columns in the final map. A missing delay is kept as NaN and labelled as such.
delayScans = np.full(numberOfScans, np.nan)
for dataIndex, dataFilePath in enumerate(allDataFiles):
    try:
        delayScans[dataIndex] = readDelay(
            dataFilePath,
            h5DelayPath,
            delayZero,
        )
    except (KeyError, OSError, ValueError) as error:
        print(f"Data [{dataIndex}] delay unavailable: {error}")

distance = None
profileDistance = None
intensityProfiles = None
qxReference = None
qyReference = None

figQspace = None
figProfile = None
figProfiles = None
heatmapQspace = None
lineProfile = None
lineBackground = None
lineCorrectedProfile = None
heatmapProfiles = None
profilesColorbar = None

# The accumulated map uses one equally wide column per physical scan index.
# Delay values are displayed as tick labels. This keeps repeated delays and
# invalid scans as separate columns instead of merging or hiding them.
scanColumns = np.arange(numberOfScans)
tickCount = min(12, numberOfScans)
tickIndexes = np.unique(
    np.linspace(0, numberOfScans - 1, tickCount, dtype=int)
)

plt.ion()

for dataIndex, dataFilePath in enumerate(allDataFiles):
    skippedReason = None

    if dataIndex in excludedIndexes:
        skippedReason = "explicitly excluded"
    elif dataFilePath in results["brokenFiles"]:
        skippedReason = "broken scan"
    elif dataFilePath in results["filesWithoutBackground"]:
        skippedReason = "no suitable background set"

    if skippedReason is None:
        try:
            (
                qxCenters,
                qyCenters,
                intensityQxQy,
                delayScan,
                _,
            ) = createQSpaceMap(
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
                alignMasks=False,
                roiAllignMasks=roiAllignMasks,
                roiBG=roiBG,
                maskBS=maskBS,
                Q_SPACE_BINS_MAX=Q_SPACE_BINS_MAX,
            )

            currentDistance, sumIntensity = createRadialIntensityProfile(
                qxCenters=qxCenters,
                qyCenters=qyCenters,
                intensity=intensityQxQy,
                ZETA=ZETA,
                D_ZETA=D_ZETA,
                ZETA_SYMMETRY=ZETA_SYMMETRY,
                RADIAL_STEP_BIN=RADIAL_STEP_BIN,
            )

            (
                currentCutoffMask,
                backgroundIntensity,
                correctedIntensity,
                qLow,
                qHigh,
            ) = subtractPolynomialBackground(
                distance=currentDistance,
                intensity=sumIntensity,
                Q_LOW_CUTOFF=Q_LOW_CUTOFF,
                Q_HIGH_CUTOFF=Q_HIGH_CUTOFF,
                BACKGROUND_NPOLY=BACKGROUND_NPOLY,
            )
        except (KeyError, OSError, ValueError) as error:
            skippedReason = f"processing failed: {error}"

    if skippedReason is not None:
        print(
            f"Data [{dataIndex}] at delay {delayScans[dataIndex]:g} ps: "
            f"{skippedReason}; storing NaN"
        )

        # Once plotting has been initialized, blank the per-scan displays so a
        # skipped scan is not mistaken for the preceding valid acquisition.
        if heatmapQspace is not None:
            heatmapQspace.set_array(
                np.full(heatmapQspace.get_array().shape, np.nan)
            )
            lineProfile.set_ydata(np.full(distance.shape, np.nan))
            lineBackground.set_ydata(np.full(distance.shape, np.nan))
            lineCorrectedProfile.set_ydata(
                np.full(distance.shape, np.nan)
            )
            axQspace.set_title(
                f"Scan {scanNo}, data batch {dataIndex}: {skippedReason}"
            )
            axProfile.set_title(
                f"Data batch {dataIndex}: {skippedReason}"
            )
    else:
        delayScans[dataIndex] = delayScan

        if distance is None:
            # Geometry and binning settings are constant, so every valid scan
            # must produce these same axes.
            distance = currentDistance
            cutoffMask = currentCutoffMask
            profileDistance = distance[cutoffMask]
            qLowReference = qLow
            qHighReference = qHigh
            qxReference = qxCenters
            qyReference = qyCenters
            intensityProfiles = np.full(
                (profileDistance.size, numberOfScans),
                np.nan,
            )

            finiteIntensity = intensityQxQy[np.isfinite(intensityQxQy)]
            colourLimit = np.percentile(np.abs(finiteIntensity), 99)
            if colourLimit == 0:
                colourLimit = 1.0

            qxSpan = np.ptp(qxCenters)
            qySpan = np.ptp(qyCenters)
            longSide = 7.0
            if qxSpan >= qySpan:
                plotWidth = longSide
                plotHeight = longSide * qySpan / qxSpan
            else:
                plotHeight = longSide
                plotWidth = longSide * qxSpan / qySpan

            figQspace, axQspace = plt.subplots(
                figsize=(plotWidth + 1.2, plotHeight),
                layout="constrained",
            )
            heatmapQspace = axQspace.pcolormesh(
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
                angleGrid = np.degrees(np.arctan2(qyGrid, qxGrid))
                symmetryPeriod = 360.0 / ZETA_SYMMETRY
                angleDifference = (
                    (angleGrid - ZETA + symmetryPeriod / 2)
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

            axQspace.set_aspect("equal", adjustable="box")
            axQspace.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
            axQspace.set_ylabel(r"$Q_y$ (nm$^{-1}$)")
            figQspace.colorbar(
                heatmapQspace,
                ax=axQspace,
                label="Mean intensity per bin",
            )

            figProfile, axProfile = plt.subplots(layout="constrained")
            (lineProfile,) = axProfile.plot(
                distance,
                sumIntensity,
                label="Original",
            )
            (lineBackground,) = axProfile.plot(
                distance,
                backgroundIntensity,
                label=(
                    "Background fitted after cutoffs "
                    f"(order {BACKGROUND_NPOLY})"
                ),
            )
            (lineCorrectedProfile,) = axProfile.plot(
                distance,
                correctedIntensity,
                label="Profile - background",
            )
            axProfile.axvline(
                qLowReference,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"Low cutoff ({Q_LOW_CUTOFF:g}%)",
            )
            axProfile.axvline(
                qHighReference,
                color="black",
                linestyle=":",
                linewidth=1,
                label=f"High cutoff ({Q_HIGH_CUTOFF:g}%)",
            )
            axProfile.set_xlabel(r"$|Q|$ (nm$^{-1}$)")
            axProfile.set_ylabel("Summed intensity")
            axProfile.grid(True, alpha=0.3)
            axProfile.legend()

            profileCmap = plt.get_cmap("seismic").copy()
            profileCmap.set_bad("lightgray")
            radialStep = (
                profileDistance[1] - profileDistance[0]
                if profileDistance.size > 1
                else 1.0
            )
            figProfiles, axProfiles = plt.subplots(
                figsize=(11, 7),
                layout="constrained",
            )
            heatmapProfiles = axProfiles.imshow(
                intensityProfiles,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[
                    -0.5,
                    numberOfScans - 0.5,
                    profileDistance[0] - radialStep / 2,
                    profileDistance[-1] + radialStep / 2,
                ],
                cmap=profileCmap,
            )
            axProfiles.set_xlabel("Delay (ps; one column per physical scan)")
            axProfiles.set_ylabel(r"$|Q|$ (nm$^{-1}$)")
            axProfiles.set_xticks(tickIndexes)
            axProfiles.set_xticklabels(
                [f"{delayScans[index]:g}" for index in tickIndexes]
            )
            profilesColorbar = figProfiles.colorbar(
                heatmapProfiles,
                ax=axProfiles,
                label="Background-subtracted summed intensity",
            )

            moveFigure(figQspace, 20, 100)
            moveFigure(figProfile, 600, 100)
            moveFigure(figProfiles, 1300, 100)
        else:
            if not (
                np.array_equal(qxCenters.shape, qxReference.shape)
                and np.array_equal(qyCenters.shape, qyReference.shape)
                and np.allclose(qxCenters, qxReference)
                and np.allclose(qyCenters, qyReference)
                and currentDistance.shape == distance.shape
                and np.allclose(currentDistance, distance)
                and np.array_equal(currentCutoffMask, cutoffMask)
                and np.isclose(qLow, qLowReference)
                and np.isclose(qHigh, qHighReference)
            ):
                raise ValueError(
                    f"Data [{dataIndex}] produced incompatible Q/profile axes"
                )

            heatmapQspace.set_array(intensityQxQy.ravel())
            finiteIntensity = intensityQxQy[np.isfinite(intensityQxQy)]
            colourLimit = np.percentile(np.abs(finiteIntensity), 99)
            if colourLimit == 0:
                colourLimit = 1.0
            heatmapQspace.set_clim(-colourLimit, colourLimit)
            lineProfile.set_ydata(sumIntensity)
            lineBackground.set_ydata(backgroundIntensity)
            lineCorrectedProfile.set_ydata(correctedIntensity)

        intensityProfiles[:, dataIndex] = correctedIntensity[cutoffMask]
        axQspace.set_title(
            f"Scan {scanNo}, data batch {dataIndex}, "
            f"delay={delayScan:g} ps"
        )
        axProfile.set_title(
            f"Zetta={ZETA:g} deg, symmetry={ZETA_SYMMETRY}, "
            f"acceptance={D_ZETA:g} deg, delay={delayScan:g} ps"
        )
        axProfile.relim()
        axProfile.autoscale_view()

    if heatmapProfiles is not None:
        heatmapProfiles.set_data(intensityProfiles)
        finiteProfiles = intensityProfiles[np.isfinite(intensityProfiles)]
        if finiteProfiles.size:
            profileLimit = np.percentile(np.abs(finiteProfiles), 99)
            if profileLimit == 0:
                profileLimit = 1.0
            heatmapProfiles.set_clim(-profileLimit, profileLimit)

        axProfiles.set_title(
            f"Scan {scanNo}: processed through data batch {dataIndex} "
            f"of {numberOfScans - 1}"
        )

        if getattr(figProfiles.canvas.manager, "window", None) is not None:
            for figure in (figQspace, figProfile, figProfiles):
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            plt.pause(0.001)

if intensityProfiles is None:
    raise RuntimeError("No valid scans were available for profile analysis")

validScanCount = np.count_nonzero(
    np.any(np.isfinite(intensityProfiles), axis=0)
)
print(
    f"Finished: {validScanCount} valid profiles, "
    f"{numberOfScans - validScanCount} NaN scan columns"
)

# Group physical scan indexes by their measured delay. Delays were rounded to
# one decimal place when read, so equality here represents the acquisition
# metadata at the same precision used throughout the analysis.
delayToIndexes = {}
for dataIndex, delayScan in enumerate(delayScans):
    if np.isfinite(delayScan):
        delayToIndexes.setdefault(float(delayScan), []).append(dataIndex)

duplicateDelays = {
    delayScan: dataIndexes
    for delayScan, dataIndexes in delayToIndexes.items()
    if len(dataIndexes) > 1
}

# Infer the intended delay step from the median positive spacing between unique
# delays. Gaps that are integer multiples of this step identify delay values
# that should have been present in the acquisition sequence.
uniqueDelays = np.array(sorted(delayToIndexes), dtype=float)
suspectedMissingScans = []
nominalDelayStep = None

if uniqueDelays.size == 0:
    raise RuntimeError("No finite delay values were available for export")

if uniqueDelays.size > 1:
    uniqueDelaySteps = np.diff(uniqueDelays)
    positiveDelaySteps = uniqueDelaySteps[uniqueDelaySteps > 0]

    if positiveDelaySteps.size:
        nominalDelayStep = float(np.median(positiveDelaySteps))
        delayTolerance = max(1e-9, 0.1 * nominalDelayStep)

        for leftDelay, rightDelay in zip(
            uniqueDelays[:-1],
            uniqueDelays[1:],
        ):
            gap = rightDelay - leftDelay
            numberOfSteps = int(round(gap / nominalDelayStep))

            if (
                numberOfSteps > 1
                and np.isclose(
                    gap,
                    numberOfSteps * nominalDelayStep,
                    atol=delayTolerance,
                    rtol=0,
                )
            ):
                for stepIndex in range(1, numberOfSteps):
                    suspectedMissingScans.append(
                        {
                            "delay": leftDelay + stepIndex * nominalDelayStep,
                            "leftIndexes": delayToIndexes[float(leftDelay)],
                            "rightIndexes": delayToIndexes[float(rightDelay)],
                        }
                    )

missingDelayValues = np.array(
    [missingScan["delay"] for missingScan in suspectedMissingScans],
    dtype=float,
)

# "keep first" is also the safe baseline for "propagate delay". The first
# acquisition at every recorded delay is retained, and all later acquisitions
# at that same delay are initially considered unused duplicate batches.
figureDelays = uniqueDelays.copy()
keptDataIndexes = np.array(
    [delayToIndexes[delayScan][0] for delayScan in figureDelays],
    dtype=int,
)
duplicateCandidateIndexes = sorted(
    dataIndex
    for dataIndexes in duplicateDelays.values()
    for dataIndex in dataIndexes[1:]
)
discardedDuplicateIndexes = duplicateCandidateIndexes.copy()
missingDelayValuesToInsert = missingDelayValues.copy()
propagatedDelayAssignments = []
appliedDuplicateDelayHandling = handlingDuplicateDelays

match handlingDuplicateDelays:
    case "keep first":
        pass

    case "propagate delay":
        if len(duplicateCandidateIndexes) < missingDelayValues.size:
            appliedDuplicateDelayHandling = "keep first"
            warnings.warn(
                "The 'propagate delay' rule found "
                f"{missingDelayValues.size} missing delays but only "
                f"{len(duplicateCandidateIndexes)} unused duplicate batches. "
                "Falling back to 'keep first'.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # Pair physical batches and missing delays in increasing order. For
            # scan 27 this maps 38->330, 39->340, 41->360, and 42->370 ps.
            propagatedIndexes = np.array(
                duplicateCandidateIndexes[:missingDelayValues.size],
                dtype=int,
            )
            figureDelays = np.concatenate(
                (figureDelays, missingDelayValues)
            )
            keptDataIndexes = np.concatenate(
                (keptDataIndexes, propagatedIndexes)
            )
            figureDelayOrder = np.argsort(figureDelays, kind="stable")
            figureDelays = figureDelays[figureDelayOrder]
            keptDataIndexes = keptDataIndexes[figureDelayOrder]

            propagatedDelayAssignments = [
                (
                    int(dataIndex),
                    float(delayScans[dataIndex]),
                    float(propagatedDelay),
                )
                for dataIndex, propagatedDelay in zip(
                    propagatedIndexes,
                    missingDelayValues,
                )
            ]
            propagatedIndexSet = set(propagatedIndexes.tolist())
            discardedDuplicateIndexes = [
                dataIndex
                for dataIndex in duplicateCandidateIndexes
                if dataIndex not in propagatedIndexSet
            ]
            missingDelayValuesToInsert = np.array([], dtype=float)

    case _:
        raise ValueError(
            "Unknown handlingDuplicateDelays value "
            f"{handlingDuplicateDelays!r}. Available choices: "
            "'keep first', 'propagate delay'"
        )

figureIntensityProfiles = intensityProfiles[:, keptDataIndexes]

duplicateFigures = []

if duplicateDelays:
    print("\nWARNING: duplicate delay acquisitions detected:")

    for duplicateNumber, (delayScan, dataIndexes) in enumerate(
        sorted(duplicateDelays.items())
    ):
        print(f"  Delay {delayScan:g} ps: data indexes {dataIndexes}")

        # Compare the final profiles actually used by the delay map: both Q
        # cutoffs have already been applied and the polynomial backgrounds have
        # already been subtracted independently for every acquisition.
        figDuplicate, axDuplicate = plt.subplots(layout="constrained")

        for dataIndex in dataIndexes:
            duplicateProfile = intensityProfiles[:, dataIndex]

            if np.any(np.isfinite(duplicateProfile)):
                axDuplicate.plot(
                    profileDistance,
                    duplicateProfile,
                    label=f"Data index {dataIndex}",
                )
            else:
                # Add a legend entry even when this duplicate was excluded,
                # broken, or otherwise unavailable.
                axDuplicate.plot(
                    [],
                    [],
                    label=f"Data index {dataIndex} (NaN)",
                )

        axDuplicate.set_xlabel(r"$|Q|$ (nm$^{-1}$)")
        axDuplicate.set_ylabel(
            "Background-subtracted summed intensity"
        )
        axDuplicate.set_title(
            f"Duplicate acquisitions at delay {delayScan:g} ps"
        )
        axDuplicate.grid(True, alpha=0.3)
        axDuplicate.legend()
        duplicateFigures.append(figDuplicate)
        moveFigure(
            figDuplicate,
            1050,
            100 + 60 * duplicateNumber,
        )

if suspectedMissingScans:
    stepText = (
        f" using inferred delay step {nominalDelayStep:g} ps"
        if nominalDelayStep is not None
        else ""
    )
    print(f"\nWARNING: suspected missing scans{stepText}:")

    for missingScan in suspectedMissingScans:
        print(
            f"  Expected delay {missingScan['delay']:g} ps between "
            f"data indexes {missingScan['leftIndexes']} and "
            f"{missingScan['rightIndexes']}"
        )
else:
    print("\nNo internal missing delay steps detected.")

if propagatedDelayAssignments:
    print("\nPropagated duplicate batches to missing delays:")
    for dataIndex, recordedDelay, propagatedDelay in propagatedDelayAssignments:
        print(
            f"  Data index {dataIndex}: recorded {recordedDelay:g} ps, "
            f"assigned {propagatedDelay:g} ps"
        )

# Under "keep first", or after a propagation fallback, represent every still
# unfilled delay as an explicit NaN column. Successful propagation leaves this
# list empty because real profile columns now occupy all inferred delay steps.
if missingDelayValuesToInsert.size:
    missingProfiles = np.full(
        (profileDistance.size, missingDelayValuesToInsert.size),
        np.nan,
    )
    figureDelays = np.concatenate(
        (figureDelays, missingDelayValuesToInsert)
    )
    figureIntensityProfiles = np.concatenate(
        (figureIntensityProfiles, missingProfiles),
        axis=1,
    )
    figureDelayOrder = np.argsort(figureDelays, kind="stable")
    figureDelays = figureDelays[figureDelayOrder]
    figureIntensityProfiles = figureIntensityProfiles[:, figureDelayOrder]

# Replace the live physical-index view with the final delay-coordinate figure.
# This same deduplicated matrix is written below, so the CSV and figure agree.
profilesColorbar.remove()
axProfiles.clear()
finiteFigureProfiles = figureIntensityProfiles[
    np.isfinite(figureIntensityProfiles)
]
profileLimit = (
    np.percentile(np.abs(finiteFigureProfiles), 99)
    if finiteFigureProfiles.size
    else 1.0
)
if profileLimit == 0:
    profileLimit = 1.0

heatmapProfiles = axProfiles.pcolormesh(
    figureDelays,
    profileDistance,
    figureIntensityProfiles,
    shading="nearest",
    cmap=profileCmap,
    vmin=-profileLimit,
    vmax=profileLimit,
)
axProfiles.set_xlabel("Delay (ps)")
axProfiles.set_ylabel(r"$|Q|$ (nm$^{-1}$)")
axProfiles.set_title(
    f"Scan {scanNo}: background-subtracted profiles versus delay"
)
figProfiles.colorbar(
    heatmapProfiles,
    ax=axProfiles,
    label="Background-subtracted summed intensity",
)

# Batch times use filesystem metadata because the HDF5 files do not expose a
# clear acquisition timestamp. Record the paths and label the timestamps
# explicitly to keep that provenance unambiguous.
firstUsedIndex = int(np.min(keptDataIndexes))
lastUsedIndex = int(np.max(keptDataIndexes))
firstUsedFile = Path(allDataFiles[firstUsedIndex])
lastUsedFile = Path(allDataFiles[lastUsedIndex])
outputPath = folderOutput / (
    f"{sampleName}_Scan_{scanNo:03d}_Q_vs_delay_"
    f"a{ZETA:g}deg_d{D_ZETA:g}deg_sym{ZETA_SYMMETRY:g}.csv"
)
pngOutputPath = outputPath.with_suffix(".png")
brokenDataIndexes = [
    dataIndex
    for dataIndex, dataFilePath in enumerate(allDataFiles)
    if dataFilePath in results["brokenFiles"]
]
indexesWithoutBackground = [
    dataIndex
    for dataIndex, dataFilePath in enumerate(allDataFiles)
    if dataFilePath in results["filesWithoutBackground"]
]

metadata = [
    ("First data batch physical index", firstUsedIndex),
    ("First data batch file", firstUsedFile),
    ("First data batch file modification time", fileModificationTime(firstUsedFile)),
    ("Last data batch physical index", lastUsedIndex),
    ("Last data batch file", lastUsedFile),
    ("Last data batch file modification time", fileModificationTime(lastUsedFile)),
    ("handlingDuplicateDelays requested", handlingDuplicateDelays),
    ("handlingDuplicateDelays applied", appliedDuplicateDelayHandling),
    ("Propagated delay assignments (index, recorded ps, assigned ps)",
     propagatedDelayAssignments),
    ("Physical data indexes represented", keptDataIndexes.tolist()),
    ("Discarded duplicate physical indexes", discardedDuplicateIndexes),
    ("Detected missing delays (ps)", missingDelayValues.tolist()),
    (
        "Inserted missing delays as NaN (ps)",
        missingDelayValuesToInsert.tolist(),
    ),
    ("Broken physical data indexes", brokenDataIndexes),
    ("Physical data indexes without background", indexesWithoutBackground),
    ("folderData", folderData),
    ("folderOutput", folderOutput),
    ("sampleName", sampleName),
    ("scanNames", scanNames),
    ("scanNo", scanNo),
    ("minimumFileSizeRatio", minimumFileSizeRatio),
    ("verbose", verbose),
    ("h5CCDImagePath", h5CCDImagePath),
    ("h5DelayPath", h5DelayPath),
    ("delayZero", delayZero),
    ("dataIndexesExcluded", dataIndexesExcluded),
    ("N_BINN", N_BINN),
    ("PIXEL_SIZE_m", PIXEL_SIZE),
    ("LAMBDA_m", LAMBDA),
    ("CY0_pixels", CY0),
    ("CX0_pixels", CX0),
    ("DCCD_m", DCCD),
    ("ALPHA_rad", ALPHA),
    ("OMEGA_rad", OMEGA),
    ("alignMasks", alignMasks),
    ("roiAllignMasks_y_x", roiAllignMasks),
    ("roiBG_y_x", roiBG),
    ("maskBS_y_x", maskBS),
    ("Q_SPACE_BINS_MAX", Q_SPACE_BINS_MAX),
    ("plotProfileAngles", plotProfileAngles),
    ("plotProfileAnglesColour", plotProfileAnglesColour),
    ("ZETA_deg", ZETA),
    ("D_ZETA_deg", D_ZETA),
    ("ZETA_SYMMETRY", ZETA_SYMMETRY),
    ("RADIAL_STEP_BIN", RADIAL_STEP_BIN),
    ("Q_LOW_CUTOFF_percent", Q_LOW_CUTOFF),
    ("Q_HIGH_CUTOFF_percent", Q_HIGH_CUTOFF),
    ("BACKGROUND_NPOLY", BACKGROUND_NPOLY),
]

existingOutputPaths = [
    path for path in (outputPath, pngOutputPath) if path.exists()
]
if existingOutputPaths:
    print("Overwriting existing output file(s):")
    for existingOutputPath in existingOutputPaths:
        print(f"  {existingOutputPath}")

delimiter = saveQDelayData(
    outputPath=outputPath,
    qValues=profileDistance,
    delayValues=figureDelays,
    intensityValues=figureIntensityProfiles,
    metadata=metadata,
)
print(f"Saved Q-vs-delay data to {outputPath} (delimiter {delimiter!r})")
# Matplotlib's savefig replaces an existing file at the same path.
figProfiles.savefig(
    pngOutputPath,
    dpi=300,
    bbox_inches="tight",
)
print(f"Saved Q-vs-delay figure to {pngOutputPath} (300 dpi)")

plt.ioff()

# Desktop backends expose a window and should remain open after processing.
# Headless backends (used for automated validation) have nothing to display.
if getattr(figProfiles.canvas.manager, "window", None) is None:
    plt.close("all")
else:
    plt.show()
