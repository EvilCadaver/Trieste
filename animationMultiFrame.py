"""Render a multi-frame Q-space scan as still images and/or videos."""

from pathlib import Path

import matplotlib.animation as mplAnimation
import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    TwoSlopeNorm,
    is_color_like,
)
from matplotlib.patches import Patch
import numpy as np

from fileScan import scanFiles
from qSpaceFunctions import createQSpaceMap


def generateThreeColourMap(colours, name="three_colour_mix", samples=256):
    """Return a continuous Matplotlib colourmap passing through three colours."""
    if len(colours) != 3:
        raise ValueError("A three-colour map requires exactly three colours")
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("COLOURMAP_SAMPLES must be an integer of at least 2")

    return LinearSegmentedColormap.from_list(name, colours, N=int(samples))


# -----------------------------------------------------------------------------
# Animation and image output settings
# -----------------------------------------------------------------------------
# Enable either or both output styles.
OUTPUT_IMAGES = True
OUTPUT_VIDEOS = True

# Formats are written without a leading dot. Multiple formats may be requested.
IMAGE_FORMATS = ("png",)
VIDEO_FORMATS = ("mp4",)
FRAMES_PER_SECOND = 6
IMAGE_DPI = 200

# Output geometry. The axes retain equal Qx/Qy scaling inside this figure size.
OUTPUT_ASPECT_RATIO = (16, 9)
OUTPUT_WIDTH_INCHES = 12.0
AXES_ASPECT = "equal"

# Enable either or both coordinate-space exports. Detector pixel space uses the
# background-corrected differential detector image before Q-space rebinning;
# detector row 0 is displayed at the top. Reciprocal space uses Qx/Qy in nm^-1.
OUTPUT_DETECTOR_PIXEL_SPACE = True
OUTPUT_RECIPROCAL_SPACE = False

# Optionally mark the detector-space beam centre used by the geometry (CX0,
# CY0). The cross has a fixed 1:1 physical size; 0.2 means 0.2 x 0.2 inches.
# Its horizontal and vertical segments are always solid.
BEAM_CENTRE_ON = True
BEAM_CENTRE_COLOUR = "#00FF0D"
BEAM_CENTRE_LINE_WIDTH_POINTS = 1.0
BEAM_CENTRE_SIZE_INCHES = 0.2

# Figure-relative rectangle: (left, bottom, width, height). The plot rectangle
# is symmetric about the horizontal centre of the output frame. The colourbar
# is placed after Matplotlib applies the equal aspect ratio, so its distance is
# measured from the actual heatmap rather than from this containing rectangle.
AXES_POSITION = (0.12, 0.14, 0.76, 0.74)
COLOURBAR_DISTANCE_INCHES = 0.25
COLOURBAR_WIDTH_INCHES = 0.18

# Text shown on the plot. GRAPH_TITLE supports these named format fields.
RECIPROCAL_X_AXIS_NAME = r"$Q_x$ (nm$^{-1}$)"
RECIPROCAL_Y_AXIS_NAME = r"$Q_y$ (nm$^{-1}$)"
DETECTOR_X_AXIS_NAME = "Detector column (pixels)"
DETECTOR_Y_AXIS_NAME = "Detector row (pixels; origin at top)"
GRAPH_TITLE = "{sampleName}, scan {scanNo}: differential intensity"
COLOURBAR_NAME = "Mean differential intensity per bin"

# Delay text is anchored inside the actual heatmap by distances in inches:
# (distance from left, distance from top). DELAY_INFO supports delay and
# dataIndex fields.
DELAY_INFO = r"$\Delta t$ = {delay:g} ps"
DELAY_INFO_OFFSET_INCHES = (0.1, 0.1)
DELAY_INFO_HORIZONTAL_ALIGNMENT = "left"
DELAY_INFO_VERTICAL_ALIGNMENT = "top"

# Font family and sizes in points. Reasonable FONT_FAMILY choices include
# "DejaVu Sans" (Matplotlib default), "Arial", "Tahoma", "Times New Roman",
# or the generic families "sans-serif" and "serif".
# Matplotlib falls back to a similar installed font if a named font is absent.
FONT_FAMILY = "DejaVu Sans"
GRAPH_TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
AXIS_TICK_FONT_SIZE = 10
DELAY_INFO_FONT_SIZE = 11
COLOURBAR_LABEL_FONT_SIZE = 12
COLOURBAR_TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10

# A heatmap does not normally need a legend because it has a colourbar. If a
# legend is enabled, it identifies the intensity layer and uses this position.
LEGEND_ON = False
LEGEND_POSITION = "upper right"
LEGEND_BBOX_TO_ANCHOR = None
LEGEND_LABEL = "Differential intensity"

# Three-colour mixing. Negative and positive limits are found independently,
# each at COLOUR_LIMIT_PERCENTILE. With ZERO_INTENSITY_IS_SECOND_COLOUR=True,
# zero is pinned to the second colour even when those two limits differ.
COLOURMAP_GENERATOR = generateThreeColourMap
COLOURMAP_COLOURS = ("#2166ac", "#f7f7f7", "#b2182b")
COLOURMAP_NAME = "negative_zero_positive"
COLOURMAP_SAMPLES = 512
COLOURMAP_BAD_COLOUR = "lightgray"
COLOUR_LIMIT_PERCENTILE = 95.0
ZERO_INTENSITY_IS_SECOND_COLOUR = False

# Shared limits prevent colour flicker and make intensities comparable between
# frames. Set False to stretch the negative and positive colours per frame.
COLOUR_LIMITS_ACROSS_ALL_FRAMES = False

# Files are placed below folderOutput/sampleName in this subfolder.
OUTPUT_SUBFOLDER = "Animations"

# Matplotlib chooses a writer through this map; no non-Matplotlib plotting or
# video API is used. FFmpeg must be installed for its associated formats.
VIDEO_WRITERS = {
    "mp4": "ffmpeg",
    "mkv": "ffmpeg",
    "avi": "ffmpeg",
    "mov": "ffmpeg",
    "webm": "ffmpeg",
    "gif": "pillow",
    "apng": "pillow",
    "webp": "pillow",
}


# -----------------------------------------------------------------------------
# Scan-specific settings
# -----------------------------------------------------------------------------
verbose = False
from configs.FeRh_A04_S40_50 import (
    folderData,
    sampleName,
    scanNames,
    scanNo,
    minimumFileSizeRatio,
    h5CCDImagePath,
    h5DelayPath,
    delayZero,
    dataIndexesExcluded,
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
)

# Base output folder, matching analysisMultiFrame.py.
folderOutput = Path(r"./Output_DS") / sampleName


def normalizedFormats(formats, settingName):
    """Return validated, lower-case output extensions."""
    if isinstance(formats, str):
        raise TypeError(f"{settingName} must be a tuple/list, not a string")

    normalized = []
    for outputFormat in formats:
        if not isinstance(outputFormat, str) or not outputFormat.strip(". "):
            raise ValueError(f"{settingName} contains an invalid format")
        normalized.append(outputFormat.lower().strip(". "))

    if not normalized:
        raise ValueError(f"{settingName} must contain at least one format")

    return tuple(dict.fromkeys(normalized))


def validateOutputSettings():
    """Validate output constants before scanning any data."""
    if not OUTPUT_IMAGES and not OUTPUT_VIDEOS:
        raise ValueError("Enable OUTPUT_IMAGES and/or OUTPUT_VIDEOS")
    if not OUTPUT_DETECTOR_PIXEL_SPACE and not OUTPUT_RECIPROCAL_SPACE:
        raise ValueError(
            "Enable OUTPUT_DETECTOR_PIXEL_SPACE and/or "
            "OUTPUT_RECIPROCAL_SPACE"
        )
    if not isinstance(BEAM_CENTRE_ON, (bool, np.bool_)):
        raise TypeError("BEAM_CENTRE_ON must be True or False")
    if not is_color_like(BEAM_CENTRE_COLOUR):
        raise ValueError("BEAM_CENTRE_COLOUR is not a valid Matplotlib colour")

    for settingName, value in (
        ("BEAM_CENTRE_LINE_WIDTH_POINTS", BEAM_CENTRE_LINE_WIDTH_POINTS),
        ("BEAM_CENTRE_SIZE_INCHES", BEAM_CENTRE_SIZE_INCHES),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{settingName} must be positive and finite")

    if (
        isinstance(FRAMES_PER_SECOND, (bool, np.bool_))
        or not isinstance(
            FRAMES_PER_SECOND,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(FRAMES_PER_SECOND)
        or FRAMES_PER_SECOND <= 0
    ):
        raise ValueError("FRAMES_PER_SECOND must be a positive finite number")

    if (
        not isinstance(OUTPUT_ASPECT_RATIO, (tuple, list))
        or len(OUTPUT_ASPECT_RATIO) != 2
        or not all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0
            for value in OUTPUT_ASPECT_RATIO
        )
    ):
        raise ValueError("OUTPUT_ASPECT_RATIO must contain two positive numbers")

    if not np.isfinite(OUTPUT_WIDTH_INCHES) or OUTPUT_WIDTH_INCHES <= 0:
        raise ValueError("OUTPUT_WIDTH_INCHES must be positive and finite")
    if not 0 < COLOUR_LIMIT_PERCENTILE <= 100:
        raise ValueError("COLOUR_LIMIT_PERCENTILE must be in (0, 100]")
    if not isinstance(FONT_FAMILY, str) or not FONT_FAMILY.strip():
        raise ValueError("FONT_FAMILY must be a non-empty string")

    for settingName, value in (
        ("GRAPH_TITLE_FONT_SIZE", GRAPH_TITLE_FONT_SIZE),
        ("AXIS_LABEL_FONT_SIZE", AXIS_LABEL_FONT_SIZE),
        ("AXIS_TICK_FONT_SIZE", AXIS_TICK_FONT_SIZE),
        ("DELAY_INFO_FONT_SIZE", DELAY_INFO_FONT_SIZE),
        ("COLOURBAR_LABEL_FONT_SIZE", COLOURBAR_LABEL_FONT_SIZE),
        ("COLOURBAR_TICK_FONT_SIZE", COLOURBAR_TICK_FONT_SIZE),
        ("LEGEND_FONT_SIZE", LEGEND_FONT_SIZE),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{settingName} must be positive and finite")
    if (
        not isinstance(DELAY_INFO_OFFSET_INCHES, (tuple, list))
        or len(DELAY_INFO_OFFSET_INCHES) != 2
        or not all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value >= 0
            for value in DELAY_INFO_OFFSET_INCHES
        )
    ):
        raise ValueError(
            "DELAY_INFO_OFFSET_INCHES must contain two finite, non-negative "
            "numbers"
        )

    if (
        not isinstance(AXES_POSITION, (tuple, list))
        or len(AXES_POSITION) != 4
        or not all(np.isfinite(value) for value in AXES_POSITION)
        or AXES_POSITION[0] < 0
        or AXES_POSITION[1] < 0
        or AXES_POSITION[2] <= 0
        or AXES_POSITION[3] <= 0
        or AXES_POSITION[0] + AXES_POSITION[2] > 1
        or AXES_POSITION[1] + AXES_POSITION[3] > 1
    ):
        raise ValueError(
            "AXES_POSITION must be a positive rectangle within the figure"
        )

    for settingName, value in (
        ("COLOURBAR_DISTANCE_INCHES", COLOURBAR_DISTANCE_INCHES),
        ("COLOURBAR_WIDTH_INCHES", COLOURBAR_WIDTH_INCHES),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{settingName} must be positive and finite")

    imageFormats = (
        normalizedFormats(IMAGE_FORMATS, "IMAGE_FORMATS")
        if OUTPUT_IMAGES
        else ()
    )
    videoFormats = (
        normalizedFormats(VIDEO_FORMATS, "VIDEO_FORMATS")
        if OUTPUT_VIDEOS
        else ()
    )

    unsupportedImageFormats = sorted(
        set(imageFormats) - set(plt.figure().canvas.get_supported_filetypes())
    )
    plt.close()
    if OUTPUT_IMAGES and unsupportedImageFormats:
        raise ValueError(
            "Unsupported IMAGE_FORMATS for this Matplotlib backend: "
            f"{unsupportedImageFormats}"
        )

    unsupportedVideoFormats = sorted(set(videoFormats) - set(VIDEO_WRITERS))
    if OUTPUT_VIDEOS and unsupportedVideoFormats:
        raise ValueError(
            "Unsupported VIDEO_FORMATS: "
            f"{unsupportedVideoFormats}; available: {sorted(VIDEO_WRITERS)}"
        )

    return imageFormats, videoFormats


def calculateColourLimits(intensityValues):
    """Return independent percentile limits below and above zero."""
    finiteValues = np.asarray(intensityValues)[np.isfinite(intensityValues)]
    if finiteValues.size == 0:
        return -1.0, 1.0

    negativeValues = -finiteValues[finiteValues < 0]
    positiveValues = finiteValues[finiteValues > 0]

    negativeLimit = (
        float(np.percentile(negativeValues, COLOUR_LIMIT_PERCENTILE))
        if negativeValues.size
        else 0.0
    )
    positiveLimit = (
        float(np.percentile(positiveValues, COLOUR_LIMIT_PERCENTILE))
        if positiveValues.size
        else 0.0
    )

    overallScale = max(negativeLimit, positiveLimit)
    if not np.isfinite(overallScale) or overallScale <= 0:
        return -1.0, 1.0

    # TwoSlopeNorm requires strict vmin < 0 < vmax. Retain an extremely small
    # opposite-sign interval when a data set contains only one sign.
    minimumLimit = max(np.finfo(float).eps, overallScale * 1e-12)
    negativeLimit = max(negativeLimit, minimumLimit)
    positiveLimit = max(positiveLimit, minimumLimit)
    return -negativeLimit, positiveLimit


def createColourNorm(intensityValues):
    """Create the selected zero-aware or linearly centred normalization."""
    colourMinimum, colourMaximum = calculateColourLimits(intensityValues)
    if ZERO_INTENSITY_IS_SECOND_COLOUR:
        return TwoSlopeNorm(
            vmin=colourMinimum,
            vcenter=0.0,
            vmax=colourMaximum,
        )
    return Normalize(vmin=colourMinimum, vmax=colourMaximum)


def figureSize():
    """Convert width and aspect-ratio settings to a Matplotlib figure size."""
    aspectWidth, aspectHeight = OUTPUT_ASPECT_RATIO
    return (
        float(OUTPUT_WIDTH_INCHES),
        float(OUTPUT_WIDTH_INCHES) * aspectHeight / aspectWidth,
    )


def addColourbarAxes(fig, ax):
    """Add colourbar axes at the configured distance from the heatmap axes."""
    # Drawing applies AXES_ASPECT and resolves the heatmap's final position.
    fig.canvas.draw()
    heatmapBounds = ax.get_position()
    figureWidthInches = fig.get_size_inches()[0]
    distance = COLOURBAR_DISTANCE_INCHES / figureWidthInches
    width = COLOURBAR_WIDTH_INCHES / figureWidthInches
    colourbarLeft = heatmapBounds.x1 + distance

    if colourbarLeft + width > 1:
        raise ValueError(
            "The colourbar does not fit in the output frame; reduce "
            "COLOURBAR_DISTANCE_INCHES or COLOURBAR_WIDTH_INCHES"
        )

    return fig.add_axes(
        [
            colourbarLeft,
            heatmapBounds.y0,
            width,
            heatmapBounds.height,
        ]
    )


def delayInfoPosition(fig, ax):
    """Return a figure position at fixed left/top distances from the heatmap."""
    heatmapBounds = ax.get_position()
    figureWidthInches, figureHeightInches = fig.get_size_inches()
    leftOffset, topOffset = DELAY_INFO_OFFSET_INCHES
    heatmapWidthInches = heatmapBounds.width * figureWidthInches
    heatmapHeightInches = heatmapBounds.height * figureHeightInches

    if leftOffset > heatmapWidthInches or topOffset > heatmapHeightInches:
        raise ValueError(
            "DELAY_INFO_OFFSET_INCHES places the delay anchor outside the "
            "heatmap"
        )

    return (
        heatmapBounds.x0 + leftOffset / figureWidthInches,
        heatmapBounds.y1 - topOffset / figureHeightInches,
    )


def addBeamCentre(fig, ax):
    """Draw a fixed-physical-size cross at the configured detector centre."""
    # Resolve the equal-aspect axes size before converting inches to pixels.
    fig.canvas.draw()
    axesSizeInches = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    xLimits = ax.get_xlim()
    yLimits = ax.get_ylim()
    xDataPerInch = abs(np.diff(xLimits)[0]) / axesSizeInches.width
    yDataPerInch = abs(np.diff(yLimits)[0]) / axesSizeInches.height
    xHalfSize = 0.5 * BEAM_CENTRE_SIZE_INCHES * xDataPerInch
    yHalfSize = 0.5 * BEAM_CENTRE_SIZE_INCHES * yDataPerInch

    horizontalLine = ax.plot(
        [CX0 - xHalfSize, CX0 + xHalfSize],
        [CY0, CY0],
        color=BEAM_CENTRE_COLOUR,
        linewidth=BEAM_CENTRE_LINE_WIDTH_POINTS,
        linestyle="-",
        solid_capstyle="butt",
        zorder=5,
    )[0]
    verticalLine = ax.plot(
        [CX0, CX0],
        [CY0 - yHalfSize, CY0 + yHalfSize],
        color=BEAM_CENTRE_COLOUR,
        linewidth=BEAM_CENTRE_LINE_WIDTH_POINTS,
        linestyle="-",
        solid_capstyle="butt",
        zorder=5,
    )[0]
    # Overlay artists must not change the detector limits or top-origin order.
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)
    return horizontalLine, verticalLine


def moveColourbarExponentToLabel(colourbar):
    """Move the colourbar's scientific exponent into its label."""
    colourbar.update_ticks()
    formatter = colourbar.ax.yaxis.get_major_formatter()
    if hasattr(formatter, "set_useMathText"):
        formatter.set_useMathText(True)
    if hasattr(formatter, "set_locs"):
        formatter.set_locs(colourbar.get_ticks())
    exponentText = formatter.get_offset()

    colourbar.set_label(
        (
            f"{COLOURBAR_NAME}, {exponentText}"
            if exponentText
            else COLOURBAR_NAME
        ),
        family=FONT_FAMILY,
        size=COLOURBAR_LABEL_FONT_SIZE,
    )
    colourbar.ax.yaxis.get_offset_text().set_visible(False)
    for tickLabel in colourbar.ax.get_yticklabels():
        tickLabel.set_fontfamily(FONT_FAMILY)
        tickLabel.set_fontsize(COLOURBAR_TICK_FONT_SIZE)


def renderOutputs(
    intensityFrames,
    xCoordinates,
    yCoordinates,
    frameDelays,
    frameDataIndexes,
    imageFormats,
    videoFormats,
    xAxisName,
    yAxisName,
    coordinateSpaceName,
    outputNameSuffix,
    verticalAxisStartsAtTop=False,
    showBeamCentre=False,
):
    """Render collected detector- or reciprocal-space frames with Matplotlib."""
    outputDirectory = folderOutput / OUTPUT_SUBFOLDER
    outputDirectory.mkdir(parents=True, exist_ok=True)
    baseName = f"{sampleName}_Scan_{scanNo:03d}_{outputNameSuffix}"

    colourMap = COLOURMAP_GENERATOR(
        COLOURMAP_COLOURS,
        name=COLOURMAP_NAME,
        samples=COLOURMAP_SAMPLES,
    )
    colourMap.set_bad(COLOURMAP_BAD_COLOUR)

    sharedNorm = (
        createColourNorm(intensityFrames)
        if COLOUR_LIMITS_ACROSS_ALL_FRAMES
        else None
    )
    initialNorm = sharedNorm or createColourNorm(intensityFrames[0])

    fig = plt.figure(figsize=figureSize())
    ax = fig.add_axes(AXES_POSITION)
    heatmap = ax.pcolormesh(
        xCoordinates,
        yCoordinates,
        intensityFrames[0],
        shading="nearest",
        cmap=colourMap,
        norm=initialNorm,
    )
    if verticalAxisStartsAtTop:
        verticalStep = (
            abs(yCoordinates[1] - yCoordinates[0])
            if len(yCoordinates) > 1
            else 1.0
        )
        ax.set_ylim(
            yCoordinates[-1] + verticalStep / 2,
            yCoordinates[0] - verticalStep / 2,
        )
    ax.set_aspect(AXES_ASPECT, adjustable="box", anchor="C")
    if showBeamCentre:
        addBeamCentre(fig, ax)
    ax.set_xlabel(
        xAxisName,
        fontfamily=FONT_FAMILY,
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(
        yAxisName,
        fontfamily=FONT_FAMILY,
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    ax.set_title(
        (
            GRAPH_TITLE.format(sampleName=sampleName, scanNo=scanNo)
            + f" ({coordinateSpaceName})"
        ),
        fontfamily=FONT_FAMILY,
        fontsize=GRAPH_TITLE_FONT_SIZE,
    )
    for tickLabel in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tickLabel.set_fontfamily(FONT_FAMILY)
        tickLabel.set_fontsize(AXIS_TICK_FONT_SIZE)
    colourbarAxes = addColourbarAxes(fig, ax)
    colourbar = fig.colorbar(
        heatmap,
        cax=colourbarAxes,
    )
    moveColourbarExponentToLabel(colourbar)
    delayPosition = delayInfoPosition(fig, ax)
    delayText = ax.text(
        *delayPosition,
        "",
        transform=fig.transFigure,
        horizontalalignment=DELAY_INFO_HORIZONTAL_ALIGNMENT,
        verticalalignment=DELAY_INFO_VERTICAL_ALIGNMENT,
        fontfamily=FONT_FAMILY,
        fontsize=DELAY_INFO_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )

    if LEGEND_ON:
        legendArguments = {"loc": LEGEND_POSITION}
        if LEGEND_BBOX_TO_ANCHOR is not None:
            legendArguments["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
        ax.legend(
            handles=[Patch(color=COLOURMAP_COLOURS[2], label=LEGEND_LABEL)],
            prop={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE},
            **legendArguments,
        )

    def updateFrame(frameIndex):
        frameIntensity = intensityFrames[frameIndex]
        heatmap.set_array(frameIntensity.ravel())
        if sharedNorm is None:
            heatmap.set_norm(createColourNorm(frameIntensity))
            colourbar.update_normal(heatmap)
            moveColourbarExponentToLabel(colourbar)
        delayText.set_text(
            DELAY_INFO.format(
                delay=frameDelays[frameIndex],
                dataIndex=frameDataIndexes[frameIndex],
            )
        )
        return heatmap, delayText

    if OUTPUT_IMAGES:
        frameDirectory = outputDirectory / f"{baseName}_frames"
        frameDirectory.mkdir(parents=True, exist_ok=True)
        frameNumberWidth = max(4, len(str(len(intensityFrames) - 1)))

        for frameIndex in range(len(intensityFrames)):
            updateFrame(frameIndex)
            for imageFormat in imageFormats:
                imagePath = frameDirectory / (
                    f"{baseName}_{frameIndex:0{frameNumberWidth}d}."
                    f"{imageFormat}"
                )
                fig.savefig(imagePath, dpi=IMAGE_DPI)
                print(f"Saved image frame to {imagePath}")

    if OUTPUT_VIDEOS:
        frameInterval = 1000.0 / float(FRAMES_PER_SECOND)
        animationObject = mplAnimation.FuncAnimation(
            fig,
            updateFrame,
            frames=len(intensityFrames),
            interval=frameInterval,
            blit=False,
            repeat=False,
        )

        for videoFormat in videoFormats:
            writerName = VIDEO_WRITERS[videoFormat]
            if not mplAnimation.writers.is_available(writerName):
                raise RuntimeError(
                    f"Matplotlib writer {writerName!r} is unavailable for "
                    f".{videoFormat} output"
                )
            videoPath = outputDirectory / f"{baseName}.{videoFormat}"
            animationObject.save(
                videoPath,
                writer=writerName,
                fps=FRAMES_PER_SECOND,
                dpi=IMAGE_DPI,
            )
            print(f"Saved video to {videoPath}")

    plt.close(fig)


def main():
    """Load all valid scan batches and render them in acquisition order."""
    imageFormats, videoFormats = validateOutputSettings()

    if alignMasks:
        raise ValueError("Animation output requires alignMasks=False")

    results = scanFiles(
        folderData=folderData,
        sampleName=sampleName,
        scanNames=scanNames,
        scanNo=scanNo,
        verbose=verbose,
        minimumFileSizeRatio=minimumFileSizeRatio,
    )

    allDataFiles = results["allDataFiles"]
    excludedIndexes = set(dataIndexesExcluded)
    invalidExcludedIndexes = sorted(
        index
        for index in excludedIndexes
        if not isinstance(index, (int, np.integer))
        or isinstance(index, (bool, np.bool_))
        or index < 0
        or index >= len(allDataFiles)
    )
    if invalidExcludedIndexes:
        raise ValueError(
            "dataIndexesExcluded contains invalid physical scan indexes: "
            f"{invalidExcludedIndexes}"
        )

    reciprocalFrames = []
    detectorFrames = []
    frameDelays = []
    frameDataIndexes = []
    qxReference = None
    qyReference = None
    detectorShapeReference = None

    for dataIndex, dataFilePath in enumerate(allDataFiles):
        if dataIndex in excludedIndexes:
            print(f"Skipping data [{dataIndex}]: explicitly excluded")
            continue
        if dataFilePath in results["brokenFiles"]:
            print(f"Skipping data [{dataIndex}]: broken scan")
            continue
        if dataFilePath in results["filesWithoutBackground"]:
            print(f"Skipping data [{dataIndex}]: no suitable background set")
            continue

        try:
            (
                qxCenters,
                qyCenters,
                intensityQxQy,
                delayScan,
                imageCCD,
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
        except (KeyError, OSError, ValueError) as error:
            print(f"Skipping data [{dataIndex}]: processing failed: {error}")
            continue

        if OUTPUT_RECIPROCAL_SPACE:
            if qxReference is None:
                qxReference = qxCenters
                qyReference = qyCenters
            elif not (
                qxCenters.shape == qxReference.shape
                and qyCenters.shape == qyReference.shape
                and np.allclose(qxCenters, qxReference)
                and np.allclose(qyCenters, qyReference)
            ):
                raise ValueError(
                    f"Data [{dataIndex}] produced incompatible Q-space axes"
                )

        if OUTPUT_DETECTOR_PIXEL_SPACE:
            if detectorShapeReference is None:
                detectorShapeReference = imageCCD.shape
            elif imageCCD.shape != detectorShapeReference:
                raise ValueError(
                    f"Data [{dataIndex}] produced incompatible detector "
                    f"shape {imageCCD.shape}; expected {detectorShapeReference}"
                )

        delayValue = np.asarray(delayScan).squeeze()
        if delayValue.size != 1:
            raise ValueError(
                f"Data [{dataIndex}] has non-scalar delay shape "
                f"{delayValue.shape}"
            )

        if OUTPUT_RECIPROCAL_SPACE:
            reciprocalFrames.append(
                np.asarray(intensityQxQy, dtype=np.float32)
            )
        if OUTPUT_DETECTOR_PIXEL_SPACE:
            detectorFrames.append(np.asarray(imageCCD, dtype=np.float32))
        frameDelays.append(float(delayValue))
        frameDataIndexes.append(dataIndex)

    if not frameDelays:
        raise RuntimeError("No valid scans were available for animation")

    print(
        f"Rendering {len(frameDelays)} valid frames from "
        f"{len(allDataFiles)} physical scan batches"
    )
    frameDelays = np.asarray(frameDelays)
    frameDataIndexes = np.asarray(frameDataIndexes)

    if OUTPUT_RECIPROCAL_SPACE:
        renderOutputs(
            intensityFrames=np.stack(reciprocalFrames),
            xCoordinates=qxReference,
            yCoordinates=qyReference,
            frameDelays=frameDelays,
            frameDataIndexes=frameDataIndexes,
            imageFormats=imageFormats,
            videoFormats=videoFormats,
            xAxisName=RECIPROCAL_X_AXIS_NAME,
            yAxisName=RECIPROCAL_Y_AXIS_NAME,
            coordinateSpaceName="reciprocal space",
            outputNameSuffix="Q_space",
        )

    if OUTPUT_DETECTOR_PIXEL_SPACE:
        detectorHeight, detectorWidth = detectorShapeReference
        renderOutputs(
            intensityFrames=np.stack(detectorFrames),
            xCoordinates=np.arange(detectorWidth),
            yCoordinates=np.arange(detectorHeight),
            frameDelays=frameDelays,
            frameDataIndexes=frameDataIndexes,
            imageFormats=imageFormats,
            videoFormats=videoFormats,
            xAxisName=DETECTOR_X_AXIS_NAME,
            yAxisName=DETECTOR_Y_AXIS_NAME,
            coordinateSpaceName="detector pixel space",
            outputNameSuffix="detector_pixels",
            verticalAxisStartsAtTop=True,
            showBeamCentre=BEAM_CENTRE_ON,
        )


if __name__ == "__main__":
    main()
