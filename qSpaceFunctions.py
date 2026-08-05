"""Create a binned Qx/Qy intensity map from results returned by scanFiles()."""

import h5py
import numpy as np


def createQSpaceMap(
    results,
    h5CCDImagePath,
    h5DelayPath,
    delayZero,
    dataIndex,
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
):
    """Return Q-space bin centres and mean differential intensity.

    Parameters correspond to the single-file and detector settings in
    ``analysis.py``. ``results`` must be the dictionary returned by
    ``scanFiles()``.

    ``h5DelayPath`` and ``delayZero`` are accepted so the function can receive
    the complete single-scan configuration as requested. They do not affect the
    returned intensity calculation.

    When ``alignMasks`` is false, the coordinates are Qx and Qy bin centres in
    nm^-1. When it is true, the original detector-space mask-alignment behaviour
    is preserved: the coordinates are detector column and row numbers and the
    intensity is the image cropped by ``roiAllignMasks``.

    Returns
    -------
    qxCenters : numpy.ndarray
        Qx bin centres in nm^-1, or detector column numbers in alignment mode.
    qyCenters : numpy.ndarray
        Qy bin centres in nm^-1, or detector row numbers in alignment mode.
    intensity : numpy.ndarray
        Mean differential intensity with shape
        ``(len(qyCenters), len(qxCenters))``. Empty bins contain NaN.
    """
    
    print(f"Chosen scan index: {dataIndex}")
    if not isinstance(dataIndex, (int, np.integer)):
        raise TypeError("dataIndex must be an integer")
    if dataIndex < 0 or dataIndex >= len(results["allDataFiles"]):
        raise IndexError(f"Scan index {dataIndex} does not exist")
    if PIXEL_SIZE <= 0 or LAMBDA <= 0 or DCCD <= 0:
        raise ValueError("PIXEL_SIZE, LAMBDA, and DCCD must be positive")
    if not isinstance(Q_SPACE_BINS_MAX, (int, np.integer)):
        raise TypeError("Q_SPACE_BINS_MAX must be an integer")
    if Q_SPACE_BINS_MAX < 1:
        raise ValueError("Q_SPACE_BINS_MAX must be a positive integer")

    dataFilePath = results["allDataFiles"][dataIndex]

    if dataFilePath in results["brokenFiles"]:
        raise ValueError(f"Scan index {dataIndex} is broken: {dataFilePath}")
    if dataFilePath in results["filesWithoutBackground"]:
        raise ValueError(
            f"Scan index {dataIndex} has no suitable background set: "
            f"{dataFilePath}"
        )

    backgrounds = results["backgroundsForFile"].get(dataFilePath)
    if backgrounds is None:
        raise ValueError(
            f"Scan index {dataIndex} has no background assignment: "
            f"{dataFilePath}"
        )

    print("Scan:", dataFilePath)
    print("NoProbe:", backgrounds["NoProbe"])
    print("OnlyProbe:", backgrounds["OnlyProbe"])
    print("Dark:", backgrounds["Dark"])

    # Read and orient all detector images in the same way as analysis.py.
    imageScan = _readDetectorImage(dataFilePath, h5CCDImagePath)
    with h5py.File(dataFilePath, "r") as h5:
        delayScan = round(- h5[h5DelayPath][...] + delayZero,1)
    imageDark = _readDetectorImage(backgrounds["Dark"], h5CCDImagePath)
    imageNoProbe = _readDetectorImage(backgrounds["NoProbe"], h5CCDImagePath)
    imageOnlyProbe = _readDetectorImage(backgrounds["OnlyProbe"], h5CCDImagePath)


    imageShapes = {
        imageScan.shape,
        imageDark.shape,
        imageNoProbe.shape,
        imageOnlyProbe.shape,
    }
    if len(imageShapes) != 1:
        raise ValueError("The scan and background detector images differ in shape")

    # True pixels participate in normalization and Q-space rebinning. Each
    # slice in maskBS removes one beam-stop region from the detector image.
    maskBeamStop = np.ones_like(imageScan, dtype=bool)
    for mask in maskBS:
        maskBeamStop[mask] = False

    # Normalize and subtract the matching backgrounds using the detector-space
    # background ROI, preserving the current analysis.py formulation.
    noProbeDenominator = np.sum(imageNoProbe[roiBG])
    darkDenominator = np.sum(imageDark[roiBG])
    if noProbeDenominator == 0 or darkDenominator == 0:
        raise ValueError("A background normalization ROI has zero total intensity")

    normNoProbe = np.sum(imageScan[roiBG]) / noProbeDenominator
    imageDifferential = imageScan - normNoProbe * imageNoProbe

    normDark = np.sum(imageOnlyProbe[roiBG]) / darkDenominator
    imageDifferentialNoProbe = imageOnlyProbe - normDark * imageDark

    normScan = np.sum(imageDifferential * maskBeamStop)
    normOnlyProbe = np.sum(imageDifferentialNoProbe * maskBeamStop)
    if normScan == 0 or normOnlyProbe == 0:
        raise ValueError("A differential detector image has zero normalization")

    image = (imageDifferential / normScan - imageDifferentialNoProbe / normOnlyProbe)

    if alignMasks:
        # Preserve the detector-space diagnostic from analysis.py. The XOR
        # exposes roiBG even where it overlaps the beam-stop mask, which makes
        # the relative positions of those masks visible during alignment.
        maskBGroi = np.zeros_like(imageScan, dtype=bool)
        maskBGroi[roiBG] = True
        intensity = (image * (maskBeamStop ^ maskBGroi))[roiAllignMasks]

        if intensity.ndim != 2 or intensity.size == 0:
            raise ValueError(
                "roiAllignMasks must select a non-empty rectangular 2D region"
            )

        # Crop coordinate grids with the same ROI instead of assuming that its
        # slices start at zero or have a step of one.
        detectorRows, detectorColumns = np.indices(image.shape)
        selectedColumns = detectorColumns[roiAllignMasks]
        selectedRows = detectorRows[roiAllignMasks]

        if (
            selectedColumns.shape != intensity.shape
            or selectedRows.shape != intensity.shape
            or not np.all(selectedColumns == selectedColumns[0, :])
            or not np.all(selectedRows == selectedRows[:, 0, None])
        ):
            raise ValueError(
                "roiAllignMasks must describe a rectangular row/column slice"
            )

        qxCenters = selectedColumns[0, :]
        qyCenters = selectedRows[:, 0]
        return qxCenters, qyCenters, intensity, delayScan

    # Remove the residual offset using the same background ROI, then exclude
    # beam-stop pixels from all subsequent calculations.
    levelBG = np.median(image[roiBG])
    imageCCD = (image - levelBG)
    image = imageCCD * maskBeamStop

    # Calculate the outgoing ray and scattering vector for every detector pixel.
    height, width = image.shape
    v, u = np.indices((height, width))

    R0 = np.array([ DCCD / np.cos(OMEGA) * np.cos(ALPHA),
                    0.0,
                    -DCCD / np.cos(OMEGA) * np.sin(ALPHA)])
    BETA = OMEGA + ALPHA
    detectorRows = np.array([0.0, 1.0, 0.0])
    detectorColumns = np.array([-np.sin(BETA), 0.0, -np.cos(BETA)])

    du = (u - CX0) * PIXEL_SIZE
    dv = (v - CY0) * PIXEL_SIZE
    detectorPosition = (
        R0
        + du[..., None] * detectorColumns
        + dv[..., None] * detectorRows
    )

    scatteredDirection = detectorPosition / np.linalg.norm(
        detectorPosition,
        axis=-1,
        keepdims=True,
    )
    incidentDirection = np.array(
        [np.cos(ALPHA), 0.0, np.sin(ALPHA)]
    )
    q = 2 * np.pi / LAMBDA * (scatteredDirection - incidentDirection)

    # Flatten only usable, finite detector samples before histogramming them.
    valid = (
        maskBeamStop
        & np.isfinite(image)
        & np.all(np.isfinite(q[..., :2]), axis=-1)
    )
    qxValues = q[..., 0][valid] * 1e-9  # m^-1 to nm^-1
    qyValues = q[..., 1][valid] * 1e-9
    intensityValues = image[valid]

    if intensityValues.size == 0:
        raise ValueError("No finite, unmasked detector pixels are available")

    qxMin, qxMax = qxValues.min(), qxValues.max()
    qyMin, qyMax = qyValues.min(), qyValues.max()
    qxSpan = qxMax - qxMin
    qySpan = qyMax - qyMin

    # Use square reciprocal-space bins. The longer dimension has at most the
    # requested number of bins; the shorter dimension scales proportionally.
    dq = max(qxSpan, qySpan) / Q_SPACE_BINS_MAX
    if not np.isfinite(dq) or dq <= 0:
        raise ValueError("The Qx/Qy projection has no finite extent")

    qxBinCount = min(
        Q_SPACE_BINS_MAX,
        max(1, int(np.ceil(qxSpan / dq))),
    )
    qyBinCount = min(
        Q_SPACE_BINS_MAX,
        max(1, int(np.ceil(qySpan / dq))),
    )
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
    intensityQxQy = np.divide(
        intensitySum,
        pixelCount,
        out=np.full_like(intensitySum, np.nan),
        where=pixelCount > 0,
    )

    if not np.any(np.isfinite(intensityQxQy)):
        raise ValueError("No detector intensities were assigned to the Q-space grid")

    qxCenters = 0.5 * (qxEdges[:-1] + qxEdges[1:])
    qyCenters = 0.5 * (qyEdges[:-1] + qyEdges[1:])

    # histogram2d returns axes in (Qx, Qy) order. Transpose once here so the
    # returned grid follows plotting convention: rows are Qy, columns are Qx.
    intensity = intensityQxQy.T

    return qxCenters, qyCenters, intensity, delayScan, imageCCD


def createRadialIntensityProfile(
    qxCenters,
    qyCenters,
    intensity,
    ZETA,
    D_ZETA,
    ZETA_SYMMETRY,
    RADIAL_STEP_BIN,
):
    """Sum intensity versus distance inside symmetry-related angular sectors.

    Angles are specified in degrees. ``ZETTA`` is the reference direction
    measured counter-clockwise from +Qx. ``ZETTA_SYMMETRY`` creates equally
    spaced directions over 360 degrees. ``D_ZETTA`` is the total acceptance
    angle around every direction, so each direction accepts ±D_ZETTA/2.
    ``RADIAL_STEP_BIN`` is a positive integer multiplier of the native Q-space
    grid spacing. For example, 2 combines distances into shells twice as wide
    as one Q-space bin.

    Radial bins use the Q-space grid spacing. Bins containing no accepted,
    finite intensity values are returned as NaN rather than as a false zero.
    """
    qxCenters = np.asarray(qxCenters, dtype=float)
    qyCenters = np.asarray(qyCenters, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if qxCenters.ndim != 1 or qyCenters.ndim != 1:
        raise ValueError("qxCenters and qyCenters must be one-dimensional")
    expectedShape = (qyCenters.size, qxCenters.size)
    if intensity.shape != expectedShape:
        raise ValueError(
            f"intensity must have shape {expectedShape}, "
            f"received {intensity.shape}"
        )
    if not isinstance(ZETA_SYMMETRY, (int, np.integer)):
        raise TypeError("ZETA_SYMMETRY must be an integer")
    if ZETA_SYMMETRY < 1:
        raise ValueError("ZETA_SYMMETRY must be at least 1")
    if (
        isinstance(RADIAL_STEP_BIN, (bool, np.bool_))
        or not isinstance(RADIAL_STEP_BIN, (int, np.integer))
    ):
        raise TypeError("RADIAL_STEP_BIN must be an integer")
    if RADIAL_STEP_BIN < 1:
        raise ValueError("RADIAL_STEP_BIN must be at least 1")
    if not np.isfinite(ZETA) or not np.isfinite(D_ZETA):
        raise ValueError("ZETA and D_ZETTA must be finite")
    if D_ZETA <= 0 or D_ZETA > 360:
        raise ValueError("D_ZETA must be greater than 0 and at most 360 degrees")

    qxGrid, qyGrid = np.meshgrid(qxCenters, qyCenters)
    distanceGrid = np.hypot(qxGrid, qyGrid)
    angleGrid = np.degrees(np.arctan2(qyGrid, qxGrid))

    # Folding by the symmetry period finds the signed angular difference from
    # the nearest equivalent direction: ZETA + k*360/ZETA_SYMMETRY.
    symmetryPeriod = 360.0 / ZETA_SYMMETRY
    angleDifference = (
        (angleGrid - ZETA + symmetryPeriod / 2) % symmetryPeriod
        - symmetryPeriod / 2
    )
    accepted = (
        np.abs(angleDifference) <= D_ZETA / 2
    ) & np.isfinite(intensity)

    if not np.any(accepted):
        raise ValueError("No finite Q-space bins fall inside the angular acceptance")

    # The Q map uses square bins, so its axis spacing is also a natural radial
    # step. Obtain it from either axis to avoid introducing another constant.
    axisSteps = np.concatenate(
        [np.abs(np.diff(qxCenters)), np.abs(np.diff(qyCenters))]
    )
    axisSteps = axisSteps[np.isfinite(axisSteps) & (axisSteps > 0)]
    if axisSteps.size == 0:
        raise ValueError("Cannot determine radial spacing from the Q-space axes")
    radialStep = RADIAL_STEP_BIN * np.median(axisSteps)

    acceptedDistances = distanceGrid[accepted]
    acceptedIntensity = intensity[accepted]
    radialMaximum = acceptedDistances.max()
    radialEdges = np.arange(
        0.0,
        radialMaximum + radialStep,
        radialStep,
    )
    if radialEdges.size < 2:
        radialEdges = np.array([0.0, radialStep])
    elif radialEdges[-1] <= radialMaximum:
        radialEdges = np.append(radialEdges, radialEdges[-1] + radialStep)

    sumIntensity, _ = np.histogram(
        acceptedDistances,
        bins=radialEdges,
        weights=acceptedIntensity,
    )
    acceptedCount, _ = np.histogram(
        acceptedDistances,
        bins=radialEdges,
    )
    sumIntensity = sumIntensity.astype(float)
    sumIntensity[acceptedCount == 0] = np.nan
    distance = 0.5 * (radialEdges[:-1] + radialEdges[1:])

    return distance, sumIntensity


def subtractPolynomialBackground(
    distance,
    intensity,
    Q_LOW_CUTOFF,
    Q_HIGH_CUTOFF,
    BACKGROUND_NPOLY,
):
    """Cut a radial profile, then fit and subtract its polynomial background.

    Cutoffs are first applied as percentages of the complete radial-distance
    span. The polynomial background is then calculated using only finite
    intensity points that remain inside that cutoff interval. Returned
    background and corrected arrays retain the original profile shape, with
    NaN outside the selected interval. This makes the fitted range explicit and
    prevents polynomial extrapolation from entering the final analysis.
    """
    distance = np.asarray(distance, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if distance.ndim != 1 or intensity.ndim != 1:
        raise ValueError("distance and intensity must be one-dimensional")
    if distance.shape != intensity.shape:
        raise ValueError("distance and intensity must have the same shape")
    if distance.size < 2:
        raise ValueError("At least two radial profile points are required")
    if not np.all(np.isfinite(distance)):
        raise ValueError("distance must contain only finite values")
    if not np.all(np.diff(distance) > 0):
        raise ValueError("distance must be strictly increasing")

    for name, cutoff in (
        ("Q_LOW_CUTOFF", Q_LOW_CUTOFF),
        ("Q_HIGH_CUTOFF", Q_HIGH_CUTOFF),
    ):
        if (
            isinstance(cutoff, (bool, np.bool_))
            or not isinstance(
                cutoff,
                (int, float, np.integer, np.floating),
            )
        ):
            raise TypeError(f"{name} must be a finite number")
        if not np.isfinite(cutoff):
            raise ValueError(f"{name} must be finite")

    if not 0 <= Q_LOW_CUTOFF < Q_HIGH_CUTOFF <= 100:
        raise ValueError(
            "Cutoffs must satisfy "
            "0 <= Q_LOW_CUTOFF < Q_HIGH_CUTOFF <= 100"
        )
    if (
        isinstance(BACKGROUND_NPOLY, (bool, np.bool_))
        or not isinstance(BACKGROUND_NPOLY, (int, np.integer))
    ):
        raise TypeError("BACKGROUND_NPOLY must be an integer")
    if BACKGROUND_NPOLY < 0:
        raise ValueError("BACKGROUND_NPOLY must be non-negative")

    distanceSpan = distance[-1] - distance[0]
    qLow = distance[0] + Q_LOW_CUTOFF / 100 * distanceSpan
    qHigh = distance[0] + Q_HIGH_CUTOFF / 100 * distanceSpan
    cutoffMask = (distance >= qLow) & (distance <= qHigh)

    # The cutoff is deliberately part of fitMask: no profile point below qLow
    # or above qHigh contributes to the polynomial background calculation.
    fitMask = cutoffMask & np.isfinite(intensity)

    requiredPointCount = BACKGROUND_NPOLY + 1
    if np.count_nonzero(fitMask) < requiredPointCount:
        raise ValueError(
            f"Polynomial order {BACKGROUND_NPOLY} requires at least "
            f"{requiredPointCount} finite points inside the cutoff interval"
        )

    # Polynomial.fit scales the Q domain internally, which is better
    # conditioned than fitting raw powers of small Q values directly.
    backgroundModel = np.polynomial.Polynomial.fit(
        distance[fitMask],
        intensity[fitMask],
        deg=BACKGROUND_NPOLY,
    )
    background = np.full_like(intensity, np.nan)
    background[cutoffMask] = backgroundModel(distance[cutoffMask])

    correctedIntensity = np.full_like(intensity, np.nan)
    correctedIntensity[cutoffMask] = (
        intensity[cutoffMask] - background[cutoffMask]
    )

    return cutoffMask, background, correctedIntensity, qLow, qHigh


def _readDetectorImage(filePath, h5CCDImagePath):
    """Read a two-dimensional detector image and apply its fixed orientation."""
    with h5py.File(filePath, "r") as h5:
        image = h5[h5CCDImagePath][...]

    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional detector image in {filePath}, "
            f"received shape {image.shape}"
        )

    return np.rot90(image.T, 2)
