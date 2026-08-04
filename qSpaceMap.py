"""Create a binned Qx/Qy intensity map from results returned by scanFiles()."""

import h5py
import numpy as np


def createQSpaceMap(
    results,
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
    BETA,
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
    # These settings are part of the caller's complete analysis configuration,
    # although delay is not needed to calculate the requested three outputs.
    del h5DelayPath, delayZero

    if not isinstance(dataIndex, (int, np.integer)):
        raise TypeError("dataIndex must be an integer")
    if dataIndex < 0 or dataIndex >= len(results["allDataFiles"]):
        raise IndexError(f"Scan index {dataIndex} does not exist")
    if N_BINN <= 0:
        raise ValueError("N_BINN must be positive")
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

    # Read and orient all detector images in the same way as analysis.py.
    imageScan = _readDetectorImage(dataFilePath, h5CCDImagePath)
    imageDark = _readDetectorImage(backgrounds["Dark"], h5CCDImagePath)
    imageNoProbe = _readDetectorImage(
        backgrounds["NoProbe"],
        h5CCDImagePath,
    )
    imageOnlyProbe = _readDetectorImage(
        backgrounds["OnlyProbe"],
        h5CCDImagePath,
    )

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

    image = (
        imageDifferential / normScan
        - imageDifferentialNoProbe / normOnlyProbe
    )

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
        return qxCenters, qyCenters, intensity

    # Remove the residual offset using the same background ROI, then exclude
    # beam-stop pixels from all subsequent calculations.
    levelBG = np.median(image[roiBG])
    image = (image - levelBG) * maskBeamStop

    # Calculate the outgoing ray and scattering vector for every detector pixel.
    height, width = image.shape
    v, u = np.indices((height, width))

    R0 = np.array(
        [
            DCCD / np.cos(OMEGA) * np.cos(ALPHA),
            0.0,
            -DCCD / np.cos(OMEGA) * np.sin(ALPHA),
        ]
    )
    detectorRows = np.array([0.0, 1.0, 0.0])
    detectorColumns = np.array(
        [-np.sin(BETA), 0.0, -np.cos(BETA)]
    )

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

    return qxCenters, qyCenters, intensity


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
