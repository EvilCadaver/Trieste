from pathlib import Path
import re
from statistics import median


def scanFiles(
    folderData,
    sampleName,
    scanNames,
    scanNo,
    verbose=False,
    minimumFileSizeRatio=0.95,
):
    """Find one scan and associate its usable data with background triplets.

    Parameters
    ----------
    folderData : str or Path
        Root directory containing the sample directories.
    sampleName : str
        Name of the sample directory.
    scanNames : sequence of str
        Naming components in the form
        [commonPrefix, "", backgroundType1, backgroundType2, ...].
    scanNo : int
        Scan number. Any amount of leading zero padding is accepted on disk.
    verbose : bool, optional
        Print detailed file assignments and diagnostics.
    minimumFileSizeRatio : float, optional
        Files smaller than this fraction of the median size are rejected.

    Returns
    -------
    dict
        File lists, valid/invalid groups, per-file background assignments,
        broken-file diagnostics, and the reference file size.
    """
    folderData = Path(folderData)

    if len(scanNames) < 3 or scanNames[1] != "":
        raise ValueError(
            "scanNames must be [commonPrefix, '', backgroundType1, ...]"
        )
    if not 0 < minimumFileSizeRatio <= 1:
        raise ValueError("minimumFileSizeRatio must be greater than 0 and at most 1")

    folderSample = folderData / sampleName
    if not folderSample.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {folderSample}")

    commonName = re.escape(scanNames[0])

    # Converting through int removes any padding supplied by the caller. The
    # regular expressions add 0* in front of this value, so scanNo=68 matches
    # both Scan_68 and Scan_00068, but does not match Scan_680.
    number = str(int(scanNo))
    backgroundNames = list(scanNames[2:])

    # ------------------------------------------------------------------
    # Stage 1: locate every file belonging to the requested scan number.
    # ------------------------------------------------------------------
    # Each record is (firstBunchId, scanType, path). The bunch ID at the end
    # of each filename provides the true acquisition order across directories.
    fileRecords = []
    filesByScanName = {scanName: [] for scanName in scanNames[1:]}

    for scanName in scanNames[1:]:
        # The empty scan name represents normal data. It deliberately produces
        # no suffix and no extra underscore:
        #   normal:     Scan_068
        #   background: Scan_068_NoProbe
        scanSuffix = f"_{re.escape(scanName)}" if scanName else ""

        # fullmatch() is used below, so only the complete expected folder or
        # filename is accepted. This prevents scanNo=68 from selecting 680.
        folderRegex = re.compile(rf"{commonName}_0*{number}{scanSuffix}")
        filenameRegex = re.compile(
            rf"{commonName}_0*{number}{scanSuffix}_"
            rf"(?P<bunchId>\d+)\.h5"
        )

        # Search only immediate acquisition folders inside the sample folder.
        # Other sample directories and unrelated nested folders are ignored.
        for scanFolder in folderSample.iterdir():
            if not scanFolder.is_dir() or not folderRegex.fullmatch(scanFolder.name):
                continue

            # Acquisition HDF5 files are expected directly in rawdata. Using
            # glob rather than rglob prevents accidental inclusion of backups
            # or processed files stored in deeper subdirectories.
            folderRawdata = scanFolder / "rawdata"
            if not folderRawdata.is_dir():
                continue

            for file in folderRawdata.glob("*.h5"):
                match = filenameRegex.fullmatch(file.name)
                if match is None:
                    continue

                bunchId = int(match.group("bunchId"))
                fileRecords.append((bunchId, scanName, file))
                filesByScanName[scanName].append(file)

    # Paths from different directories cannot be ordered meaningfully by their
    # names alone. Merge them by the numeric bunch ID captured from each name.
    fileRecords.sort(key=lambda record: record[0])

    # Keep the per-type convenience lists chronological as well. rsplit takes
    # only the final underscore field, which is the bunch ID.
    for files in filesByScanName.values():
        files.sort(key=lambda file: int(file.stem.rsplit("_", 1)[-1]))

    # ------------------------------------------------------------------
    # Stage 2: detect files from interrupted acquisitions by their size.
    # ------------------------------------------------------------------
    # The median is insensitive to a small number of interrupted acquisitions.
    fileSizes = {
        file: file.stat().st_size
        for _, _, file in fileRecords
    }
    referenceFileSize = median(fileSizes.values()) if fileSizes else None
    minimumFileSize = (
        referenceFileSize * minimumFileSizeRatio
        if referenceFileSize is not None
        else None
    )
    brokenFiles = {
        file
        for file, fileSize in fileSizes.items()
        if fileSize < minimumFileSize
    }

    # Broken normal files remain in allDataFiles because their physical indices
    # must still count when locating the 5*k background timing windows.
    allDataFiles = [
        file
        for _, scanName, file in fileRecords
        if scanName == ""
    ]
    dataFiles = [file for file in allDataFiles if file not in brokenFiles]
    dataIndexByFile = {
        file: index
        for index, file in enumerate(allDataFiles)
    }
    bunchIdByFile = {
        file: bunchId
        for bunchId, _, file in fileRecords
    }
    backgroundRecords = [
        (bunchId, scanName, file)
        for bunchId, scanName, file in fileRecords
        if scanName in backgroundNames
    ]

    backgroundGroups = []
    invalidBackgroundGroups = []

    # ------------------------------------------------------------------
    # Stage 3: separate background attempts into acquisition-time windows.
    # ------------------------------------------------------------------
    # A background attempt occurs after normal scan 5*k and before the next
    # normal file. A later valid retry can replace a broken attempt in a window.
    for anchorIndex in range(0, len(allDataFiles), 5):
        # Normal indices 0, 5, 10, ... are the scans after which a background
        # triplet is expected. The anchor remains in its own five-scan block.
        anchorFile = allDataFiles[anchorIndex]
        windowStartBunchId = bunchIdByFile[anchorFile]
        nextDataIndex = anchorIndex + 1

        # Only files acquired between the anchor and the immediately following
        # normal scan belong to this attempt. For the final normal scan there
        # is no upper boundary, so infinity keeps any later backgrounds.
        windowEndBunchId = (
            bunchIdByFile[allDataFiles[nextDataIndex]]
            if nextDataIndex < len(allDataFiles)
            else float("inf")
        )
        windowBackgrounds = [
            (bunchId, scanName, file)
            for bunchId, scanName, file in backgroundRecords
            if windowStartBunchId < bunchId < windowEndBunchId
        ]

        backgroundSet = {}
        extraBackgrounds = []
        rejectedBackgrounds = []

        # Records are already chronological. Select the first usable file of
        # each required type, retain undersized attempts as rejected, and mark
        # any additional usable acquisitions of the same type as extras.
        for _, scanName, file in windowBackgrounds:
            if file in brokenFiles:
                rejectedBackgrounds.append(file)
            elif scanName in backgroundSet:
                extraBackgrounds.append(file)
            else:
                backgroundSet[scanName] = file

        group = {
            "anchorIndex": anchorIndex,
            "anchorFile": anchorFile,
            "backgrounds": backgroundSet,
            "extraBackgrounds": extraBackgrounds,
            "rejectedBackgrounds": rejectedBackgrounds,
            "dataFiles": [],
        }

        # A group is usable only when every requested background type exists.
        # Incomplete attempts remain available for diagnostics, but are never
        # assigned to normal data.
        target = (
            backgroundGroups
            if all(name in backgroundSet for name in backgroundNames)
            else invalidBackgroundGroups
        )
        target.append(group)

    backgroundGroups.sort(key=lambda group: group["anchorIndex"])

    # ------------------------------------------------------------------
    # Stage 4: assign clean normal files to the closest applicable valid group.
    # ------------------------------------------------------------------
    backgroundsForFile = {}
    filesWithoutBackground = []
    brokenDataFiles = []

    # Leading data fall forward to the earliest complete background group.
    # Data after an incomplete final window retain the closest preceding group.
    currentGroup = backgroundGroups[0] if backgroundGroups else None
    nextGroupIndex = 1 if backgroundGroups else 0

    # Walk the normal stream once in physical index order. currentGroup changes
    # only when the next complete group's anchor is reached. Consequently, an
    # incomplete middle window is bridged by the preceding complete group.
    for dataIndex, dataFile in enumerate(allDataFiles):
        while (
            nextGroupIndex < len(backgroundGroups)
            and backgroundGroups[nextGroupIndex]["anchorIndex"] <= dataIndex
        ):
            currentGroup = backgroundGroups[nextGroupIndex]
            nextGroupIndex += 1

        if dataFile in brokenFiles:
            # Broken data retain their original positions but are not usable.
            brokenDataFiles.append(dataFile)
        elif currentGroup is None:
            # This occurs only if no complete background group was found.
            filesWithoutBackground.append(dataFile)
        else:
            # Store both directions: the group contains its normal files, and
            # each normal file can directly retrieve its background dictionary.
            currentGroup["dataFiles"].append(dataFile)
            backgroundsForFile[dataFile] = currentGroup["backgrounds"]

    # This is the final analysis-ready list: files must pass the size check and
    # also have a complete background set assigned.
    usableDataFiles = [
        file
        for file in dataFiles
        if file in backgroundsForFile
    ]

    results = {
        "backgroundGroups": backgroundGroups,
        "invalidBackgroundGroups": invalidBackgroundGroups,
        "backgroundsForFile": backgroundsForFile,
        "usableDataFiles": usableDataFiles,
        "dataFiles": dataFiles,
        "allDataFiles": allDataFiles,
        "filesByScanName": filesByScanName,
        "filesWithoutBackground": filesWithoutBackground,
        "brokenFiles": brokenFiles,
        "brokenDataFiles": brokenDataFiles,
        "fileSizes": fileSizes,
        "referenceFileSize": referenceFileSize,
        "minimumFileSize": minimumFileSize,
        "backgroundNames": backgroundNames,
        "dataIndexByFile": dataIndexByFile,
    }

    if verbose:
        _printScanResults(results)

    return results


def _printScanResults(results):
    """Print the detailed diagnostics produced by scanFiles()."""
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

    if referenceFileSize is None:
        print("No matching HDF5 files found.")
        return

    print(f"Reference file size: {referenceFileSize:.0f} bytes")
    print(f"Minimum accepted size: {minimumFileSize:.0f} bytes")
    print(f"Broken files: {len(brokenFiles)}")

    for brokenFile in sorted(brokenFiles):
        print(f"  {brokenFile}: {fileSizes[brokenFile]} bytes")

    for dataFile in allDataFiles:
        dataIndex = dataIndexByFile[dataFile]
        print(f"\nData [{dataIndex}]: {dataFile}")

        if dataFile in brokenFiles:
            sizeRatio = fileSizes[dataFile] / referenceFileSize
            print(
                f"  BROKEN: {fileSizes[dataFile]} bytes "
                f"({sizeRatio:.1%} of reference size)"
            )
            continue

        backgroundSet = backgroundsForFile.get(dataFile)
        if backgroundSet is None:
            print("  No background set assigned")
            continue

        for backgroundName in backgroundNames:
            print(f"  {backgroundName}: {backgroundSet[backgroundName]}")

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
    print("\nBackground groups:")

    for groupIndex, group in enumerate(backgroundGroups):
        print(
            f"\nGroup {groupIndex}: "
            f"starts at data index {group['anchorIndex']}"
        )
        print(f"  Anchor: {group['anchorFile']}")
        print(f"  Number of data files: {len(group['dataFiles'])}")

        for backgroundName in backgroundNames:
            print(f"  {backgroundName}: {group['backgrounds'][backgroundName]}")
        for rejectedFile in group["rejectedBackgrounds"]:
            print(f"  Rejected attempt: {rejectedFile}")

    if invalidGroups:
        print("\nRejected background groups:")

        for group in invalidGroups:
            print(f"  Anchor: {group['anchorFile']}")

            for backgroundName in backgroundNames:
                backgroundFile = group["backgrounds"].get(backgroundName)
                if backgroundFile is None:
                    print(f"    {backgroundName}: MISSING")
                else:
                    print(f"    {backgroundName} (valid): {backgroundFile}")

            for extraFile in group["extraBackgrounds"]:
                print(f"    Extra: {extraFile}")
            for rejectedFile in group["rejectedBackgrounds"]:
                print(f"    Rejected attempt: {rejectedFile}")


if __name__ == "__main__":
    results = scanFiles(
        folderData=Path(r"C:/git/Trieste/Data"),
        sampleName="FeRh_A04",
        scanNames=["Scan", "", "NoProbe", "OnlyProbe", "Dark"],
        scanNo=68,
        verbose=False,
        minimumFileSizeRatio=0.95,
    )

    # The two main outputs for downstream analysis:
    backgroundGroups = results["backgroundGroups"]
    usableDataFiles = results["usableDataFiles"]
