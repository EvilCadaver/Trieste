import h5py
from pathlib import Path
import re
import numpy as np
from statistics import median

# This script discovers one numbered acquisition and associates each normal
# data file with the background set acquired after the first file in its
# block. Acquisition order is determined from the numeric bunch ID at the
# end of every filename, rather than from directory or filesystem order.

# Root containing one directory per sample.
folderData = Path(r"C:/git/Trieste/Data")

# Acquisition naming convention:
#   scanNames[0]  -> common folder/file prefix
#   scanNames[1]  -> normal data (an empty suffix)
#   scanNames[2:] -> the background acquisition types
sampleName = "FeRh_A04"

scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]

# Leading zeroes in the on-disk scan number are accepted, so scanNo=61
# matches names containing both "61" and "061".
scanNo = 61

# Enable the detailed data-to-background mapping printed at the end.
verbose = True

# A file is considered broken when it is more than 5% smaller than the median
# file size for this acquisition. The median represents the majority well and
# is not strongly affected by a small number of incomplete files.
minimumFileSizeRatio = 0.95

folderSample = folderData / sampleName

name0 = re.escape(scanNames[0])
number = str(int(scanNo))

# General scan-folder expression retained as a description of the naming
# convention. A more specific expression is built for each scan type below.
folderRegex = re.compile(
    rf"{name0}_.*?(?<!\d)0*{number}"
)

# A usable background set is complete only after all these types are found.
backgroundNames = scanNames[2:]

# Each record has the form (firstBunchId, scanType, path). Keeping all scan
# types in one list allows them to be merged into their true acquisition order.
fileRecords = []

# This additional index keeps the files separated by type for later analysis,
# e.g. filesByScanName["Dark"] or filesByScanName[""].
filesByScanName = {
    scanName: [] for scanName in scanNames[1:]
}

# Find files for only the selected scanNo.
for scanName in scanNames[1:]:
    # Normal data has no suffix ("Scan_061"); background types do
    # (for example, "Scan_061_NoProbe").
    scanSuffix = f"_{re.escape(scanName)}" if scanName else ""

    # fullmatch() makes sure scanNo=61 does not accidentally select scan 610.
    folderRegex = re.compile(
        rf"{name0}_0*{number}{scanSuffix}"
    )

    # The final numeric field is the first bunch ID stored in the HDF5 file.
    # It is captured so files from all four directories can be sorted together.
    filenameRegex = re.compile(
        rf"{name0}_0*{number}{scanSuffix}_"
        rf"(?P<bunchId>\d+)\.h5"
    )

    for scanFolder in folderSample.iterdir():
        if not scanFolder.is_dir():
            continue

        if not folderRegex.fullmatch(scanFolder.name):
            continue

        # Only HDF5 files directly inside the acquisition's rawdata directory
        # belong to this stream.
        folderRawdata = scanFolder / "rawdata"

        if not folderRawdata.is_dir():
            continue

        for file in folderRawdata.glob("*.h5"):
            match = filenameRegex.fullmatch(file.name)

            if match is None:
                continue

            # Integer comparison is required: lexicographic filename sorting
            # would not reliably represent bunch order for different lengths.
            bunchId = int(match.group("bunchId"))

            fileRecords.append((bunchId, scanName, file))
            filesByScanName[scanName].append(file)

# Determine the typical size using all normal and background files belonging
# to this scan number. Their expected sizes differ by less than 1%, while an
# acquisition interrupted by a FEL failure should be a much smaller outlier.
fileSizes = {
    file: file.stat().st_size
    for bunchId, scanName, file in fileRecords
}

referenceFileSize = (
    median(fileSizes.values())
    if fileSizes
    else None
)

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

# Merge all four folders into acquisition order.
fileRecords.sort(key=lambda record: record[0])

# Keep broken normal files in this list so their acquisition positions still
# count. Removing them here would shift the 5*k background anchor indices.
allDataFiles = [
    file
    for bunchId, scanName, file in fileRecords
    if scanName == ""
]

# This is the clean normal-data list intended for subsequent analysis.
dataFiles = [
    file
    for file in allDataFiles
    if file not in brokenFiles
]

# Background groups store an anchor Path; this lookup converts it to the
# zero-based position in the normal-data stream.
dataIndexByFile = {
    file: index
    for index, file in enumerate(allDataFiles)
}

pendingBackgrounds = {}
backgroundAnchorFile = None
latestDataFile = None
backgroundGroups = []
invalidBackgroundGroups = []

# Walk the complete time-ordered stream to find the normal scan immediately
# preceding each background sequence. That scan is the 5*k anchor: the newly
# completed background applies to it and to subsequent normal scans until the
# next anchor is reached.
for bunchId, scanName, file in fileRecords:
    if scanName == "":
        latestDataFile = file
        continue

    if scanName not in backgroundNames:
        continue

    # The first background file marks the start of a new set. Save the latest
    # normal file now because the three background types may arrive in any order.
    if not pendingBackgrounds:
        backgroundAnchorFile = latestDataFile

    pendingBackgrounds[scanName] = file

    # Do not publish a partial set. It becomes usable only when every required
    # background type has appeared.
    if all(name in pendingBackgrounds for name in backgroundNames):
        backgroundSet = {
            name: pendingBackgrounds[name]
            for name in backgroundNames
        }

        if backgroundAnchorFile is not None:
            completedGroup = {
                "anchorIndex": dataIndexByFile[backgroundAnchorFile],
                "anchorFile": backgroundAnchorFile,
                "backgrounds": backgroundSet,
                "dataFiles": [],
            }

            # One incomplete background file makes the entire subtraction set
            # unreliable. Keep it for diagnostics, but do not activate it.
            if any(file in brokenFiles for file in backgroundSet.values()):
                invalidBackgroundGroups.append(completedGroup)
            else:
                backgroundGroups.append(completedGroup)

        # Reset the accumulator so the next background file starts a new set.
        pendingBackgrounds.clear()
        backgroundAnchorFile = None

# Usually this is already chronological, but sorting by anchor makes the
# assignment below independent of background acquisition order.
backgroundGroups.sort(
    key=lambda group: group["anchorIndex"]
)

# A background applies from its anchor scan up to, but not
# including, the next background's anchor scan.
backgroundsForFile = {}
filesWithoutBackground = []
brokenDataFiles = []

currentGroup = None
nextGroupIndex = 0

# Move forward through every normal acquisition, including broken ones, so the
# original indices remain stable. Only clean files are added to analysis groups.
for dataIndex, dataFile in enumerate(allDataFiles):
    while (
        nextGroupIndex < len(backgroundGroups)
        and backgroundGroups[nextGroupIndex]["anchorIndex"]
        <= dataIndex
    ):
        currentGroup = backgroundGroups[nextGroupIndex]
        nextGroupIndex += 1

    if dataFile in brokenFiles:
        brokenDataFiles.append(dataFile)
        continue

    if currentGroup is None:
        # This can happen only when normal files precede the first background
        # anchor. Retain them explicitly instead of silently dropping them.
        filesWithoutBackground.append(dataFile)
        continue

    currentGroup["dataFiles"].append(dataFile)
    backgroundsForFile[dataFile] = currentGroup["backgrounds"]

if verbose:
    print(f"Reference file size: {referenceFileSize:.0f} bytes")
    print(f"Minimum accepted size: {minimumFileSize:.0f} bytes")
    print(f"Broken files: {len(brokenFiles)}")

    for brokenFile in sorted(brokenFiles):
        print(
            f"  {brokenFile}: {fileSizes[brokenFile]} bytes"
        )

    # Print every normal scan and its assigned background set. This is the most
    # direct way to verify the 0..4, 5..9, ... grouping convention.
    for dataFile in dataFiles:
        dataIndex = dataIndexByFile[dataFile]
        print(f"\nData [{dataIndex}]: {dataFile}")

        backgroundSet = backgroundsForFile.get(dataFile)

        if backgroundSet is None:
            print("  No background set assigned")
            continue

        for backgroundName in backgroundNames:
            print(
                f"  {backgroundName}: "
                f"{backgroundSet[backgroundName]}"
            )

    # Optional summary of each background group.
    print("\nBackground groups:")

    for groupIndex, group in enumerate(backgroundGroups):
        firstDataIndex = group["anchorIndex"]
        groupDataFiles = group["dataFiles"]

        print(
            f"\nGroup {groupIndex}: "
            f"starts at data index {firstDataIndex}"
        )
        print(f"  Anchor: {group['anchorFile']}")
        print(f"  Number of data files: {len(groupDataFiles)}")

        for backgroundName in backgroundNames:
            print(
                f"  {backgroundName}: "
                f"{group['backgrounds'][backgroundName]}"
            )

    if invalidBackgroundGroups:
        print("\nRejected background groups:")

        for group in invalidBackgroundGroups:
            print(f"  Anchor: {group['anchorFile']}")

            for backgroundName in backgroundNames:
                backgroundFile = group["backgrounds"][backgroundName]
                status = (
                    "BROKEN"
                    if backgroundFile in brokenFiles
                    else "valid"
                )
                print(
                    f"    {backgroundName} ({status}): "
                    f"{backgroundFile}"
                )

