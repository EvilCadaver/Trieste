import h5py
from pathlib import Path
import re
import numpy as np

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

# Merge all four folders into acquisition order.
fileRecords.sort(key=lambda record: record[0])

dataFiles = [
    file
    for bunchId, scanName, file in fileRecords
    if scanName == ""
]

# Background groups store an anchor Path; this lookup converts it to the
# zero-based position in the normal-data stream.
dataIndexByFile = {
    file: index
    for index, file in enumerate(dataFiles)
}

pendingBackgrounds = {}
backgroundAnchorFile = None
latestDataFile = None
backgroundGroups = []

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
        if backgroundAnchorFile is not None:
            backgroundGroups.append({
                "anchorIndex": dataIndexByFile[backgroundAnchorFile],
                "anchorFile": backgroundAnchorFile,
                "backgrounds": {
                    name: pendingBackgrounds[name]
                    for name in backgroundNames
                },
                "dataFiles": [],
            })

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

currentGroup = None
nextGroupIndex = 0

# Move forward through normal data once. Whenever an anchor is reached, switch
# to its background set. This also makes trailing files use the most recent
# background, as required when acquisition ends before another set is taken.
for dataIndex, dataFile in enumerate(dataFiles):
    while (
        nextGroupIndex < len(backgroundGroups)
        and backgroundGroups[nextGroupIndex]["anchorIndex"]
        <= dataIndex
    ):
        currentGroup = backgroundGroups[nextGroupIndex]
        nextGroupIndex += 1

    if currentGroup is None:
        # This can happen only when normal files precede the first background
        # anchor. Retain them explicitly instead of silently dropping them.
        filesWithoutBackground.append(dataFile)
        continue

    currentGroup["dataFiles"].append(dataFile)
    backgroundsForFile[dataFile] = currentGroup["backgrounds"]

if verbose:
    # Print every normal scan and its assigned background set. This is the most
    # direct way to verify the 0..4, 5..9, ... grouping convention.
    for dataIndex, dataFile in enumerate(dataFiles):
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

