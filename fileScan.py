import h5py
from pathlib import Path
import re
import numpy as np

folderData = Path(r"C:/git/Trieste/Data")

sampleName = "FeRh_A04"

scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]

scanNo = 61

verbose = True

folderSample = folderData / sampleName

name0 = re.escape(scanNames[0])
number = str(int(scanNo))

# Example matches:
# scanA_test5
# scanA_test05
# scanA_test00005
folderRegex = re.compile(
    rf"{name0}_.*?(?<!\d)0*{number}"
)

backgroundNames = scanNames[2:]

fileRecords = []
filesByScanName = {
    scanName: [] for scanName in scanNames[1:]
}

# Find files for only the selected scanNo.
for scanName in scanNames[1:]:
    scanSuffix = f"_{re.escape(scanName)}" if scanName else ""

    folderRegex = re.compile(
        rf"{name0}_0*{number}{scanSuffix}"
    )
    filenameRegex = re.compile(
        rf"{name0}_0*{number}{scanSuffix}_"
        rf"(?P<bunchId>\d+)\.h5"
    )

    for scanFolder in folderSample.iterdir():
        if not scanFolder.is_dir():
            continue

        if not folderRegex.fullmatch(scanFolder.name):
            continue

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

# Merge all four folders into acquisition order.
fileRecords.sort(key=lambda record: record[0])

dataFiles = [
    file
    for bunchId, scanName, file in fileRecords
    if scanName == ""
]

dataIndexByFile = {
    file: index
    for index, file in enumerate(dataFiles)
}

pendingBackgrounds = {}
backgroundAnchorFile = None
latestDataFile = None
backgroundGroups = []

# Find the normal scan immediately preceding each background set.
for bunchId, scanName, file in fileRecords:
    if scanName == "":
        latestDataFile = file
        continue

    if scanName not in backgroundNames:
        continue

    # The first background file marks the start of a new set.
    if not pendingBackgrounds:
        backgroundAnchorFile = latestDataFile

    pendingBackgrounds[scanName] = file

    # The background order does not matter.
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

        pendingBackgrounds.clear()
        backgroundAnchorFile = None

backgroundGroups.sort(
    key=lambda group: group["anchorIndex"]
)

# A background applies from its anchor scan up to, but not
# including, the next background's anchor scan.
backgroundsForFile = {}
filesWithoutBackground = []

currentGroup = None
nextGroupIndex = 0

for dataIndex, dataFile in enumerate(dataFiles):
    while (
        nextGroupIndex < len(backgroundGroups)
        and backgroundGroups[nextGroupIndex]["anchorIndex"]
        <= dataIndex
    ):
        currentGroup = backgroundGroups[nextGroupIndex]
        nextGroupIndex += 1

    if currentGroup is None:
        filesWithoutBackground.append(dataFile)
        continue

    currentGroup["dataFiles"].append(dataFile)
    backgroundsForFile[dataFile] = currentGroup["backgrounds"]

if verbose:
    # Print every normal scan and its assigned background set.
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

