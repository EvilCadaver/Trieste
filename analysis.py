import h5py
from fileScan import scanFiles
import numpy as np
import matplotlib.pyplot as plt

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
dataIndex = 40

## Scan measurement parameters
NBinn = 2                       #Binning of the detector
SizePixel = NBinn * 13.5e-6     #Pixel edge length, m
Lambda = 23.5e-9                #Wavelength, m
CY1 = 1024/NBinn - 1            #Reflection centre Y, pixels
CX1 = 150/NBinn - 1             #Reflection centre X, pixels
RCCD = 67e-3                    #Distance to detector's CY1, CX1, m
OMEGA = 33 /180*np.pi         #Incidence beam angle, rad
CHI = 45 /180*np.pi             #Detector angle, rad

## Make true for masks allignment
allignMasks = False
roiAllignMasks = np.s_[400:600, 0:220]

# ROI for the background zero substraction
roiBG = np.s_[400:450, 160:210] 

## Masking regions, will be added to the empty mask, add rectangles as np.s_[y0:y1,x0:x1] to the list [ ] structure.
maskBS = [np.s_[379:647,0:222], np.s_[511:1024,29:120]]

## Calling the folder scanning function
results = scanFiles(
    folderData=folderData,
    sampleName=sampleName,
    scanNames=scanNames,
    scanNo=scanNo,
    verbose=verbose,
    minimumFileSizeRatio=minimumFileSizeRatio,
)

## Unpacking results dictionary
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

## Files scanning short report
if referenceFileSize is None:
    print("No matching HDF5 files found.")
    exit()

print(f"Reference file size: {referenceFileSize:.0f} bytes")
print(f"Minimum accepted size: {minimumFileSize:.0f} bytes")
print(f"Borked files: {len(brokenFiles)}")

for brokenFile in sorted(brokenFiles):
    print(f"  {brokenFile}: {fileSizes[brokenFile]} bytes")

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

## Check for validity
print(f"Chosen scan index: {dataIndex}")
try:
    dataFilePath = results["allDataFiles"][dataIndex]
    
except IndexError:
    print(f"Scan index {dataIndex} does not exist")
    exit()

else:
    if dataFilePath in results["brokenFiles"]:
        print("Scan is broken")
        exit()

    elif dataFilePath in results["filesWithoutBackground"]:
        print("No suitable background set")
        exit()

    else:
        backgrounds = results["backgroundsForFile"].get(dataFilePath)

        if backgrounds is None:
            print("No background assignment found")
        else:
            print("Scan:", dataFilePath)
            print("NoProbe:", backgrounds["NoProbe"])
            print("OnlyProbe:", backgrounds["OnlyProbe"])
            print("Dark:", backgrounds["Dark"])

## Reading data files for 'dataIndex'

with h5py.File(dataFilePath, "r") as h5:
    imageScan = h5[h5CCDImagePath][...]
    imageScan = np.rot90(imageScan.T, 2)
    delayScan = round(- h5[h5DelayPath][...] + delayZero,1)

with h5py.File(backgrounds["Dark"], "r") as h5:
    imageDark = h5[h5CCDImagePath][...]
    imageDark = np.rot90(imageDark.T, 2)

with h5py.File(backgrounds["NoProbe"], "r") as h5:
    imageNoProbe = h5[h5CCDImagePath][...]
    imageNoProbe = np.rot90(imageNoProbe.T, 2)

with h5py.File(backgrounds["OnlyProbe"], "r") as h5:
    imageOnlyProbe = h5[h5CCDImagePath][...]
    imageOnlyProbe = np.rot90(imageOnlyProbe.T, 2)

## BeamStop masking
maskBeamStop = np.ones_like(imageScan, dtype=bool)
for mask in maskBS:
    maskBeamStop[mask] = False
maskBGroi = ~np.ones_like(imageScan, dtype=bool)
maskBGroi[roiBG] = True

normNoProbe = np.sum(imageScan[roiBG])/np.sum(imageNoProbe[roiBG])
imageDifferential = (imageScan - normNoProbe * imageNoProbe)

normDark = np.sum(imageOnlyProbe[roiBG])/np.sum(imageDark[roiBG])
imageDifferentialNoProbe = (imageOnlyProbe - normDark * imageDark)

normScan = np.sum(imageDifferential * maskBeamStop)
normOnlyProbe = np.sum(imageDifferentialNoProbe * maskBeamStop)

image = (imageDifferential/normScan - imageDifferentialNoProbe/normOnlyProbe)

if allignMasks:
    image = image * (maskBeamStop ^ maskBGroi)
    image = image[roiAllignMasks]
    valid = image[maskBGroi[roiAllignMasks] & np.isfinite(image)]
    vmin = valid.min()
    vmax = valid.max()
    
else:
    levelBG = np.median(image[roiBG])
    image = (image.astype(np.float64) - levelBG) * maskBeamStop
    valid = image[maskBeamStop & np.isfinite(image)]
    vmin = 0.1 * valid.min()
    vmax = 0.1 * valid.max()




## Plotting detector image
plt.figure()

plt.imshow(
    image,
    origin="upper",
    cmap="RdBu",
    vmin=vmin,
    vmax=vmax,
)

plt.xlabel("Detector column")
plt.ylabel("Detector row")
plt.title(f"Scan {dataIndex}, delay = {delayScan} ps")
plt.colorbar(label="Intensity")
plt.show(block=True)

print(f"Delay: {delayScan}")