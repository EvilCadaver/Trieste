import numpy as np

## Scan specific settings
# Data folder location
folderData = r"C:/git/Trieste/Data"
# Sample name / subfolder in folderData
sampleName = "FeRh_A06"
# Scan specific names: 0) Scan name; 1) Data aqusition name; 2) Background 1 acqusition name; 3) Background 2 acquistion name; 4) Background 3 acqusition name
scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]
# Scan number chosen for analysis
scanNo = 28
# File size ratio for scan batches rejection
minimumFileSizeRatio = 0.95
# Refresh live figures after this many processed batches. Decimal values are
# rounded to the nearest integer; zero and negative values are treated as 1.
updateFiguresInterval = 10

## Single file analysis for now
## HDF5 package paths
h5CCDImagePath = "/CCD/Image"
h5DelayPath = "/photon_source/SeedLaser/Delay_line_2"
# Delay zero setting
delayZero = -3096.49
# Single scan index to analyse
dataIndex = 63
# Data indexes to exclude
dataIndexesExcluded = [62]

## Scan measurement parameters
N_BINN = 2                      #Binning of the detector
PIXEL_SIZE = N_BINN * 13.5e-6   #Pixel edge length, m
LAMBDA = 23.5e-9                #Wavelength, m
CY0 = 492                       #Reflection centre Y, pixels
CX0 = 48                        #Reflection centre X, pixels
DCCD = 67e-3                    #Shortest distance from the incident point on the sample to the detector, m 
ALPHA = 17 /180*np.pi           #Incidence beam angle, rad
OMEGA = 16.5 /180*np.pi         #Angle between scattered beam maximum and the detector normal (positive towards the sample surface), rad

## Masks allignemnt
# Make true for masks allignment
alignMasks = False
# Limit the region to show during masks allignment
roiAllignMasks = np.s_[345:610, 0:220]

# ROI for the background zero substraction
roiBG = np.s_[420:450, 60:90] 

## Masking regions, will be added to the empty mask, add rectangles as np.s_[y0:y1,x0:x1] to the list [ ] structure.
maskBS = [np.s_[342:612,0:222], np.s_[511:1024,24:118]]

## Q-space plot settings
# Maximum number of reciprocal-space bins along the longer Qx/Qy dimension.
Q_SPACE_BINS_MAX = 512

## Q-space analysis
# Show angles chosen for integration
plotProfileAngles = True
# Choose the highlight colour
plotProfileAnglesColour = r"green"
# Angle from Qx in deg
ZETA = 0
# Acceptance angle in deg
D_ZETA = 20
# Symmetry
ZETA_SYMMETRY = 4
# Radial step bin
RADIAL_STEP_BIN = 1
# Q-distance cutoffs in percent
Q_LOW_CUTOFF = 10
Q_HIGH_CUTOFF = 90
# Polynomial background fitted only after applying both Q cutoffs
BACKGROUND_NPOLY = 3
