# Elettra Sincrotrone Trieste EIS-Timer HDF5 Scan Analysis

Tools for discovering FERMI HDF5 acquisitions, rejecting incomplete files,
matching normal scans to their background acquisitions, correcting detector
images, and projecting their intensities into reciprocal space.

The main workflow is implemented in `analysis.py`. Reusable file discovery and
background grouping are provided by `fileScan.py`.

## Requirements

The file-discovery and grouping function uses only the Python standard library.
The image-analysis scripts additionally require:

- Python 3
- NumPy
- h5py
- Matplotlib

Install the Python dependencies with:

```powershell
python -m pip install numpy h5py matplotlib
```

Some earlier analysis and detector-projection routines are provided as MATLAB
files and require a suitable MATLAB installation.

## Expected data layout

For a sample called `FeRh_A04` and scan number `68`, the scanner expects:

```text
Data/
└── FeRh_A04/
    ├── Scan_068/
    │   └── rawdata/
    │       ├── Scan_068_326409867.h5
    │       └── ...
    ├── Scan_068_NoProbe/
    │   └── rawdata/
    │       └── Scan_068_NoProbe_326428931.h5
    ├── Scan_068_OnlyProbe/
    │   └── rawdata/
    │       └── Scan_068_OnlyProbe_326430793.h5
    └── Scan_068_Dark/
        └── rawdata/
            └── Scan_068_Dark_326432632.h5
```

The numeric value at the end of each filename is treated as the acquisition's
first bunch ID and determines chronological order. The scan number may contain
any number of leading zeroes on disk, so `scanNo=68` also matches `Scan_068` and
`Scan_00068`.

The `scanNames` argument describes the naming convention:

```python
scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]
```

- `scanNames[0]` is the common folder and filename prefix.
- `scanNames[1]` is the normal acquisition suffix. It is empty, so no extra
  underscore is added.
- `scanNames[2:]` are the required background acquisition types.

## Scanning and grouping files

Import `scanFiles` into another script located in the same directory:

```python
from fileScan import scanFiles

results = scanFiles(
    folderData=r"C:/git/Trieste/Data",
    sampleName="FeRh_A04",
    scanNames=["Scan", "", "NoProbe", "OnlyProbe", "Dark"],
    scanNo=68,
    verbose=False,
    minimumFileSizeRatio=0.95,
)
```

Set `verbose=True` to print file sizes, rejected acquisitions, every physical
normal-scan index, its assigned backgrounds, and a summary of all background
groups.

### Grouping convention

Normal scan indices are zero-based. A background triplet is expected immediately
after normal scan indices `0, 5, 10, ...`. Background files are associated with
an anchor only when their bunch IDs lie between that anchor and the next normal
scan.

For each timing window, the first usable `NoProbe`, `OnlyProbe`, and `Dark` file
forms the background set. An undersized attempt is rejected, but a later valid
retry inside the same timing window can repair the set. A window missing any
required background type is retained for diagnostics and is not used directly.

Data assignment handles incomplete background windows as follows:

- Data before the first complete background set use the closest following set.
- An incomplete middle or final window continues using the closest preceding
  complete set.
- If no complete background set exists, the clean scans are reported in
  `filesWithoutBackground`.

### Broken-file detection

The median size of all matching normal and background files is used as the
reference size. A file is classified as broken when:

```text
file size < median file size × minimumFileSizeRatio
```

With the default ratio of `0.95`, files more than 5% smaller than the median are
rejected. Broken normal files remain in `allDataFiles` so their physical indices
still count, but they are excluded from `usableDataFiles` and from each group's
`dataFiles` list.

## Returned results

`scanFiles()` returns a dictionary. Its main entries are:

| Key | Contents |
| --- | --- |
| `backgroundGroups` | Complete background groups and their assigned clean data files |
| `usableDataFiles` | Clean data files with a complete background assignment |
| `backgroundsForFile` | Mapping from each usable data path to its background dictionary |
| `allDataFiles` | Every normal scan in physical order, including broken files |
| `dataFiles` | Normal scans that pass the file-size check |
| `brokenFiles` | All normal or background files rejected by size |
| `brokenDataFiles` | Broken normal scans only |
| `invalidBackgroundGroups` | Incomplete background timing windows |
| `filesWithoutBackground` | Clean scans for which no complete set exists |
| `filesByScanName` | Chronological files separated by acquisition type |
| `fileSizes` | Mapping from every discovered path to its size in bytes |
| `referenceFileSize` | Median file size used as the reference |
| `minimumFileSize` | Calculated rejection threshold in bytes |

Each item in `backgroundGroups` has this structure:

```python
{
    "anchorIndex": 0,
    "anchorFile": Path(".../Scan_068_....h5"),
    "backgrounds": {
        "NoProbe": Path(".../Scan_068_NoProbe_....h5"),
        "OnlyProbe": Path(".../Scan_068_OnlyProbe_....h5"),
        "Dark": Path(".../Scan_068_Dark_....h5"),
    },
    "dataFiles": [Path("..."), ...],
    "extraBackgrounds": [],
    "rejectedBackgrounds": [],
}
```

## Selecting one scan and its backgrounds

Use `allDataFiles` when the index must refer to the original physical acquisition
order. Do not use `usableDataFiles[index]` for this purpose, because rejected
files are absent from that list and later indices can shift.

```python
scanIndex = 3

try:
    scanFilePath = results["allDataFiles"][scanIndex]
except IndexError:
    print(f"Scan index {scanIndex} does not exist")
else:
    if scanFilePath in results["brokenFiles"]:
        print("Scan is broken")
    elif scanFilePath in results["filesWithoutBackground"]:
        print("No suitable background set")
    else:
        backgrounds = results["backgroundsForFile"].get(scanFilePath)

        if backgrounds is None:
            print("No background assignment found")
        else:
            print("Scan:", scanFilePath)
            print("NoProbe:", backgrounds["NoProbe"])
            print("OnlyProbe:", backgrounds["OnlyProbe"])
            print("Dark:", backgrounds["Dark"])
```

The returned values are `pathlib.Path` objects and can be passed directly to
`h5py.File`:

```python
import h5py

with h5py.File(scanFilePath, "r") as h5:
    imageScan = h5["/CCD/Image"][...]
```

## Detector and reciprocal-space analysis

`analysis.py` performs the complete single-acquisition analysis. Configure the
data location, sample and scan names, `scanNo`, and `dataIndex` near the top of
the file. The script then:

1. Calls `scanFiles()` and validates the selected physical acquisition index.
2. Loads the scan and its assigned `NoProbe`, `OnlyProbe`, and `Dark`
   acquisitions.
3. Normalizes the backgrounds using `roiBG`, forms the differential image, and
   applies the regions in `maskBS`.
4. Displays the corrected detector image.
5. Calculates each detector pixel's scattering vector in the sample coordinate
   system:

   ```python
   Q = 2 * np.pi / LAMBDA * (S_f - S_i)
   ```

6. Optionally displays the detector surface in three-dimensional reciprocal
   space when `Q3D_PLOT = True`. `Q3D_STEP` controls the pixel subsampling used
   to keep that plot responsive.
7. Rebins the valid intensities onto a regular Qx/Qy grid and plots the mean
   intensity in each occupied bin. `Q_SPACE_BINS_MAX` sets the maximum number
   of square reciprocal-space bins along the projection's longer dimension;
   empty bins remain undefined rather than being displayed as zero intensity.

Reciprocal-space axes are displayed in nm^-1. Under the coordinate convention
used by the script, increasing detector column primarily maps toward decreasing
Qx, while increasing detector row maps toward increasing Qy. Since the detector
image is displayed with its origin at the top, the Qx/Qy projection naturally
appears horizontally and vertically flipped relative to that image.

Matplotlib displays are sequential. Close the detector-image window to continue
to the optional 3D view and the binned Qx/Qy plot.

## Repository files

- `analysis.py` — detector correction and reciprocal-space analysis workflow.
- `fileScan.py` — reusable discovery, validation, grouping, and reporting logic.

## AI-assisted development

This project was developed with assistance from OpenAI Codex. The tool was
used for code generation, refactoring, documentation, and debugging. All
AI-assisted changes were reviewed and tested by the project maintainer, who
remains responsible for the final implementation.

## Licence

This project is licensed under **EUPL-1.2-or-later**. See `LICENSE` for the
project licensing statement and `EUPL` for the full official EUPL 1.2 text.

When distributing or communicating the work or a derivative, retain the required
copyright and licence notices, identify modifications, and provide the source as
required by the EUPL.
