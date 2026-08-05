# Elettra Sincrotrone Trieste EIS-Timer HDF5 Scan Analysis

Python tools for discovering FERMI HDF5 acquisitions, rejecting incomplete
files, matching scans to their background acquisitions, correcting detector
images, projecting intensity into reciprocal space, and analysing radial
profiles in the time and frequency domains.

The processing flow is:

```text
fileScan.py
    -> qSpaceFunctions.py
        -> analysisSingleFrame.py
        -> analysisMultiFrame.py
            -> analysisFourierTransform.py
```

## Requirements

- Python 3
- NumPy
- h5py
- Matplotlib

Install the Python dependencies with:

```powershell
python -m pip install numpy h5py matplotlib
```

## Expected data layout

Set `folderData` near the beginning of `analysisSingleFrame.py` or
`analysisMultiFrame.py` to the directory containing the sample folders. At
EIS-Timer, this normally follows the UNC path pattern
`//online4eis.esce.elettra.trieste.it/store/eis-timer/%ProposalNo%`, where
`%ProposalNo%` is the eight-digit proposal number beginning with the four-digit
year (`YYYYNNNN`). For example:

```python
folderData = r"//online4eis.esce.elettra.trieste.it/store/eis-timer/20251234"
```

For sample `FeRh_A04` and scan number `68`, the expected directory structure is:

```text
folderData/
`-- FeRh_A04/
    |-- Scan_068/
    |   `-- rawdata/
    |       |-- Scan_068_326409867.h5
    |       `-- ...
    |-- Scan_068_NoProbe/
    |   `-- rawdata/
    |       `-- Scan_068_NoProbe_326428931.h5
    |-- Scan_068_OnlyProbe/
    |   `-- rawdata/
    |       `-- Scan_068_OnlyProbe_326430793.h5
    `-- Scan_068_Dark/
        `-- rawdata/
            `-- Scan_068_Dark_326432632.h5
```

The numeric suffix of each filename is treated as the acquisition's first
bunch ID and determines chronological order. Scan numbers may have any number
of leading zeroes, so `scanNo=68` matches both `Scan_068` and `Scan_00068`.

The acquisition naming convention is configured as:

```python
scanNames = ["Scan", "", "NoProbe", "OnlyProbe", "Dark"]
```

- `scanNames[0]` is the common folder and filename prefix.
- `scanNames[1]` is the normal-acquisition suffix. An empty string means that
  no additional suffix is used.
- `scanNames[2:]` are the required background acquisition types.

## File discovery and background grouping

`fileScan.py` provides the reusable `scanFiles()` function:

```python
from fileScan import scanFiles

results = scanFiles(
    folderData=r"//online4eis.esce.elettra.trieste.it/store/eis-timer/20251234",
    sampleName="FeRh_A04",
    scanNames=["Scan", "", "NoProbe", "OnlyProbe", "Dark"],
    scanNo=68,
    verbose=False,
    minimumFileSizeRatio=0.95,
)
```

Set `verbose=True` to print discovered files, rejected acquisitions, physical
scan indices, assigned backgrounds, and a summary of the background groups.

### Grouping convention

Normal scan indices are zero-based. Normal and background files are merged into
chronological order using the numeric bunch ID suffix in each filename;
filesystem timestamps are not used. Every contiguous run of background files
starts a new group, so grouping does not assume a fixed number of normal data
files between background acquisitions.

Within each detected background run, the first usable `NoProbe`, `OnlyProbe`,
and `Dark` file forms the background set. An undersized attempt is rejected,
but a later valid retry in the same run can complete the set. Incomplete runs
are retained for diagnostics.

- Data before the first complete background set use the closest following set.
- An incomplete middle or final window continues using the closest preceding
  complete set.
- If no complete set exists, the affected scans are reported in
  `filesWithoutBackground`.

### Broken-file detection

The median size of all matching normal and background files is used as the
reference. A file is classified as broken when:

```text
file size < median file size * minimumFileSizeRatio
```

With the default ratio of `0.95`, files more than 5% smaller than the median are
rejected. Broken normal files remain in `allDataFiles` so physical indices do
not shift, but they are excluded from the usable data.

### Returned results

The main entries returned by `scanFiles()` are:

| Key | Contents |
| --- | --- |
| `backgroundGroups` | Complete background groups and assigned clean data files |
| `usableDataFiles` | Clean data files with a complete background assignment |
| `backgroundsForFile` | Mapping from each usable data path to its backgrounds |
| `allDataFiles` | Every normal scan in physical order, including broken files |
| `dataFiles` | Normal scans that pass the file-size check |
| `brokenFiles` | All normal or background files rejected by size |
| `brokenDataFiles` | Broken normal scans only |
| `invalidBackgroundGroups` | Incomplete background timing windows |
| `filesWithoutBackground` | Clean scans without a complete background set |
| `filesByScanName` | Chronological files separated by acquisition type |
| `fileSizes` | Mapping from every discovered path to its size in bytes |
| `referenceFileSize` | Median file size used as the reference |
| `minimumFileSize` | Calculated rejection threshold in bytes |

Use `allDataFiles[dataIndex]` when `dataIndex` must refer to the physical
acquisition order. Using an index into `usableDataFiles` can select the wrong
acquisition because rejected files are absent from that list.

## Reciprocal-space functions

`qSpaceFunctions.py` contains the shared numerical operations used by both
frame-analysis scripts:

- `createQSpaceMap()` reads a scan and its assigned backgrounds, normalizes and
  subtracts the backgrounds, applies detector masks, calculates scattering
  vectors, and bins the corrected intensity onto a regular Qx/Qy grid.
- `createRadialIntensityProfile()` integrates Q-space intensity into radial
  shells within the configured angular sector and its symmetry-equivalent
  directions.
- `subtractPolynomialBackground()` applies radial cutoffs, fits a polynomial
  background, and returns the background-subtracted profile.

Q-space coordinates and radial distances are expressed in nm^-1. Empty bins
remain `NaN` instead of being represented as zero intensity.

## Single-frame analysis

`analysisSingleFrame.py` analyses one physical acquisition selected by
`dataIndex`. Configure the scan location, detector geometry, background ROI,
beam-stop masks, Q-space resolution, and radial-profile sector near the top of
the script, then run:

```powershell
python analysisSingleFrame.py
```

The script:

1. Discovers the scan and its background groups with `scanFiles()`.
2. Corrects the selected detector image with its assigned backgrounds.
3. Projects the detector data onto a Qx/Qy intensity map.
4. Optionally overlays the angular sectors used for radial integration.
5. Plots the summed radial intensity profile versus `|Q|`.

This workflow is useful for checking detector geometry, masks, angular sectors,
and radial binning before processing the complete scan.

## Multi-frame analysis

`analysisMultiFrame.py` applies the same correction and Q-space projection to
the usable acquisitions in a scan. Configure its input, detector, mask,
profile, cutoff, background-fit, and output settings, then run:

```powershell
python analysisMultiFrame.py
```

The script builds a background-subtracted radial profile for every retained
delay, handles excluded, duplicate, broken, and missing acquisitions, and
creates a Q-versus-delay heatmap. It saves both a PNG and a CSV in `folderOutput`.

`handlingDuplicateDelays` controls duplicate delay metadata. `"keep first"`
keeps the first batch at each recorded delay and represents inferred gaps as
NaN. `"propagate delay"` assigns otherwise-unused duplicate batches to inferred
missing delays in chronological order. If there are fewer duplicate batches
than missing delays, it warns and falls back to `"keep first"`.

The CSV contains a metadata header followed by:

```text
***DATA***
Q,dt,Intensity
nm^-1,ps,a.u.
```

This CSV is the input expected by `analysisFourierTransform.py`.

## Fourier analysis

`analysisFourierTransform.py` transforms every Q profile along the delay axis.
Set `folderData` and `analysisName` to a Q-versus-delay CSV produced by
`analysisMultiFrame.py`, then run:

```powershell
python analysisFourierTransform.py
```

Before the FFT, the script linearly interpolates missing delay samples for each
Q, subtracts the trace mean, and applies a Hann window. It computes a real,
one-sided FFT, omits the zero-frequency bin, and reports frequency in GHz.

Two files are saved beside the input CSV:

- `<input stem>_Fourier.png` contains the frequency-versus-Q heatmap.
- `<input stem>_Fourier.csv` contains the relevant sample and processing
  metadata followed by `Q,f,Intensity` data in `nm^-1,GHz,a.u.`.

## Supported repository files

- `fileScan.py` — file discovery, validation, and background grouping.
- `qSpaceFunctions.py` — detector correction, Q-space projection, radial
  integration, and polynomial background subtraction.
- `analysisSingleFrame.py` — interactive inspection of one acquisition.
- `analysisMultiFrame.py` — complete delay-scan processing and Q-versus-delay
  export.
- `analysisFourierTransform.py` — time-axis Fourier analysis and
  frequency-versus-Q export.

## AI-assisted development

This project was developed with assistance from OpenAI Codex for code
generation, refactoring, documentation, and debugging. All assisted changes
were reviewed and tested by the project maintainer, who remains responsible for
the final implementation.

## Licence

This project is licensed under **EUPL-1.2-or-later**. See `LICENSE` for the
project licensing statement and `EUPL` for the full official EUPL 1.2 text.

When distributing or communicating the work or a derivative, retain the
required copyright and licence notices, identify modifications, and provide the
source as required by the EUPL.
