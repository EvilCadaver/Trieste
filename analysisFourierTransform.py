"""Fourier-analyse Q-versus-delay data exported by analysisMultiFrame.py.

The input CSV contains a metadata header followed by a ``***DATA***`` marker
and the columns Q, dt, and Intensity. For every Q value, this script fills
missing time samples, removes the mean, applies a Hann window, and computes a
real FFT along the delay axis. It writes the positive-frequency amplitudes to
``<input stem>_Fourier.csv`` and saves the corresponding heatmap as PNG.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Input selection. Output files are created in the same directory as this CSV.
folderData = r"./Output_DS/FeRh_A04"
analysisName = r"FeRh_A04_Scan_068_Q_vs_delay_a45deg_d20deg_sym4.csv"
fileData = Path(folderData) / analysisName

if not fileData.exists():
    raise FileNotFoundError(f"File '{fileData}' doesn't exist!")

with open(fileData, newline="", encoding="utf-8") as inputCSV:
    # Match the delimiter used by the source file (normally comma or semicolon).
    dialect = csv.Sniffer().sniff(inputCSV.read(8096))
    print(f"Delimiters detected: {repr(dialect.delimiter)}")
    inputCSV.seek(0)
    reader = csv.reader(inputCSV, dialect)
    input_metadata = {}

    # Retain the header as key/value pairs until the data-section marker. Only
    # the FFT-relevant subset is copied to the output later in the script.
    for row in reader:
        if row and row[0].strip() == "***DATA***":
            break
        if len(row) >= 2:
            input_metadata[row[0].strip()] = row[1].strip()
    else:
        raise ValueError("The CSV file does not contain a ***DATA*** marker")

    # The two rows immediately after the marker contain names and units. Every
    # remaining non-empty row must contain the three numeric data values.
    column_names = next(reader)
    column_units = next(reader)
    data = np.array(
        [[float(value) for value in row] for row in reader if row],
        dtype=float,
    )

if data.ndim != 2 or data.shape[1] != 3:
    raise ValueError(
        f"Expected three data columns after ***DATA***, found shape {data.shape}"
    )

q_values = np.unique(data[:, 0])
delay_values = np.unique(data[:, 1])
intensity = np.full((len(q_values), len(delay_values)), np.nan)

# Convert the row-oriented table into a rectangular intensity[Q, time] grid.
# Starting with NaN preserves absent Q/time combinations for interpolation.
q_indexes = np.searchsorted(q_values, data[:, 0])
delay_indexes = np.searchsorted(delay_values, data[:, 1])
intensity[q_indexes, delay_indexes] = data[:, 2]

finite_intensity = intensity[np.isfinite(intensity)]
if finite_intensity.size == 0:
    raise ValueError("The intensity column contains no finite values")

# The FFT cannot operate on NaNs. Fill internal acquisition gaps independently
# for each Q using linear interpolation between the nearest valid time samples.
# np.interp also uses the nearest finite endpoint if an edge sample is missing.
intensity_interpolated = np.empty_like(intensity)
for q_index, trace in enumerate(intensity):
    finite = np.isfinite(trace)
    if finite.sum() < 2:
        raise ValueError(
            f"Q = {q_values[q_index]:g} has fewer than two finite intensity values"
        )
    intensity_interpolated[q_index] = np.interp(
        delay_values,
        delay_values[finite],
        trace[finite],
    )

delay_steps = np.diff(delay_values)
# np.fft assigns frequencies assuming a constant sampling interval, so reject
# irregular time grids instead of silently producing an incorrect frequency axis.
if not np.allclose(delay_steps, delay_steps[0]):
    raise ValueError("The delay values must be uniformly spaced for the FFT")

# Subtracting each trace's mean suppresses the zero-frequency (DC) component.
# The Hann window reduces leakage caused by the finite measurement interval.
intensity_zero_mean = intensity_interpolated - np.mean(
    intensity_interpolated,
    axis=1,
    keepdims=True,
)
window = np.hanning(len(delay_values))
windowed_intensity = intensity_zero_mean * window

# rfft returns the non-negative half of the spectrum for real input. Its native
# frequency unit is cycles/ps, which is numerically equal to THz. Multiplication
# by 2/sum(window) compensates for the one-sided spectrum and Hann-window gain.
frequencies = np.fft.rfftfreq(len(delay_values), d=delay_steps[0])
fourier_amplitude = (
    2.0
    * np.abs(np.fft.rfft(windowed_intensity, axis=1))
    / np.sum(window)
)

# Drop the DC bin and convert THz to GHz for both the CSV and frequency plot.
output_frequencies_ghz = frequencies[1:] * 1000.0
output_amplitude = fourier_amplitude[:, 1:]

# Record enough processing information to reproduce and interpret the FFT while
# deliberately omitting the source CSV's unrelated acquisition metadata.
missing_delay_values = delay_values[np.isnan(intensity).any(axis=0)]
frequency_step_ghz = output_frequencies_ghz[1] - output_frequencies_ghz[0]
fourier_metadata = [
    ("sourceFile", fileData.name),
    ("sampleName", input_metadata.get("sampleName", "not available")),
    ("ZETA_deg", input_metadata.get("ZETA_deg", "not available")),
    ("D_ZETA_deg", input_metadata.get("D_ZETA_deg", "not available")),
    (
        "ZETA_SYMMETRY",
        input_metadata.get("ZETA_SYMMETRY", "not available"),
    ),
    ("CX0_pixels", input_metadata.get("CX0_pixels", "not available")),
    ("CY0_pixels", input_metadata.get("CY0_pixels", "not available")),
    ("timeSampleCount", len(delay_values)),
    ("timeStep_ps", f"{delay_steps[0]:.12g}"),
    ("timeRange_ps", f"[{delay_values[0]:.12g}, {delay_values[-1]:.12g}]"),
    (
        "missingTimeSteps_ps",
        "[" + ", ".join(f"{value:.12g}" for value in missing_delay_values) + "]",
    ),
    ("dataGapTreatment", "linear interpolation along time for each Q"),
    ("DCRemoval", "subtract mean of each Q trace"),
    ("windowFunction", "Hann"),
    ("FFT", "real FFT along time axis"),
    ("amplitudeNormalization", "2 / sum(Hann window)"),
    ("zeroFrequencyOmitted", True),
    ("frequencyBinCount", len(output_frequencies_ghz)),
    ("frequencyStep_GHz", f"{frequency_step_ghz:.12g}"),
    (
        "frequencyRange_GHz",
        (
            f"[{output_frequencies_ghz[0]:.12g}, "
            f"{output_frequencies_ghz[-1]:.12g}]"
        ),
    ),
]

fourier_csv = fileData.with_name(f"{fileData.stem}_Fourier.csv")
with fourier_csv.open("w", newline="", encoding="utf-8") as output_file:
    # Reuse the source delimiter and the analysisMultiFrame.py layout:
    # metadata, ***DATA***, column names, units, and finally numeric rows.
    writer = csv.writer(
        output_file,
        delimiter=dialect.delimiter,
        lineterminator="\n",
    )
    writer.writerows(fourier_metadata)
    writer.writerow(["***DATA***"])
    writer.writerow(["Q", "f", "Intensity"])
    writer.writerow([column_units[0], "GHz", column_units[2]])

    # Q is the primary sort key; frequency increases within each Q block.
    for q_index, q_value in enumerate(q_values):
        for frequency_index, frequency_ghz in enumerate(output_frequencies_ghz):
            writer.writerow(
                [
                    f"{q_value:.12g}",
                    f"{frequency_ghz:.12g}",
                    f"{output_amplitude[q_index, frequency_index]:.12g}",
                ]
            )

print(f"Fourier transform data saved to '{fourier_csv}'")

# Plot the original time-domain data for comparison. A symmetric colour scale
# gives positive and negative intensity changes equal visual weight.
colour_limit = np.max(np.abs(finite_intensity))
figure, axes = plt.subplots(layout="constrained")
plot = axes.pcolormesh(
    delay_values,
    q_values,
    intensity,
    shading="auto",
    cmap="seismic",
    norm=TwoSlopeNorm(vmin=-colour_limit, vcenter=0.0, vmax=colour_limit),
)

axes.set_xlabel(f"{column_names[1]} ({column_units[1]})")
axes.set_ylabel(f"{column_names[0]} ({column_units[0]})")
axes.set_title(analysisName)
figure.colorbar(
    plot,
    ax=axes,
    label=f"{column_names[2]} ({column_units[2]})",
)

# Plot the one-sided Fourier amplitude. Transposition is required because
# pcolormesh expects array dimensions in [vertical frequency, horizontal Q] order.
fourier_figure, fourier_axes = plt.subplots(layout="constrained")
fourier_plot = fourier_axes.pcolormesh(
    q_values,
    output_frequencies_ghz,
    output_amplitude.T,
    shading="auto",
    cmap="cividis",
)
fourier_axes.set_xlabel(f"{column_names[0]} ({column_units[0]})")
fourier_axes.set_ylabel("Frequency (GHz)")
fourier_axes.set_title("Fourier amplitude along the delay axis")
# Optional: uncomment to force every Q/frequency heatmap cell to be square.
# fourier_axes.set_box_aspect(len(output_frequencies_ghz) / len(q_values))
fourier_figure.colorbar(
    fourier_plot,
    ax=fourier_axes,
    label=f"Fourier amplitude ({column_units[2]})",
)

fourier_image = fileData.with_name(f"{fileData.stem}_Fourier.png")
# Save before plt.show(), because show may block until the GUI windows are closed.
fourier_figure.savefig(fourier_image, dpi=300)
print(f"Fourier transform image saved to '{fourier_image}'")

plt.show()
