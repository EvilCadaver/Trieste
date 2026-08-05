import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

folderData = r"./Output_DS/FeRh_A04"
analysisName = r"FeRh_A04_Scan_068_Q_vs_delay_a0deg_d180deg_sym2.csv"
fileData = Path(folderData) / analysisName

if not fileData.exists():
    raise FileNotFoundError(f"File '{fileData}' doesn't exist!")

with open(fileData, newline="", encoding="utf-8") as inputCSV:
    dialect = csv.Sniffer().sniff(inputCSV.read(8096))
    print(f"Delimiters detected: {repr(dialect.delimiter)}")
    inputCSV.seek(0)
    reader = csv.reader(inputCSV, dialect)
    input_metadata = {}

    for row in reader:
        if row and row[0].strip() == "***DATA***":
            break
        if len(row) >= 2:
            input_metadata[row[0].strip()] = row[1].strip()
    else:
        raise ValueError("The CSV file does not contain a ***DATA*** marker")

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

q_indexes = np.searchsorted(q_values, data[:, 0])
delay_indexes = np.searchsorted(delay_values, data[:, 1])
intensity[q_indexes, delay_indexes] = data[:, 2]

finite_intensity = intensity[np.isfinite(intensity)]
if finite_intensity.size == 0:
    raise ValueError("The intensity column contains no finite values")

# The FFT requires finite, uniformly sampled traces. Fill missing values by
# linear interpolation along the delay axis for each Q value.
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
if not np.allclose(delay_steps, delay_steps[0]):
    raise ValueError("The delay values must be uniformly spaced for the FFT")

# Remove the DC component and apply a Hann window to reduce spectral leakage.
intensity_zero_mean = intensity_interpolated - np.mean(
    intensity_interpolated,
    axis=1,
    keepdims=True,
)
window = np.hanning(len(delay_values))
windowed_intensity = intensity_zero_mean * window

# Because delay is measured in ps, cycles/ps are numerically equal to THz.
frequencies = np.fft.rfftfreq(len(delay_values), d=delay_steps[0])
fourier_amplitude = (
    2.0
    * np.abs(np.fft.rfft(windowed_intensity, axis=1))
    / np.sum(window)
)

# The zero-frequency bin represents the removed mean and is omitted from output.
output_frequencies_ghz = frequencies[1:] * 1000.0
output_amplitude = fourier_amplitude[:, 1:]

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
    writer = csv.writer(
        output_file,
        delimiter=dialect.delimiter,
        lineterminator="\n",
    )
    writer.writerows(fourier_metadata)
    writer.writerow(["***DATA***"])
    writer.writerow(["Q", "f", "Intensity"])
    writer.writerow([column_units[0], "GHz", column_units[2]])

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

# A symmetric colour scale makes positive and negative intensity changes comparable.
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

# Omit the zero-frequency bin, which represents the removed mean intensity.
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
# # Match the axes box to the grid dimensions so every heatmap cell is square.
# fourier_axes.set_box_aspect(len(output_frequencies_ghz) / len(q_values))
fourier_figure.colorbar(
    fourier_plot,
    ax=fourier_axes,
    label=f"Fourier amplitude ({column_units[2]})",
)

fourier_image = fileData.with_name(f"{fileData.stem}_Fourier.png")
fourier_figure.savefig(fourier_image, dpi=300)
print(f"Fourier transform image saved to '{fourier_image}'")

plt.show()
