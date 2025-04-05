from nex_sdk import NexFileReaders, NexFileData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
import re

def nex_to_dataframe(nexfile: Union[str, Path]) -> pd.DataFrame:
    """
    Read a Nex (.nex) file and extract waveform data into a Pandas DataFrame.

    Parameters:
        nexfile (str or Path): Path to the .nex file.

    Returns:
        pd.DataFrame: A DataFrame where each row is a spike with metadata
                      and waveform values as separate columns (v0..vN).
    """
    nexfile = Path(nexfile)

    # Create reader and parse file
    reader = NexFileReaders.NexFileReader()
    file_data = reader.ReadNexFile(str(nexfile))

    records = []
    for wf in file_data.Waveforms:
        unit_name = wf.Name
        timestamps = wf.Timestamps
        wave_values = wf.Values

        for ts, vals in zip(timestamps, wave_values):
            row = {
                "Name": unit_name,
                "Timestamp": ts,
                "Amplitude": float(np.ptp(vals))
            }
            row.update({f"v{i}": val for i, val in enumerate(vals)})
            records.append(row)

    df = pd.DataFrame(records)
    return df
def plot_unit_waveforms(df: pd.DataFrame, channel: int, unit: str = "U", compare_unsorted: bool = False):
    """
    Plot all waveforms and the mean waveform for a specific unit from a Nex DataFrame.
    Optionally compare with the unsorted unit on the same channel.

    Parameters:
        df (pd.DataFrame): The DataFrame generated from a .nex file.
        channel (int): Channel number (1–64).
        unit (str, optional): Unit letter (a, b, ..., U for unsorted). Default is "U".
        compare_unsorted (bool, optional): If True, overlay unsorted unit waveforms in gray.

    Raises:
        ValueError: If inputs are invalid or unit not found.
    """
    if not (1 <= channel <= 64):
        raise ValueError("Channel must be between 1 and 64.")
    if not re.fullmatch(r"[a-zA-Z]", unit):
        raise ValueError("Unit must be a single letter (a–z, A–Z).")

    # Ensure required columns exist
    if "Name" not in df.columns:
        raise ValueError("DataFrame must include a 'Name' column.")
    waveform_cols = [col for col in df.columns if col.startswith("v")]
    if len(waveform_cols) != 32:
        raise ValueError("Expected 32 waveform columns (v0 to v31), found: " + str(len(waveform_cols)))

    unit_name = f"SPK{channel:02d}{unit}_wf"
    df_unit = df[df["Name"] == unit_name]
    if df_unit.empty:
        raise ValueError(f"No spikes found for unit '{unit_name}'.")
    n_spikes = len(df_unit)

    # Optional: overlay unsorted unit if comparing
    if compare_unsorted and unit.upper() != "U":
        unsorted_name = f"SPK{channel:02d}U_wf"
        df_unsorted = df[df["Name"] == unsorted_name]
    else:
        df_unsorted = pd.DataFrame()

    # Begin plotting
    plt.figure(figsize=(10, 6))

    # Plot unsorted spikes (if applicable)
    if not df_unsorted.empty:
        for _, row in df_unsorted.iterrows():
            waveform = row[waveform_cols].values.astype(float)
            plt.plot(waveform, color="gray", alpha=0.05, linewidth=0.8, label="_nolegend_")

    # Plot sorted unit spikes
    for _, row in df_unit.iterrows():
        waveform = row[waveform_cols].values.astype(float)
        plt.plot(waveform, color="blue", alpha=0.1)

    # Plot mean waveform
    mean_waveform = df_unit[waveform_cols].mean().values.astype(float)
    plt.plot(mean_waveform, color="red", linewidth=2, label="Average Waveform")

    # Text box showing spike count
    plt.text(0.05, 0.95, f"# Spikes: {n_spikes}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    title = f"Waveforms for {unit_name}"
    if not df_unsorted.empty:
        title += f" (compared to unsorted: {len(df_unsorted)})"
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_nex_units(data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """
    Summarizes waveform info from a .nex file or preprocessed DataFrame.
    
    Parameters:
        data (str, Path, or pd.DataFrame): Path to .nex file or a processed DataFrame.

    Returns:
        pd.DataFrame: A table of [channel, unit, spike_count]
    """
    if isinstance(data, (str, Path)):
        df = nex_to_dataframe(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a DataFrame or a path to a .nex file.")

    if "Name" not in df.columns:
        raise ValueError("DataFrame must contain a 'Name' column.")

    pattern = re.compile(r"SPK(\d{2})([a-zA-Z])_wf")
    summary = []

    for name, group in df.groupby("Name"):
        match = pattern.match(name)
        if not match:
            continue  # Skip non-matching entries

        ch_num = int(match.group(1))
        unit = match.group(2)
        n_spikes = len(group)
        summary.append((ch_num, unit, n_spikes))

    # Build a DataFrame from the results
    summary_df = pd.DataFrame(summary, columns=["Channel", "Unit", "SpikeCount"])

    # Ensure unsorted units 'U' are shown even if no sorted ones exist
    all_channels = summary_df["Channel"].unique()
    for ch in range(1, 65):
        if ch not in all_channels:
            continue
        if not ((summary_df["Channel"] == ch) & (summary_df["Unit"] == "U")).any():
            summary_df = pd.concat([
                summary_df,
                pd.DataFrame([{"Channel": ch, "Unit": "U", "SpikeCount": 0}])
            ], ignore_index=True)

    # Sort by channel and unit
    summary_df.sort_values(by=["Channel", "Unit"], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    return summary_df