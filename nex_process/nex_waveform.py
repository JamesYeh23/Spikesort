from nex_sdk import NexFileReaders, NexFileData
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path

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