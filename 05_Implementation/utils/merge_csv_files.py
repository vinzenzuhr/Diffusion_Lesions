"""Merges multiple csv files into one csv file.

DL+Direct produces multiple csv for every MRI. We want to merge them into one csv for every MRI.
The new csv files will be saved in the same folder as the original csv files.

Args:
    path (str): Path to the folder containing the csv files.
"""

from pathlib import Path
import sys

path = Path(sys.argv[1]) 

write_header=True
folders = path.iterdir()
for folder in folders:
    csv_files = folder.rglob("*.csv")
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            header = f.readline()
            content = f.readline()
        with open(path / csv_file.name, 'a') as f:
            if write_header:
                f.write(header)  
            f.write(content) 
    write_header = False


