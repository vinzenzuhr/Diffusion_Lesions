"""Calculates the difference between the results of two thickness measurements.

This script calculates the difference between the results of the thickness measurements, before and after lesion filling. 
It saves the results in a csv. 

Args:
    folder1 (str): Path to the first folder containing the thickness measurements.
    folder2 (str): Path to the second folder containing the thickness measurements.

"""

import csv 
from pathlib import Path
import sys

folder1 = Path(sys.argv[1]) # "dataset_eval/segm"
folder2 = Path(sys.argv[2]) # "lesion-filling-256-cond-lesions/segmentations_3D"

file_list1 = list(folder1.rglob("*result-thick.csv"))
file_list2 = list(folder2.rglob("*result-thick.csv"))
file_list3 = list(folder1.rglob("*result-thickstd.csv"))
file_list4 = list(folder2.rglob("*result-thickstd.csv"))
file_list5 = list(folder1.rglob("*result-vol.csv"))
file_list6 = list(folder2.rglob("*result-vol.csv"))
assert len(file_list1) == len(file_list2) == len(file_list3) == len(file_list4) == len(file_list5) == len(file_list6), "Different number of files in the folders"
lists = [[file_list1, file_list2], [file_list3, file_list4], [file_list5, file_list6]]

for list1, list2 in lists:
    for file1, file2 in zip(list1, list2):
        lines1 = []
        with open(file1, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            values = next(reader)
        with open(file2, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header_2 = next(reader)
            values_2 = next(reader)
        #start at 1 to skip patient name
        diff = [float(values[i]) - float(values_2[i]) for i in range(1, len(values))]
        diff.insert(0, values[0])

        result = file2.parent / (file2.stem + "_diff.csv")

        with open(result, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(header)   
            writer.writerow(diff)


