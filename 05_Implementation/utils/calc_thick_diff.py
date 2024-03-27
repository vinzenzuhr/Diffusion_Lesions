import csv 

file1="test1.csv"
file2="test2.csv"
result="result.csv"

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
diff = [float(values[i])-float(values_2[i]) for i in range(1, len(values))]
diff.insert(0, values[0])

with open(result, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)   
    writer.writerow(diff)