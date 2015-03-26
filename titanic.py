
import csv
import numpy as np

data=[]                                                     # Create a variable to hold the data

with open('train.csv', 'rb') as csvfile:
    csv_file_object = csv.reader(csvfile)
    header = csv_file_object.next()                             # Skip the fist line as it is a header
    for row in csv_file_object:
        data.append(row)                        # adding each row to the data variable
#        print ', '.join(row)


data = np.array(data)                       # Then convert from a list to an array

print data[0]
