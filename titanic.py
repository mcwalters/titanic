
import csv

import matplotlib
matplotlib.use('Agg')                       #use backend that doesn't use a window
import matplotlib.pyplot as plt

import numpy as np
import pylab as pl

data=[]                                                     # Create a variable to hold the data

with open('train.csv', 'rb') as csvfile:
    csv_file_object = csv.reader(csvfile)
    header = csv_file_object.next()                             # Skip the fist line as it is a header
    for row in csv_file_object:
        data.append(row)                        # adding each row to the data variable
#        print ', '.join(row)


data = np.array(data)                       # Then convert from a list to an array

print header
print data[0]

# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

# I can now find the stats of all the women on board,
# by making an array that lists True/False whether each row is female
women_only_stats = data[0::,4] == "female" 	# This finds where all the women are
men_only_stats = data[0::,4] != "female" 	# This finds where all the men are (note != means 'not equal')

print men_only_stats

# I can now filter the whole data, to find statistics for just women, by just placing
# women_only_stats as a "mask" on my full data -- Use it in place of the '0::' part of the array index. 
# You can test it by placing it there, and requesting column index [4], and the output should all read 'female'
# e.g. try typing this:   data[women_only_stats,4]
women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

# and derive some statistics about them
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

##### Plotting age v fare, with notation for gender/survival ######


women_only_stats = data[0::,4] == "female" 	# This finds where all the women are
men_only_stats = data[0::,4] != "female" 	# This finds where all the men are (note != means 'not equal')

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print data[women_only_stats,9]

wom_sur_age = data[women_only_stats,9].astype(np.float)
wom_sur_fare = data[women_only_stats,9].astype(np.float)
#wom_die_age = data[women_only_stats,1].astype(np.float)
#wom_die_fare
men_sur_age = data[men_only_stats,5] #.astype(np.float)
men_sur_fare = data[men_only_stats,9] #.astype(np.float)
#men_die_age
#men_die_fare

plt.plot (wom_sur_age, wom_sur_fare, 'mo')
plt.savefig('myfigure')


######## End Plotting Section ###########

# Now that I have my indicator that women were much more likely to survive,
# I am done with the training set.
# Now I will read in the test file and write out my simplistic prediction:
# if female, then model that she survived (1) 
# if male, then model that he did not survive (0)

# First, read in test.csv
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. Call it something descriptive
# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
# Write out the PassengerId, and my prediction.

predictions_file = open("newpredictions.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in test_file_object:									# For each row in test file,
    if row[3] == 'female':										# is it a female, if yes then
        predictions_file_object.writerow([row[0], "1"])			# write the PassengerId, and predict 1
    else:														# or else if male,
        predictions_file_object.writerow([row[0], "0"])			# write the PassengerId, and predict 0.
test_file.close()												# Close out the files.
predictions_file.close()
