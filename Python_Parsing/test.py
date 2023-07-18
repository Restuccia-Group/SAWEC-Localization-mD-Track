my_array = [(0, 0), (1, 2), (2, 4)]
CSI = [10, 20, 30, 40, 50]

# Read all the first values from the array
first_values = [pair[0] for pair in my_array]

# Delete the indices from the CSI array
for value in first_values:
    if value in CSI:
        CSI.remove(value)

# Printing the updated CSI array
print(CSI)
