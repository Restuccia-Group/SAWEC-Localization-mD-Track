A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 34, 0, 2, 3, 4,]


lower_value = A[0]
a = 80000

for i in range(len(A)):
    if A[i] < lower_value:
        lower_value = A[i]
        a *= 2

        A[i] += a



        #print(lower_value)

    else:
        lower_value = A[i]

        A[i] += a


print("Modified Array:", A)

