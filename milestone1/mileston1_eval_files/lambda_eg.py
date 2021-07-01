# Python program to print positive Numbers in a List

# list of numbers
list1 = [-10, 21, 4, -45, -66, 93, -11]


# we can also print positive no's using lambda exp.
pos_nos = list(filter(lambda x: (x >= 0), list1))

print("Positive numbers in the list: ", *pos_nos)
