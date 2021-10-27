a = input(int())
b = input(int())
c = input(int())
d = input(int())
e = input(int())
f = input(int())
g = input(int())
h = input(int())
count = 0
print(a, b, c, d, e, f, g, h)
list1 = [(e, a), (f, b), (g, c), (h, d)]
for i in range(len(list1)):
    for j in range((len(list1))-i):
        if list1[j][1] > list1[j+1][1]:
            list1[j], list1[j+1] = list1[j+1], list1[j]
        elif list1[j][1] == list1[j+1][1]:
            if list1[j][0] > list1[j+1][0]:
                list1[j], list1[j+1] = list1[j+1], list1[j]


