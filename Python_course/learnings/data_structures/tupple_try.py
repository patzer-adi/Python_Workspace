t = (1, 2, 3, 2, 5)

# Methods
print(t.count(2))            # 2 occurs twice
print(t.index(3))            # first index of 3 is 2

# Built-in functions
print(len(t))                # 5
print(max(t))                # 5
print(min(t))                # 1
print(sum(t))                # 13
print(sorted(t))             # [1, 2, 2, 3, 5]
print(list(reversed(t)))     # [5, 2, 3, 2, 1]
print(list(enumerate(t)))    # [(0,1),(1,2),(2,3),(3,2),(4,5)]
print(any(t))                # True (since not all are 0/False)
print(all(t))                # True (no 0/False in tuple)
print(tuple("abc"))          # ('a', 'b', 'c')
