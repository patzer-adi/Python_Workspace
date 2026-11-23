
d = {"a": 1, "b": 2, "c": 3}

print(d.copy())             # shallow copy
print(dict.fromkeys(['x','y'], 0)) # new dict with same value
print(d.get("a"))           # get value of 'a'
print(d.items())            # all items
print(d.keys())             # all keys
print(d.values())           # all values
print(d.pop("b"))           # remove 'b'
print(d.popitem())          # remove last inserted
print(d.setdefault("d", 4)) # insert if not exists
d.update({"e": 5})          # add/overwrite
print(d)                    # final dictionary
d.clear()                   # empty dictionary
print(d)


hell_info = {
    
    "batch": 32,
    "Subjects": 4,
    "Feeling": "hell"
}

print(hell_info)

hell_info.update({"Dept":"ISSC"})
print(hell_info)

# dict.clear()

copy_hell = hell_info.copy()
print("copy:",hell_info)


iter = copy_hell.fromkeys(['batch','Dept'],'bad')
print(iter)

print(copy_hell.get('batch'))


print(copy_hell.items())

print(copy_hell.values())

print(iter.pop('batch'))
print(iter.popitem())

print(iter.)