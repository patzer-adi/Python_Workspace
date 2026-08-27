ques = "Enter a string:\n"
ans = input(ques)

print(f"Original string: {ans}")

print("Upper",ans.upper())
print("Lower",ans.lower())
print("Capitalize",ans.capitalize())
print(f"Swap case: {ans.swapcase()}")
print(f"Strip whitespace {ans.strip()}")
print(f"Title case: {ans.title()}")
print(f"Count a in str: {ans.count('a')}")
print(f"Length of str: {len(ans)}")
print(f"Is alpha: {ans.isalpha()}")
print(f"Is digit: {ans.isdigit()}")
print(f"Is alnum: {ans.isalnum()}")


