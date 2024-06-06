nums = [3,2,2,3]
val = 3

i = 0
for j in range(len(nums)):
    # If the current element is not equal to val,
    # overwrite the element at position i with the current element
    # and increment i to move to the next position
    if nums[j] != val:
        nums[i] = nums[j]
        i += 1
