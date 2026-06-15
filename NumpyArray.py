import numpy as np

# 1. Create 1D Array
arr = np.array([10, 20, 30, 40, 50])

print("=== 1D Array ===")
print(arr)

# 2. Access Elements
print("\n=== Access Elements ===")
print("First Element:", arr[0])
print("Third Element:", arr[2])

# 3. Array Information
print("\n=== Array Information ===")
print("Size:", arr.size)
print("Shape:", arr.shape)
print("Data Type:", arr.dtype)

# 4. Mathematical Operations
print("\n=== Mathematical Operations ===")
print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Maximum:", arr.max())
print("Minimum:", arr.min())

# 5. Arithmetic Operations
print("\n=== Arithmetic Operations ===")
print("Array + 5:", arr + 5)
print("Array * 2:", arr * 2)

# 6. Create 2D Array
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n=== 2D Array ===")
print(arr2)

# 7. Access Row and Column
print("\n=== Row and Column Access ===")
print("First Row:", arr2[0])
print("Second Column:", arr2[:, 1])

# 8. Student Marks Mini Project
marks = np.array([85, 90, 78, 92, 88])

print("\n=== Student Marks Analysis ===")
print("Marks:", marks)
print("Total Marks:", marks.sum())
print("Average Marks:", marks.mean())
print("Highest Mark:", marks.max())
print("Lowest Mark:", marks.min())