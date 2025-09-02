def max_rectangle_of_zero(matrix: list[list[int]]):
    if not matrix or not matrix[0]:
        return 0
    n, m = len(matrix), len(matrix[0])
    heights = [0] * m
    max_area = 0

    def largest_histogram_area(heights: list[int]):
        stack = []
        max_area = 0
        heights.append(0)
        for i , h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] -1
                max_area = max(max_area, height*width)
            stack.append(i)
        heights.pop()
        return max_area
    
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                heights[j] += 1
            else:
                heights[j] = 0
        max_area = max(max_area, largest_histogram_area(heights))
    return max_area


matrix = [
[0,0, 1,0,0],
[0,0, 1,1,0],
[0,0, 1,1,0],
[0,1, 1,1,0],
]

print(f"We finding the maximum area of zero {max_rectangle_of_zero(matrix)}")
