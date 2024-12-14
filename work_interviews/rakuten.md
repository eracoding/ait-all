# Entry level: Data analytics/Data Science, Rakuten Payment

The interview process: I have not prepared for the interview at all (it was basically soft skills interview).
Since I have not prepared, I was improvising thats why I had some problems/missed points:
1. Self-introductory - give story line based introduction about yourself (not so much, most important and relevant to the position)
2. Figure out some use cases, and experience' stories that can make sense - I have told about my work experience (face recognition)
3. Some basic question related to the position - I was asked if I had a big data csv file how I would approach to process it (answered as parallelizing the preprocessing) - better is to divide into chunks, or some logic-based with groupping, clustering
4. Software experience, what was implemented in my work experience project - got comments related to cloud technologies
5. Then moved to other interviewer who probably based on data science, was asked basic questions regarding ml.
6. 6 minutes to ask question that concerned me - I have not prepared them beforehand, thats why come up with some silly questions - If I am successful, do I need to move to Japan or can work remote (regarding the program)? Then asked what projects are you working on (this was more specific - good one probably)?

Have not got answer yet, but I think it was failed interview since I was not specific, my thoughts were swimming and not straight. Language usage was also not business oriented.

# Problem: Solar Power Plants

We have a map of an area where two square solar power plants are planned to be built. The map is given as a rectangular matrix `A` of boolean values. `A[R][C]` is `True` if a place in the R-th row and C-th column can be covered by a solar power plant, and `False` if it is not possible. Each solar power plant must be built in a square shape. Note that a single cell is sufficient for a valid solar power plant. Solar power plants cannot share any cell and should be of the same size.

### Goal:
Determine the area of each of the two largest possible solar power plants that can be built in a square shape. If it is not possible to build two solar power plants, return `0`.

### Function Signature:
```python
def solution(A: List[List[bool]]) -> int:
```
- **Input**: Matrix `A` consisting of `N` rows and `M` columns.
- **Output**: Integer representing the area of the largest possible square solar power plants that can be built, or `0` if no two squares can be constructed.

---

### Examples:

#### Example 1:
Given the `4x4` map:
```
A = [
    [False, True,  False, True],
    [True,  True,  True,  False],
    [True,  True,  True,  True],
    [False, True,  True,  True]
]
```
The function should return `4`. The cells where the solar power plants can be built are shown below:

- Cells on which the solar power plant can be built are colored white.
- Cells unavailable are colored black.
- Two solar power plants are hatched. The area of each plant is equal to `4`, as they occupy `2x2` cells.

#### Example 2:
Given the `6x4` map:
```
A = [
    [True,  True,  True,  True],
    [True,  True,  True,  True],
    [True,  True,  True,  True],
    [True,  True,  True,  True],
    [True,  True,  True,  True],
    [True,  True,  True,  True]
]
```
The function should return `9`. Two solar power plants with an area of `3x3` can be built without overlapping.

#### Example 3:
Given the `3x5` map:
```
A = [
    [False, False, False, False, False],
    [False, False, True,  False, False],
    [False, False, False, False, False]
]
```
The function should return `0`. It is impossible to build two solar power plants.

---

### Constraints:
- `1 ≤ N, M ≤ 700`
- The matrix `A` contains only boolean values.

---

### Assumptions:
1. The solution must be efficient to handle large values of `N` and `M`.
2. Solar power plants must not overlap and must be of the same size.
3. If no valid solution exists, return `0`.

Test was on codility.

### Solution provided:
```python
def solution(A):
    # Check input validity
    if not A or not A[0]:
        return 0
    
    N, M = len(A), len(A[0])
    
    # Precompute max square sizes possible at each cell using dynamic programming
    def compute_max_squares():
        # dp will store the max square size that can start at each cell
        dp = [[0] * M for _ in range(N)]
        
        # Initialize first row and column
        for r in range(N):
            dp[r][0] = 1 if A[r][0] else 0
        for c in range(M):
            dp[0][c] = 1 if A[0][c] else 0
        
        # Fill dp table
        for r in range(1, N):
            for c in range(1, M):
                if A[r][c]:
                    dp[r][c] = min(
                        dp[r-1][c], 
                        dp[r][c-1], 
                        dp[r-1][c-1]
                    ) + 1
                else:
                    dp[r][c] = 0
        
        return dp
    
    # Find all non-overlapping squares of a given max size
    def find_two_largest_squares(dp, max_size):
        # Find all squares of max_size
        squares = []
        for r in range(N):
            for c in range(M):
                if dp[r][c] == max_size:
                    # Find top-left corner of the square
                    top_left_r = r - max_size + 1
                    top_left_c = c - max_size + 1
                    squares.append((top_left_r, top_left_c, max_size))
        
        # Check for non-overlapping squares
        for i in range(len(squares)):
            for j in range(i+1, len(squares)):
                r1, c1, size1 = squares[i]
                r2, c2, size2 = squares[j]
                
                # Check if squares overlap
                if not (r1 + size1 <= r2 or r2 + size2 <= r1 or 
                        c1 + size1 <= c2 or c2 + size2 <= c1):
                    continue
                
                # If we found two non-overlapping squares, return their size squared
                return size1 * size1
        
        return 0
    
    # Main solving logic
    # Try from largest possible square size to smallest
    dp = compute_max_squares()
    max_size = min(N, M)
    
    while max_size > 0:
        result = find_two_largest_squares(dp, max_size)
        if result > 0:
            return result
        max_size -= 1
    
    return 0
```
