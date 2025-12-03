rows = 30
cols=30




world = [[0 for _ in range(cols)] for _ in range(rows)]

world[14][14] = 1 #forager
world[15][15] = 2 #food

foragerFOV = [[0 for _ in range(5)] for _ in range(5)]

        

def matSearch(mat, x):
    n = len(mat)
    m = (len(mat[0]))
    
    for i in range(n):
        for j in range(m):
            if mat[i][j] == x:
               return i,j
    return -1,-1





    
    
tup = matSearch(world,2)
print(matSearch(world, 2))
        