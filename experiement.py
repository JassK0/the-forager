rows = 30
cols=30




world = [[0 for _ in range(cols)] for _ in range(rows)]

world[11][13] = 1 #forager
world[15][15] = 2 #food

foragerFOV = [[0 for _ in range(5)] for _ in range(5)]

