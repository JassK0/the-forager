import pygame
import random
import time
from matrixSearch import matSearch

pygame.init()
screenMatrix = (300,300)
screen = pygame.display.set_mode(screenMatrix)
clock = pygame.time.Clock()

CELLSIZE = 10 #pixels
#forager start pos
def foragerPos():
    grid_x, grid_y = 15, 15
    return grid_x, grid_y
grid_x, grid_y = foragerPos()

#food 1 start pos
def foodPos():
    fx = random.randint(0,29)
    fy = random.randint(0,29)
    # fx,fy = 13,14
    return fx,fy
fx, fy = foodPos()



rows = screenMatrix[0]//CELLSIZE
cols = screenMatrix[1]//CELLSIZE


world = [[0 for _ in range(cols)] for _ in range(rows)]
world[grid_y][grid_x] = 1 #forager
world[fy][fx] = 2 #food

# print(matSearch(world,1))





def getFov(world, grid_x, grid_y, size=15):
    radius = size // 2
    inView = False
    fov = []

    for x in range(grid_x - radius, grid_x + radius + 1):
        row = []
        for y in range(grid_y - radius, grid_y + radius + 1):

            # Check boundaries first
            if 0 <= x < len(world) and 0 <= y < len(world[0]):
                value = world[y][x]
                row.append(value)

                # Detect food
                if value == 2:      # 2 = food
                    inView = True

            else:
                # out of bounds tile
                row.append(None)

        fov.append(row)

    return fov, inView




hunger = 100
def betterForagerMovement(world, grid_x,grid_y,fx,fy,hunger):
    time.sleep(0.1)
    fov, inView = getFov(world, grid_x, grid_y)
    if inView and hunger <= 50:
        print("Hungry, Searching for food")
        if fx > grid_x:
            grid_x += 1
        if fx < grid_x:
            grid_x -= 1
        if fy > grid_y:
            grid_y += 1
        if fy < grid_y:
            grid_y -= 1
        
        
    else:
        print("not in view")
        grid_x += random.choice([-1,0,1])
        grid_y += random.choice([-1,0,1])
    
    hunger -= 1
    print(f"hunger: {hunger}")
        
    return grid_x,grid_y,hunger
    
    
    
    



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # move randomly
    # time.sleep(0.3)
    grid_x, grid_y, hunger = betterForagerMovement(world, grid_x, grid_y, fx, fy,hunger)
    
    #handle 0 hunger
    if hunger <= 0:
        print("The forager has starved...")
        running = False

    # keep the forager inside bounds
    grid_x = max(0, min(grid_x, cols-1))
    grid_y = max(0, min(grid_y, rows-1))
    
    #update world state per frame
    world = [[0 for _ in range(cols)] for _ in range(rows)]
    world[grid_y][grid_x] = 1
    world[fy][fx] = 2
    
    
    fov, inView = getFov(world, grid_x, grid_y)
    print("Food in view:", inView)

    # convert grid coords to pixel coords for drawing
    px = grid_x * CELLSIZE
    py = grid_y * CELLSIZE
    
    


        

    screen.fill((120, 10, 0))
    
    
    # Draw FOV rectangle
    radius = 7  # since 15x15 FOV, radius is 7 tiles
    fov_size = 15 * CELLSIZE  # 15 tiles * 10 pixels = 150 pixels

    # Convert player grid coords to pixel coords
    player_px = grid_x * CELLSIZE
    player_py = grid_y * CELLSIZE

    # Calculate top-left corner of FOV in pixels
    fov_x = player_px - (radius * CELLSIZE)
    fov_y = player_py - (radius * CELLSIZE)

    # Draw the FOV box outline
    pygame.draw.rect(screen, (0, 0, 255), (fov_x, fov_y, fov_size, fov_size), 2)
    
    #check food eat
    if fx == grid_x and fy == grid_y:
        hunger += 100
        world[fy][fx] = 0
        fx, fy = foodPos()
        world[fy][fx] = 2
    
    # DRAW FOOD
    food_px = fx * CELLSIZE
    food_py = fy * CELLSIZE
    pygame.draw.circle(screen, (255,0,0), (food_px, food_py), 5)

        
    pygame.draw.circle(screen, (0, 255, 0), (px, py), 5)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
