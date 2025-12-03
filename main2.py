import pygame
import random
import time

pygame.init()
screenMatrix = (300,300)
screen = pygame.display.set_mode(screenMatrix)
clock = pygame.time.Clock()

CELLSIZE = 10

# continuous (float) position in pixels
pos_x = 15 * CELLSIZE
pos_y = 15 * CELLSIZE

def foodPos():
    return random.randint(0,29), random.randint(0,29)

fx, fy = foodPos()

rows = screenMatrix[0]//CELLSIZE
cols = screenMatrix[1]//CELLSIZE

world = [[0 for _ in range(cols)] for _ in range(rows)]

# increased starting hunger
hunger = 250

def getFov(world, grid_x, grid_y, size=15):
    radius = size // 2
    inView = False
    fov = []

    for x in range(grid_x - radius, grid_x + radius + 1):
        row = []
        for y in range(grid_y - radius, grid_y + radius + 1):
            if 0 <= y < len(world) and 0 <= x < len(world[0]):
                value = world[y][x]  # correct axis
                row.append(value)
                if value == 2:
                    inView = True
            else:
                row.append(None)
        fov.append(row)

    return fov, inView


# DECIMAL MOVEMENT SYSTEM
def betterForagerMovement(pos_x, pos_y, fx, fy, hunger):
    time.sleep(0.03)  # slight delay for readability

    grid_x = int(pos_x // CELLSIZE)
    grid_y = int(pos_y // CELLSIZE)

    _, inView = getFov(world, grid_x, grid_y)

    # tuned movement speeds
    explore_step = 3.0    # wandering
    search_step = 3.8     # going toward food

    if inView and hunger < 100:
        print("Seeking food")
        if fx > grid_x: pos_x += search_step
        if fx < grid_x: pos_x -= search_step
        if fy > grid_y: pos_y += search_step
        if fy < grid_y: pos_y -= search_step
    else:
        # smoother random drift
        pos_x += random.choice([-explore_step, 0, explore_step])
        pos_y += random.choice([-explore_step, 0, explore_step])

    hunger -= 0.3  # slower hunger decay
    return pos_x, pos_y, hunger


running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pos_x, pos_y, hunger = betterForagerMovement(pos_x, pos_y, fx, fy, hunger)

    if hunger <= 0:
        print("The forager has starved...")
        running = False

    # convert decimals to integer grid
    grid_x = int(pos_x // CELLSIZE)
    grid_y = int(pos_y // CELLSIZE)

    grid_x = max(0, min(grid_x, cols - 1))
    grid_y = max(0, min(grid_y, rows - 1))

    # update world
    world = [[0 for _ in range(cols)] for _ in range(rows)]
    world[grid_y][grid_x] = 1
    world[fy][fx] = 2

    fov, inView = getFov(world, grid_x, grid_y)
    print("Food in view:", inView)

    screen.fill((120, 10, 0))

    # draw FOV box
    radius = 7
    fov_size = 15 * CELLSIZE
    fov_x = pos_x - (radius * CELLSIZE)
    fov_y = pos_y - (radius * CELLSIZE)
    pygame.draw.rect(screen, (0, 0, 255), (fov_x, fov_y, fov_size, fov_size), 2)

    # eat food
    if fx == grid_x and fy == grid_y:
        hunger += 80   # reward increased for balance
        fx, fy = foodPos()

    # draw food
    food_px = fx * CELLSIZE
    food_py = fy * CELLSIZE
    pygame.draw.circle(screen, (255, 0, 0), (food_px, food_py), 5)

    # draw forager using float position
    pygame.draw.circle(screen, (0, 255, 0), (pos_x, pos_y), 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
