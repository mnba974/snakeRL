from snakeGameBoard import snakeGame, Direction, Point
import numpy as np

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 1, head.y)
    point_r = Point(head.x + 1, head.y)
    point_u = Point(head.x, head.y - 1)
    point_d = Point(head.x, head.y + 1)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger up
        game.is_collision(point_u),

        # Danger right
        game.is_collision(point_r),

        # Danger left
        game.is_collision(point_l),

        # Danger down
        game.is_collision(point_d),
        
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        ]

    return np.array(state, dtype=int)

def get_state_2(game):
    state = game.board.flatten()
    return state

def get_state_3(game):
    """Convert board to multiple channels with spatial information preserved
    Returns a state with 4 channels:
    - Channel 0: Snake body
    - Channel 1: Snake head
    - Channel 2: Food
    - Channel 3: Snake direction as a heat map
    """
    board = game.board
    h, w = board.shape
    
    # Initialize channels
    snake_body = np.zeros((h, w))
    snake_head = np.zeros((h, w))
    food = np.zeros((h, w))
    direction = np.zeros((h, w))
    
    # Fill snake body channel (excluding head)
    snake_body[board == 1] = 1
    
    # Fill snake head channel
    if game.head.x >= 0 and game.head.x < h and game.head.y >= 0 and game.head.y < w:
        snake_head[game.head.x, game.head.y] = 1
    
    # Fill food channel
    if game.food.x >= 0 and game.food.x < h and game.food.y >= 0 and game.food.y < w:
        food[game.food.x, game.food.y] = 1
    
    # Create direction heat map
    # This creates a gradient in the direction the snake is moving
    x, y = game.head.x, game.head.y
    if game.direction == Direction.RIGHT:
        for i in range(w-x):
            direction[x+i, y] = 1 - (i/w)
    elif game.direction == Direction.LEFT:
        for i in range(x+1):
            direction[x-i, y] = 1 - (i/w)
    elif game.direction == Direction.DOWN:
        for i in range(h-y):
            direction[x, y+i] = 1 - (i/h)
    elif game.direction == Direction.UP:
        for i in range(y+1):
            direction[x, y-i] = 1 - (i/h)
    
    
    # Stack all channels
    state = np.stack([snake_body, snake_head, food, direction])
    return state.astype(np.float32)


