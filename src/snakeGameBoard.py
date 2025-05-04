import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time


def distance(a,b):
    return np.abs(a.x - b.x) + np.abs(a.y - b.y)

class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
    
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
YELLOW = (255, 255, 0)

BLOCK_SIZE = 20

class snakeGame:
    def __init__(self, w=10, h=10):
        self.w = w
        self.h = h
        self.board = np.zeros((h,w))
        self.speed = 1000
        # init game state
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.h, self.w))
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head, 
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]
        self.draw_snake()
        self.score = 0
        self.food = None
        self.direction = Direction.RIGHT
        self.frame_iteration = 0
        self._place_food()
        
    def draw_snake(self):
        for pt in self.snake:
            self.board[pt.x, pt.y] = 1
        self.board[self.head.x, self.head.y] = 3
        
    def _place_food(self):
        x = random.randint(0, (self.w-1))
        y = random.randint(0, (self.h-1))
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()
        else:
            self.board[self.food.x, self.food.y] = 2
            
    def play_step(self,action):
        self.frame_iteration += 1
        dir = self.get_direction(action)
        d1 = distance(self.head, self.food)
        self._move(dir)
        d2 = distance(self.head, self.food)
        self.snake.insert(0, self.head)
        reward = -0.001
        game_over = False
        if self.is_collision() or self.frame_iteration > 300*len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward = 0.5
            self._place_food()
        

        # 4. place new food or just move
        elif self.board[self.head.x, self.head.y] == 0:
            pt_last = self.snake.pop()
            self.board[pt_last.x, pt_last.y] = 0

        if d1 > d2:
            reward += 0.0005
        else:
            reward -= 0.001

        self.draw_snake()

        return reward, game_over, self.score
        
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w-1 or pt.x < 0 or pt.y > self.h-1 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def get_direction(self,action):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = directions.index(self.direction)
        new_index = np.argmax(action)

        if np.abs(index - new_index) == 2:
            return self.direction
        self.direction = directions[new_index]
        return self.direction
        
    def _move(self, dir):
        x = self.head.x
        y = self.head.y
        if dir == Direction.RIGHT:
            x += 1
        elif dir == Direction.LEFT:
            x -= 1
        elif dir == Direction.DOWN:
            y += 1
        elif dir == Direction.UP:
            y -= 1
        self.head = Point(x,y)


def terminate_play():
    game = snakeGame()
    while True:
        action = np.random.randn(4)
        reward, game_over, score = game.play_step(action)
        if game_over:
            break
        
        # Create a more readable visualization of the board
        board_viz = ''
        for row in game.board:
            for cell in row:
                if cell == 0:    # Empty space
                    board_viz += '⬛ '  # Black square
                elif cell == 1:  # Snake body
                    board_viz += '🟦 '  # Blue square
                elif cell == 2:  # Food
                    board_viz += '🟥 '  # Red square
                elif cell == 3:  # Snake head
                    board_viz += '🟨 '  # Yellow square
            board_viz += '\n'
        
        print("\033[H\033[J")  # Clear console
        print(f"Score: {score}")
        print(board_viz)
        time.sleep(0.5)


class pygame_play:
    def __init__(self):
        import pygame
        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.game = snakeGame()
        self.screen = pygame.display.set_mode((self.game.w*BLOCK_SIZE, self.game.h*BLOCK_SIZE))
        self.clock = pygame.time.Clock()
        self.clock.tick(self.game.speed)
    
    def reset(self):
        self.game.reset()
        
    def play_step(self, action):
        reward, game_over, score = self.game.play_step(action)
        self.draw_snake()
        self.clock.tick(self.game.speed)
        pygame.display.flip()
        return reward, game_over, score
        
    def draw_snake(self):
        self.screen.fill(BLACK)
        for pt in self.game.snake:
            pygame.draw.rect(self.screen, BLUE1, pygame.Rect(pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, YELLOW, pygame.Rect(self.game.head.x*BLOCK_SIZE, self.game.head.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.game.food.x*BLOCK_SIZE, self.game.food.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.game.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        pygame.display.flip()


