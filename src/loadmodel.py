from agent import Agent
from PPOagent import PPO
from snakeGameBoard import pygame_play
import pygame

def play(choice = 3, model_name = 'model_fc4.pth', value_name = 'value_net4.pth'):
    record = 0
    agent = PPO(choice)
    agent.model.load(model_name)
    agent.value_net.load(value_name)
    agent.epsilon = 0
    game = pygame_play()
    game.game.speed = 10
    while True:
        # get old state
        state_old = agent.state_function(game.game)

        # get move
        final_move, old_prob = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        if score > record:
            record = score

        if done:
            game.reset()
            agent.n_games += 1

            print('Game', agent.n_games, 'Score', score, 'Record:', record)



if __name__ == '__main__':
    play(choice=3,model_name='model_fc4.pth',value_name='value_net4.pth')