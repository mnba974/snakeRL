import torch
import torch.nn.functional as F
import random
import numpy as np
from snakeGameBoard import snakeGame, Direction, Point, pygame_play
from model import Linear_QNet, PPOtrainer, board_net, CNNQNet, FCQNet
from helper import plot
from get_state import get_state, get_state_2, get_state_3
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = [Linear_QNet(12, 256, 4), board_net(100, 256, 4),CNNQNet(4, 512, 4),FCQNet(400, 256, 4) ]
value_nets = [Linear_QNet(12, 256, 1), board_net(100, 256, 1),CNNQNet(4, 512, 1),FCQNet(400, 256, 1) ]
state_functions = [get_state, get_state_2, get_state_3, get_state_3]

LR = 0.0001
GAMMA = 0.9
epochs = 10
class PPO:
    def __init__(self,choice = 3):
        self.n_games = 0
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = GAMMA # discount rate
        self.memory = []
        self.current_memory = []
        self.model = models[choice]
        self.value_net = value_nets[choice]
        self.trainer = PPOtrainer(self.model, self.value_net, lr=LR, gamma=self.gamma)
        self.state_function = state_functions[choice]

    def get_action(self, state):
        self.model.eval()  # Set to eval mode for inference
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            logits = self.model(state0)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            final_move = [0, 0, 0, 0]
            if random.random() < self.epsilon:
                action = random.randint(0, probs.shape[1] - 1)
            else:
                action = dist.sample().item()
            final_move[action] = 1
            old_prob = probs[0, action].item()
        return final_move, old_prob  # (see previous advice: return index, not one-hot)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def remember(self, state, action, reward, done, old_prob):
        self.current_memory.append((state, action, reward, done, old_prob))

    def train_long_memory(self, epochs, memory):
        states, actions, rewards, dones, old_probs = zip(*memory)
        self.trainer.train_step(states, actions, rewards, dones, old_probs, epochs)

def plot_losses(policy_losses, value_losses):
    plt.figure(2)
    plt.clf()
    plt.title('Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

def train(continue_training=False,model_name='model_fc.pth',value_name='value_net.pth'):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = PPO()
    game = pygame_play()
    cumulative_reward = 0
    if continue_training:
        print(f"Loading model from {model_name} and value net from {value_name}")
        agent.model.load(model_name)
        agent.value_net.load(value_name)
        agent.epsilon = 0.01
    while True:
        for i in range(epochs):
            done = False
            while not done:
                # get old state
                state_old = agent.state_function(game.game)

                # get move
                final_move, old_prob = agent.get_action(state_old)
                # perform move and get new state
                reward, done, score = game.play_step(final_move)
                cumulative_reward += reward

                # remember
                agent.remember(state_old, final_move, reward, done, old_prob)
            game.reset()
            agent.memory.append(agent.current_memory)
            agent.current_memory = []

        # train long memory, plot result
        for i in range(epochs):
            agent.train_long_memory(i, agent.memory[i])

        agent.memory = []
        agent.decay_epsilon()
        agent.n_games += 1

        if score > record:
            record = score
            agent.model.save()
            agent.value_net.save(f'value_net.pth')

        print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Cumulative Reward:', cumulative_reward/epochs)
        print(f"Epsilon: {agent.epsilon}, Policy Loss: {agent.trainer.policy_losses[-1]}, Value Loss: {agent.trainer.value_losses[-1]}")
        cumulative_reward = 0
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
        plot_losses(agent.trainer.policy_losses[-100:], agent.trainer.value_losses[-100:])


if __name__ == '__main__':
    train(continue_training=True,model_name='model_fc4.pth',value_name='value_net4.pth')