import torch
import torch.nn.functional as F
import numpy as np
from snakeGameBoard import snakeGame
from model import Linear_QNet, PPOtrainer, board_net, CNNQNet, FCQNet
from helper import plot
from get_state import get_state, get_state_2, get_state_3
import matplotlib.pyplot as plt
import multiprocessing as mp
import random

models = [Linear_QNet(12, 256, 4), board_net(100, 256, 4),CNNQNet(4, 512, 4),FCQNet(400, 256, 4) ]
value_nets = [Linear_QNet(12, 256, 1), board_net(100, 256, 1),CNNQNet(4, 512, 1),FCQNet(400, 256, 1) ]
state_functions = [get_state, get_state_2, get_state_3, get_state_3]

LR = 0.0001
GAMMA = 0.9
epochs = 10
n_agents = 5
n_workers = 5

class PPO:
    def __init__(self,choice = 3):
        self.n_games = 0
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = GAMMA # discount rate
        self.memory = []
        self.model = models[choice]
        self.value_net = value_nets[choice]
        self.trainer = PPOtrainer(self.model, self.value_net, lr=LR, gamma=self.gamma)
        self.state_function = state_functions[choice]

    def get_action(self, state):
        self.model.eval()  # Set to eval mode for inference
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
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
        self.memory.append((state, action, reward, done, old_prob))

    def train_long_memory(self):
        states, actions, rewards, dones, old_probs = zip(*self.memory)
        self.trainer.train_step(states, actions, rewards, dones, old_probs)

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

def run_step(agent, game):
    # get old state
    state_old = agent.state_function(game)

    # get move
    final_move, old_prob = agent.get_action(state_old)
    # perform move and get new state
    reward, done, score = game.play_step(final_move)

    # remember
    agent.remember(state_old, final_move, reward, done, old_prob)

    return done, reward, score

def run_episode(agent, game):
    done = False
    cumulative_reward = 0
    while not done:
        done, reward, score = run_step(agent, game)
        cumulative_reward += reward
    return agent.memory,cumulative_reward,score


def train(continue_training=False,model_name='model_fc.pth',value_name='value_net.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agents = [PPO() for _ in range(n_agents)]
    games = [snakeGame() for _ in range(n_agents)]
    scores = [0 for _ in range(n_agents)]
    cumulative_rewards = [0 for _ in range(n_agents)]

    if continue_training:
        print(f"Loading model from {model_name} and value net from {value_name}")
        for agent in agents:
            agent.model.load(model_name)
            agent.value_net.load(value_name)
            agent.epsilon = 0.01

    while True:
        # pass the models to cpu
        for agent in agents:
            agent.model.to('cpu')
            agent.value_net.to('cpu')

        # Create process pool and run episodes in parallel
        print(f"Running {n_agents} episodes in parallel")
        with mp.Pool(processes=n_workers) as pool:
            # Map run_episode to agent-game pairs
            results = pool.starmap(run_episode, 
                                 [(agent, game) for agent, game in zip(agents, games)])
            
            # Unpack results
            for i, (memory, cumulative_reward, score) in enumerate(results):
                scores[i] = score
                cumulative_rewards[i] = cumulative_reward
                agents[i].memory = memory
        
        # pass the models to gpu
        for agent in agents:
            agent.model.to(device)
            agent.value_net.to(device)

        print(f"Training long memory for {n_agents} agents")
        # train long memory, plot result
        for agent,game in zip(agents,games):
            agent.train_long_memory()
            agent.memory = []
            agent.decay_epsilon()
            game.reset()
            agent.n_games += 1

        if max(scores) > record:
            record = max(scores)
            agent.model.save()
            agent.value_net.save(f'value_net_{i}.pth')

        print('Game', agent.n_games, 'Score', np.mean(scores), 'Record:', record, 'Cumulative Reward:', np.mean(cumulative_rewards))
        print(f"Epsilon: {agent.epsilon}, Policy Loss: {agent.trainer.policy_losses[-1]}, Value Loss: {agent.trainer.value_losses[-1]}")
        cumulative_rewards = [0 for _ in range(n_agents)]
        plot_scores.append(np.mean(scores))
        total_score += np.mean(scores)
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
        plot_losses(agent.trainer.policy_losses[-100:], agent.trainer.value_losses[-100:])


if __name__ == '__main__':
    train(continue_training=True,model_name='model_fc4.pth',value_name='value_net4.pth')