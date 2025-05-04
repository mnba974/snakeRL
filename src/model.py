import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000):
        """Initialize the buffer with given capacity.
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity          # Maximum size of the buffer
        self.buffer = []                  # List to store experiences
        self.priorities = []              # List to store priorities
        self.pos = 0                      # Position for circular buffer

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority.
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # For first experience, use priority 1.0, else use max priority in buffer
        max_priority = max(self.priorities) if self.priorities else 1.0

        # If buffer not full, append new experience
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            # Replace old experience at current position
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority

        # Update position for next insertion
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities.
        Args:
            batch_size: Number of experiences to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # If buffer is empty, return empty lists
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        # Convert priorities to probabilities
        probs = np.array(self.priorities)
        probs = probs / sum(probs)  # Normalize to sum to 1

        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)), 
            p=probs
        )

        # Get experiences for sampled indices
        samples = [self.buffer[idx] for idx in indices]
        
        # Unzip the samples into separate lists
        states, actions, rewards, next_states, dones = zip(*samples)

        return states, actions, rewards, next_states, dones, indices

    def update_priorities(self, indices, errors):
        """Update priorities for sampled experiences.
        Args:
            indices: Indices of experiences to update
            errors: New priority values (TD errors)
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5  # Small constant to prevent 0 priority

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.memory = PrioritizedReplayBuffer()  # Initialize PER buffer

    def train_step(self, state, action, reward, next_state, done):
        # Add experience to buffer
        self.memory.add(state, action, reward, next_state, done)

        # Don't start training until we have enough samples
        if len(self.memory.buffer) < 100:
            return

        # Sample batch of experiences
        states, actions, rewards, next_states, dones, indices = self.memory.sample(64)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        # Get current Q values
        curr_Q = self.model(states)
        curr_Q = curr_Q.gather(1, torch.argmax(actions, dim=1).unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            next_Q = self.model(next_states)
            max_next_Q = next_Q.max(1)[0].unsqueeze(1)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * max_next_Q

        # Calculate TD errors for updating priorities
        td_errors = torch.abs(curr_Q - target_Q).detach().squeeze().numpy()
        
        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors)

        # Calculate loss and update model
        loss = self.criterion(curr_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class PPOtrainer:
    def __init__(self, model, value_net, lr, gamma):
        super().__init__()
        self.model = model
        self.value_net = value_net
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=self.lr)
        self.gamma = gamma
        self.policy_losses = []
        self.value_losses = []
        self.bad_states_buffer = deque(maxlen=1000)
    
    def train_step(self, states, actions, rewards, dones, old_probs, epochs):

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        old_probs = torch.tensor(old_probs, dtype=torch.float).to(device)

        
        # Calculate discounted rewards
        discounted_rewards = torch.zeros_like(rewards)
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = rewards[t] + self.gamma * running_reward
            discounted_rewards[t] = running_reward

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        # Get current probabilities from model
        self.model.train()
        logits = self.model(states)
        probs = F.softmax(logits, dim=1)
        curr_probs = probs.gather(1, torch.argmax(actions, dim=1).unsqueeze(1)).squeeze(1)

        # PPO ratio
        ratio = curr_probs / (old_probs + 1e-8)
        # advantage calculation
        curr_value = self.value_net(states)
        advantage = (discounted_rewards - curr_value.detach())
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # PPO clipped loss
        eps_clip = 0.2
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        if epochs == 9:
            self.policy_losses.append(policy_loss.item())

        value_loss = (discounted_rewards - curr_value).pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        if epochs == 9:
            self.value_losses.append(value_loss.item())


class board_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name='board_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='board_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

class CNNQNet(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Calculate size after convolutions and pooling
        # For a 20x20 input
        conv_out_size = 16 * 10 * 10  # No size reduction due to padding
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)

        self.to(device)
    def forward(self, x):
        # Convolutional layers with Leaky ReLU and batch norm
        x = F.leaky_relu((self.bn1(self.conv1(x))))
        x = F.leaky_relu((self.bn2(self.conv2(x))))
        x = F.leaky_relu((self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.leaky_relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        
        return x

    def save(self, file_name='model_cnn.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model_cnn.pth'):
        model_folder_path = './model'

        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))


class FCQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, file_name='model_fc.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model_fc.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

