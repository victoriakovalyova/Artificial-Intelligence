
import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make("MiniGrid-Empty-8x8-v0")

def preprocess_obs_for_wm(obs_image):
    img = obs_image.astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))

class RSSM(torch.nn.Module):
    # ... (тот же код класса, что и раньше)
    def __init__(self, state_dim=64, rnn_hidden=256, obs_channels=3, obs_size=7, action_dim=7):
        super().__init__()
        self.state_dim = state_dim
        self.rnn_hidden = rnn_hidden
        self.action_dim = action_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(obs_channels, 32, 3, stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 2 * 2, 256)
        )

        self.rnn = torch.nn.GRUCell(state_dim + action_dim, rnn_hidden)
        self.prior_mean = torch.nn.Linear(rnn_hidden, state_dim)
        self.prior_logstd = torch.nn.Linear(rnn_hidden, state_dim)
        self.posterior_mean = torch.nn.Linear(rnn_hidden + 256, state_dim)
        self.posterior_logstd = torch.nn.Linear(rnn_hidden + 256, state_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim + rnn_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64 * 2 * 2),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (64, 2, 2)),
            torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, obs_channels, 3, stride=2, padding=1, output_padding=0),
            torch.nn.Sigmoid()
        )

        self.reward = torch.nn.Sequential(
            torch.nn.Linear(state_dim + rnn_hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, obs_embed, prev_state, prev_action, prev_rnn):
        rnn_input = torch.cat([prev_state, prev_action], dim=-1)
        next_rnn = self.rnn(rnn_input, prev_rnn)

        prior_mean = self.prior_mean(next_rnn)
        prior_std = torch.exp(self.prior_logstd(next_rnn)) + 0.1
        prior_dist = Independent(Normal(prior_mean, prior_std), 1)

        post_input = torch.cat([next_rnn, obs_embed], dim=-1)
        post_mean = self.posterior_mean(post_input)
        post_std = torch.exp(self.posterior_logstd(post_input)) + 0.1
        posterior_dist = Independent(Normal(post_mean, post_std), 1)

        next_state = posterior_dist.rsample()
        dec_input = torch.cat([next_state, next_rnn], dim=-1)
        recon_obs = self.decoder(dec_input)
        pred_reward = self.reward(dec_input)

        return prior_dist, posterior_dist, next_rnn, next_state, recon_obs, pred_reward

    def imagine(self, prev_state, prev_action, prev_rnn):
        rnn_input = torch.cat([prev_state, prev_action], dim=-1)
        next_rnn = self.rnn(rnn_input, prev_rnn)
        mean = self.prior_mean(next_rnn)
        std = torch.exp(self.prior_logstd(next_rnn)) + 0.1
        next_state = mean + std * torch.randn_like(std)
        dec_input = torch.cat([next_state, next_rnn], dim=-1)
        pred_reward = self.reward(dec_input)
        return next_state, next_rnn, pred_reward

def collect_data(num_episodes=100, max_steps=50):
    data = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            data.append((
                preprocess_obs_for_wm(obs['image']),
                F.one_hot(torch.tensor(action), 7).float(),
                torch.tensor(reward).float().unsqueeze(0),
                preprocess_obs_for_wm(next_obs['image'])
            ))
            if done:
                break
            obs = next_obs
    return data

print("Сбор данных (100 эпизодов)...")
data = collect_data(100)
dataset = TensorDataset(
    torch.stack([d[0] for d in data]),
    torch.stack([d[1] for d in data]),
    torch.stack([d[2] for d in data]),
    torch.stack([d[3] for d in data])
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RSSM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Обучение модели мира (30 эпох)...")
model.train()
for epoch in range(30):
    total_loss = 0
    for obs_batch, act_batch, rew_batch, next_obs_batch in loader:
        obs_batch = obs_batch.to(DEVICE)
        act_batch = act_batch.to(DEVICE)
        rew_batch = rew_batch.to(DEVICE)
        next_obs_batch = next_obs_batch.to(DEVICE)
        B = obs_batch.size(0)

        state = torch.zeros(B, 64).to(DEVICE)
        rnn = torch.zeros(B, 256).to(DEVICE)

        obs_embed = model.encoder(obs_batch)
        prior_dist, posterior_dist, next_rnn, next_state, recon_obs, pred_reward = model(
            obs_embed, state, act_batch, rnn
        )

        kl = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        recon_loss = F.mse_loss(recon_obs, next_obs_batch)
        reward_loss = F.mse_loss(pred_reward, rew_batch)
        loss = recon_loss + 0.1 * kl + reward_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "rssm_minigrid.pt")
print("Модель сохранена как rssm_minigrid.pt")