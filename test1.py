import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

DIMENSION = 2  # Number of dimensions in the voting space
STD_DEV = 10.0  # Standard deviation for people distribution

class VotingSpaceEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The goal is to move the 'Current Law' as far from the origin as possible
    subject to majority approval.
    """
    def __init__(self, n_dim=DIMENSION, n_people=20, max_steps=50):
        super(VotingSpaceEnv, self).__init__()
        
        self.n_dim = n_dim
        self.n_people = n_people
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: The change (delta) to apply to the current law.
        # We bound the step size to prevent the agent from teleporting wildly,
        # forcing it to learn incremental "salami slicing" tactics.
        self.max_delta = 1.0
        self.action_space = spaces.Box(
            low=-self.max_delta, 
            high=self.max_delta, 
            shape=(n_dim,), 
            dtype=np.float32
        )
        
        # Observation space: 
        # [Current Law Coords (n)] + [People Coords (n * p)]
        # We assume coordinates won't exceed reasonable bounds for this simulation.
        low_obs = np.array([-np.inf] * (n_dim + n_dim * n_people))
        high_obs = np.array([np.inf] * (n_dim + n_dim * n_people))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # State variables
        self.people_coords = None
        self.current_law = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Generate People normally distributed around origin
        # Using a fixed seed in the generator ensures reproducibility if needed
        self.people_coords = self.np_random.normal(loc=0.0, scale=STD_DEV, size=(self.n_people, self.n_dim)).astype(np.float32)
        
        # 2. Reset Law to origin
        self.current_law = np.zeros(self.n_dim, dtype=np.float32)
        
        self.current_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten people coords and concatenate with current law
        flat_people = self.people_coords.flatten()
        return np.concatenate([self.current_law, flat_people])

    def step(self, action):
        # 1. Proposal
        # Action is the delta. New Law = Current Law + Action
        proposal = self.current_law + action
        
        # 2. Voting Process
        # Calculate distances (Utility = negative distance)
        # We want to check: distance(Person, Proposal) < distance(Person, Current)
        
        dist_current = np.linalg.norm(self.people_coords - self.current_law, axis=1)
        dist_proposal = np.linalg.norm(self.people_coords - proposal, axis=1)
        
        # Votes: 1 if proposal is strictly closer, else 0
        votes = (dist_proposal < dist_current).astype(int)
        total_votes = np.sum(votes)
        
        # 3. Outcome
        vote_passed = total_votes > (self.n_people / 2)
        
        if vote_passed:
            self.current_law = proposal
            
        # 4. Calculate Reward
        # Utility for MalAct is distance from origin.
        dist_from_origin = np.linalg.norm(self.current_law)
        
        # We reward the distance itself. 
        # To encourage finding passing votes, we might penalize failed votes,
        # but pure distance reward often works if the horizon is long enough.
        reward = float(dist_from_origin)
        
        # Optional: Heavy penalty if vote failed? 
        # Generally better to just give the (smaller) reward of the old position
        # so the agent learns that failing to move yields lower cumulative reward.
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "vote_passed": vote_passed,
            "votes_for": total_votes,
            "dist_from_origin": dist_from_origin
        }
        
        return self._get_obs(), reward, terminated, truncated, info

# --- Training the Model ---

def train_malicious_actor():
    # Create environment
    # n=2 for easy visualization logic, p=50 people
    env = VotingSpaceEnv(n_dim=DIMENSION, n_people=50, max_steps=50)
    
    # Validate environment
    check_env(env)
    
    # Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Training MalAct model...")
    # Train for 50,000 steps
    model.learn(total_timesteps=50000)
    print("Training complete.")
    
    return model, env

# --- Testing / Visualization ---

def test_model(model, env):
    obs, _ = env.reset()
    done = False
    
    print("\n--- Simulation Start ---")
    print(f"Initial Law Position: {env.current_law}")
    
    step_count = 0
    path = [env.current_law.copy()]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if info['vote_passed']:
            status = "PASSED"
        else:
            status = "REJECTED"
            
        # Only print every few steps to reduce clutter
        if step_count % 5 == 0:
            print(f"Step {step_count}: Proposed delta {np.round(action, 2)} -> Vote {status} ({info['votes_for']} votes). Law Dist: {info['dist_from_origin']:.4f}")
        
        if info['vote_passed']:
            path.append(env.current_law.copy())
            
        step_count += 1
        
    print(f"Final Law Position: {env.current_law}")
    print(f"Final Distance from Origin: {np.linalg.norm(env.current_law):.4f}")
    return np.array(path)

if __name__ == "__main__":
    trained_model, env_instance = train_malicious_actor()
    test_model(trained_model, env_instance)