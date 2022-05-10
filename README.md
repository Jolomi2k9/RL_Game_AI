# Game_RL

### Table Of Contents

[Introduction](#introduction)

[Video game environments](#video-game-environments)

[Reinforcement Learning ALgorithms](#reinforcement-learning-algorithms)

[Model building](#model-building)

[Agent Training](#agent-training)

[Testing and evaluation](#testing-and-evaluation)

[Hyperparameter tuning](#hyperparameter-tuning)

[Agent training with tuned hyperparameters](#agent-training-with-tuned-hyperparameters)

[Re-testing and Re-Evaluation](#re-testing-and-re-evaluation)



## Introduction

The idea behind this project is to implement and test different reinforcement learning algorithms in different video game environment types.
In this project we will be focusing on [VizDoom](http://vizdoom.cs.put.edu.pl/) and Unity game engine environments.

Examples of learning environment used in this project are as follows:


<table>
      <tr>
<td><img src="https://github.com/Jolomi2k9/RL_Game_AI/blob/main/images/GitGifBasic.gif" width="480" height="320"></td>
<td><img src="https://github.com/Jolomi2k9/RL_Game_AI/blob/main/images/Gif_doom2.gif" width="480" height="320"></td>
   </tr>
   <tr>
</table>


The project structure is as follows
 ![alt text](https://github.com/Jolomi2k9/RL_Game_AI/blob/main/images/Software%20Structure.png "Project Structure")


We create interfaces that can relay information from the learning environment to the gym using wrappers, the reinforcement learning algorithms can then interact with the game using openAI gym 

## Video game environments

The unity environment used is the [Unity ML-agents Toolkit example environement](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) which are built in the Unity game engine and provided by Unity technologies as a test environment for RL and also as a template to build new enviroments.

However,these environments are built primarily for use with Unity developed algorithms and tools, Unity does include a wrapper for use in openai gym, but these are not specific to ML-agents and we had to write an appropriate interface for our purpose
and with the algorithms we employed.


```python
class UnityGymBasic(Env):
    def __init__(self, render=False):
        super(UnityGymBasic, self).__init__()
        
        #Used for modifying the unity environment
        channel = EngineConfigurationChannel()
        
        #Our Unity script is written in C# while most RL algorithms are in python 
        #We will use the UnityEnvironment wrapper to be able to communicate with the
        #unity environment using python.       
        #We will also decide if we want to render game environment while training. ,side_channels=[channel]
        if render == False:
            env = UnityEnvironment(env_path, worker_id = 0,side_channels=[channel], no_graphics=True)
        else:
            env = UnityEnvironment(env_path, worker_id = 0,side_channels=[channel])       
        
        
        #We change the time scale of the game using a unity environment side channel
        #This enables us to speed up the learning process but the physics in the game may perform unpredictably
        channel.set_configuration_parameters(time_scale = 4.0)
        #Wrapping the python unity environment so that we can use it in openai gym
        env = UnityToGymWrapper(env, allow_multiple_obs=True)
        
        
        
        #We define the action space and size as well as the observation space
        #this allows our RL algorithms to effectivly communicate with the environment
        self.env = env
        self.action_space = self.env.action_space
        self.action_size = self.env.action_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,64), dtype=np.uint8)       

    #reset the environment 
    def reset(self):       
        return self.env.reset()
    #This moves the game foward and returns the game state s, reward r
    #done d(to indicate if the game is done) and information info from the game such as player lives
    def step(self, action):
        s, r, d, info = self.env.step(action)
        return s, float(r), d, info
    #Closes the environment
    def close(self):
        #self.env.close()
        pass
    #render the environment
    def render(self, mode="human"):
        self.env.render()
        #pass
```

This enables our custom openai algorithms to interact and communicate effectively with the Unity environment, we configure the game to send the agent observations and rewards which it will use to learning.
We also defined the action space, this is how the agent will take action in the game environment.

The interface we built for Vizdoom is based on [this](https://github.com/nicknochnack/DoomReinforcementLearning/blob/main/VizDoom-Basic-Tutorial.ipynb) implementation with modifications to suit our project.

We clone the game from github 
```python
cd github & git clone https://github.com/mwydmuch/ViZDoom.git
``` 

The way our agent interacts with the vizdoom environment is similar to the unity environment with observations and rewards being sent to the agent from the environment and the agent taking action in the environment.

```python
#Define our vizdoom environment class
class VizDoomGym(Env):
    #Initialize our environment
    def __init__(self, render=False):
        #inherit from env base class
        super().__init__()
        self.game = DoomGame()
        #This allows us to load up our configurations which defines our maps,rewards,buttons etc...
        self.game.load_config('github/ViZDoom/scenarios/basic.cfg')
        
        
        #Determine if to render game window
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        #start up the game
        self.game.init()
        
        #create our observation space.
        #We want the same of the observation space to match the game frame exactly- 
        #This is what is used to establish the parameters for the underlying models.
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        #define our action space
        self.action_space = Discrete(3)
    #take step in environment
    def step(self, action):
        #Define the action to take
        actions = np.identity(3, dtype=np.uint8)
        #this actions will be a matrix defining if the agent go left,
        #right or shoot and also our frame skip parameter
        reward = self.game.make_action(actions[action],4)        
        
        #return numpy zeroes array if nothing is returned
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            #gray scaling the captured image
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0      
        
        
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info
    #render game 
    def render():
        pass    
    def reset(self):
        self.game.new_episode()        
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    #grayscale and resize the image
    def grayscale(self, observation):
        #take the observation, grab the color channel and move it to the end
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    #close the game
    def close(self):
        #self.game.close()
        pass
```
To load the game we simply call the vizdoom method and pass in if we want the environment rendered or not
```python
env = VizDoomGym(render=True)
``` 



## Callbacks

Callbacks are a way to periodically save the best model from our training at several intervals during training, this enables us to observe the progression of the model.
It also enables us save logs from the trainging which is helpfull in visualizing the agent performance. Again, our implementation will be base on [this](https://github.com/nicknochnack/DoomReinforcementLearning/blob/main/VizDoom-Basic-Tutorial.ipynb) implementation with modification made to suit this project for both our Unity and Vizdoom environment.

```python
class TrainingAndLoggingCallback(BaseCallback):
    
   
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingAndLoggingCallback, self). __init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
``` 

```python
CHECKPOINT_DIR = './train/train_basic3'
LOG_DIR = './train/log_basic3'
``` 

```python
callback = TrainingAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
``` 

## Reinforcement Learning ALgorithms

For the reinforcement learning algorithms we used algorithms from [stablebaselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) with some modifications to the algorithms for more granular control

Below are customized CNN feature extractor we used for getting observations from the Vizdoom environment, this is based on a template from openai stablebaselines 3

```python

class CustomCNN(BaseFeaturesExtractor):
    
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),            
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    

policy_kwargs = dict(
    features_extractor_class=CustomCNN,    
)

```


## Model building

Building a model is strainghtforward, we specify the algorithm we want to use pass in arguments such as the policy network and environment

```python

model = DQN("CnnPolicy", env, buffer_size = 320000,batch_size = 64, policy_kwargs=policy_kwargs,
             verbose=1,optimize_memory_usage = True, learning_rate=0.001)

```
In the code above we also passed in our custom feature extractor using policy_kwargs

## Agent Training

To train the agent we simply call learn on the created model

```python
model.learn(total_timesteps=150000,callback=callback)
```
This will train the model 150000 timesteps and will save the best model every 10000 timesteps based on our callbacks

## Testing and evaluation

To test the model trained by the algorithm we load our best model and run the agent to episodes of the game

```python
model = PPO.load('<Directory of best model>')
```

```python
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.20)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)
```
Evaluation is done using stablebaselines3 evaluate_policy

## Hyperparameter tuning

This project makes use of Optuna for search for the best hyperparameters, our implementation is based on [this](https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py) implementation with modifications made to suit the project.

A snippet of the code used, specifically the search parameters are:

```python
def sample_DQN_params(trial: optuna.Trial) -> Dict[str, Any]:
    
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    buffer_size = trial.suggest_categorical("buffer_size ",[int(1e3),int(1e4),int(1e5)])    
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64,128])    
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    seed  = trial.suggest_categorical("seed ", [1, 2,3,4,5])
    optimize_memory_usage = True

    # Display true values
    trial.set_user_attr("gamma_", gamma)    

    net_arch = [
        16,32 if net_arch == "tiny" else 32,32,32
    ]

    activation_fn = { "relu": nn.ReLU, "tanh": nn.Tanh,}[activation_fn]

    return {        
        "gamma": gamma,
        "buffer_size": buffer_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "optimize_memory_usage": optimize_memory_usage,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,            
        },
    }
```
the remaing code is in the jupyter notebook, above we list the range of values we want optuna to search including differenct network architechture

## Agent training with tuned hyperparameters

After the hyperparameters has been found , the agent is retrained using those hyperparameters

```python
policy_kwargs = dict(    
    activation_fn = th.nn.Tanh,
    net_arch = [32,32,64],
)
```

```python
model = DQN("CnnPolicy", env, tensorboard_log=LOG_DIR, verbose=1,
           gamma = 0.9978186848304191, max_grad_norm = 2.8095652180149444,
           buffer_size = 100000, learning_rate = 0.00015581771449402266,
           batch_size = 64,seed = 1,optimize_memory_usage = True, policy_kwargs=policy_kwargs)
```


## Re-testing and Re-Evaluation

Finally the agent is retested and evaluated
