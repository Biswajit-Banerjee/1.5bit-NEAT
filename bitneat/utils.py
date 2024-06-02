# virtual display for later video recording
import gymnasium as gym
import numpy as np

from tqdm.auto import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import getpass
from datetime import datetime as dt
    

def log(data, init=False, path="log.csv"):
    if init:
        log_file = open(path, 'w')
        # f.write(",".join(["Timestamp", "Generation", "Best fitness", "Avg fitness"]) + '\n')
    else:
        data.insert(0, str(dt.now()))
        log_file = open(path, 'a')
    
    log_file.write(",".join(data) + '\n')
    log_file.close()

def read_genome(filename):
    with open(filename, 'rb') as f:
        genome = pickle.load(f)

    return genome

def write_genome(genome, filename):
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)

def laststate(model, env, frame_count=1000, gen=''):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off') 
    fig.patch.set_visible(False)
    plt.tight_layout()
    
    obs, _ = env.reset()
    total_reward = 0
    
    for f in tqdm(range(frame_count)):
        action = model.activate(obs)
        action = np.squeeze(action)

        obs, reward, done, truncated, info = env.step(action)
        rendered_frame = env.render()

        if (type(rendered_frame) == type(None) or done or truncated):
            break
        total_reward += reward
    
    reward_text = ax.text(
        0.43, 0.67, 
        f'Generation: {gen}\nFitness: {total_reward:.2f}\nSteps: {f}', 
        transform=ax.transAxes, fontsize=26, color='black',
        bbox=dict(boxstyle='round,pad=.2', fc='gray', alpha=0.3)
    )

    ax.imshow(rendered_frame);
        
def animate_generations(models_list, output_path='', frame_count=1000, env=None):
    start = dt.now()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off') 
    fig.patch.set_visible(False)
    plt.tight_layout()
    
    frames = []
    if type(env) == type(None):
        env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")

    for gen, model in tqdm(models_list):
        
        obs, _ = env.reset()
        total_reward = 0
        for f in range(frame_count):
            action = model.activate(obs)
            action = np.squeeze(action)
    
            obs, reward, done, truncated, info = env.step(action)
            rendered_frame = env.render()
    
            if (type(rendered_frame) == type(None) or done or truncated):
                break
            total_reward += reward
            reward_text = ax.text(
                0.02, 0.75, 
                f'Gen: {gen}\nReward: {reward:.2f}\nTotal: {total_reward:.2f}\nSteps: {f}', 
                transform=ax.transAxes, fontsize=12, color='black',
                bbox=dict(boxstyle='round,pad=.2', fc='gray', alpha=0.3)
            )
    
            img = ax.imshow(rendered_frame)
            frames.append([img, reward_text])

    # Save the animation as an MP4 video
    if output_path.endswith('mp4'):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist=getpass.getuser(),time=str(dt.now())), bitrate=1800)
        
    elif output_path.endswith('gif'):
        # Save the animation as a GIF
        writer = "pillow"
    else:
        raise Exception("Only .gif or .mp4 formates are supported!")
    
    anim = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    anim.save(output_path, writer=writer)
    
    print(f"Animation saved at: {output_path}\nNumber of Frames: {len(frames)}\nTime taken: {dt.now() - start}")
        