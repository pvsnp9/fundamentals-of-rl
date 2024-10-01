import imageio
import os
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import Config 

from IPython.display import display, HTML
import math

# Save the heatmap of the value function at each iteration
def save_value_function_as_image(V: np.ndarray, iteration: int, title: str = "Value Function", policy=None):
    """
    Save images the value function or policy over a grid world.

    Args:
    - values (np.ndarray): The value function (2D array of values for each state).
    - title (str): The title of the plot.
    - policy (np.ndarray): (Optional) The policy corresponding to each state, used to overlay arrows.
    """
    fig, ax = plt.subplots()
    ax.imshow(V, cmap="coolwarm", interpolation="none")
    
    # Add the actual V in the grid cells
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, f'{V[i, j]:.2f}', ha='center', va='center', color='black')
    
    ax.set_title(f"{title}_sweep_{iteration}")
    plt.colorbar(ax.imshow(V, cmap="coolwarm"))
    
    # overlay the policy as arrows
    if policy is not None:
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                if np.sum(policy[i, j]) == 0:  # Skip terminal states
                    continue
                action = np.argmax(policy[i, j])
                if action == 0:  # Up
                    ax.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 1:  # Down
                    ax.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 2:  # Left
                    ax.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 3:  # Right
                    ax.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                    
    if not os.path.exists(Config.temp_image_dir): os.makedirs(Config.temp_image_dir)
    plt.savefig(f"{Config.temp_image_dir}/iteration_{iteration:03d}.png")
    plt.close()


# Create a GIF from saved images
def create_gif_from_images(file_name:str):
    images = []
    filenames = sorted((f"{Config.temp_image_dir}/{file}" for file in os.listdir(Config.temp_image_dir) if file.endswith(".png")))
    for filename in filenames:
        images.append(imageio.imread(filename))
    
    if not os.path.exists(Config.gif_dir): os.makedirs(Config.gif_dir)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_file_name = f"{file_name}.gif"
    output_path = os.path.join(Config.gif_dir, gif_file_name)
    # Create the gif using imageio
    imageio.mimsave(output_path, images, duration=0.5)
    
    # Delete the individual image files after the gif is created
    for filename in filenames:
        os.remove(filename)
    
    print(f"GIF saved as {output_path} and images have been deleted.")


def visualize_values(values, title="Value Function", policy=None):
    """
    Visualizes the value function or policy over a grid world.

    Args:
    - values (np.ndarray): The value function (2D array of values for each state).
    - title (str): The title of the plot.
    - policy (np.ndarray): (Optional) The policy corresponding to each state, used to overlay arrows.
    """
    fig, ax = plt.subplots()
    ax.imshow(values, cmap="coolwarm", interpolation="none")
    
    # Add the actual values in the grid cells
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='black')
    
    ax.set_title(title)
    plt.colorbar(ax.imshow(values, cmap="coolwarm"))
    
    # Optionally overlay the policy as arrows
    if policy is not None:
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                if np.sum(policy[i, j]) == 0:  # Skip terminal states
                    continue
                action = np.argmax(policy[i, j])
                if action == 0:  # Up
                    ax.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 1:  # Down
                    ax.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 2:  # Left
                    ax.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                elif action == 3:  # Right
                    ax.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    plt.show()


def visualize_value_function(V:np.ndarray, title: str="Value function")->None:
    plt.figure(figsize=(6,6))
    sns.heatmap(V,annot=True, cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

def display_images_in_grid(image_paths: list, max_cols: int = 3):
    """
    Display a list of images in a grid in a Jupyter notebook. Each row will have up to `max_cols` images.
    
    Args:
    - image_paths (list): A list of paths to images (e.g., GIFs or PNGs).
    - max_cols (int): Maximum number of columns per row. Defaults to 3.
    """
    # Calculate the number of rows needed
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / max_cols)
    
    # Generate the HTML code to display images in grid
    html_code = "<div style='display: flex; flex-direction: column;'>"
    
    for row in range(num_rows):
        html_code += "<div style='display: flex; justify-content: space-evenly; margin-bottom: 10px;'>"
        
        # Add up to max_cols images in each row
        for col in range(max_cols):
            idx = row * max_cols + col
            if idx < num_images:
                img_path = image_paths[idx]
                html_code += f"""
                <div style="flex: 1; margin: 0 10px;">
                    <img src="{img_path}" alt="Image {idx+1}" style="max-width: 100%; height: auto;">
                </div>
                """
        
        html_code += "</div>"  # Close the row div
    
    html_code += "</div>"  # Close the main container div
    
    # Display the generated HTML in the notebook
    display(HTML(html_code))


# plot state-wise-values 
def plot_statewise_value_function(V, grid_size=(4, 4)):
    value_grid = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            value_grid[i, j] = V[(i, j)]

    plt.imshow(value_grid, cmap="coolwarm", origin="upper")
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            plt.text(j, i, f'{value_grid[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.title("State Value Function")
    plt.colorbar(label="Value")
    plt.show()

