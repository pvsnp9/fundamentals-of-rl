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
def save_value_function_as_image(V: np.ndarray, iteration: int, title: str = "Value Function"):
    plt.figure(figsize=(6, 6))
    sns.heatmap(V, annot=True, cmap="coolwarm", cbar=True)
    plt.title(f"{title} (Iteration {iteration})")
    plt.xlabel("Column")
    plt.ylabel("Row")
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
