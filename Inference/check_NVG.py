import requests
import socket
import os
import matplotlib.pyplot as plt

# Change it to your input images. Make sure only the first image has the marker '1'. (See the function 'plot' in line 49 to overlay the maker)
input_image_paths = ["xxx/marked_img.jpg", "HoliSpatial-2M/Problem1_2.png"]
direction = "left" #Choose from ["left", "right", "backwards", "to the front"]
save_folder = "tmp" # The output image will be saved here.
os.makedirs(save_folder, exist_ok=True)
orig_index = 0
cnt = 0
edit_seed = 42

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except OSError:
        local_ip = socket.gethostbyname(socket.gethostname())

print("Local IP:", local_ip)
EDIT_SERVER_URL = f"http://{local_ip}:8000/edit"

_session = requests.Session()

def call_edit_model(image_path, prompt, save_folder, num, cnt, seed=42, steps=25, cfg=10.0, timeout=600):
    _session.post(EDIT_SERVER_URL, json={
        "input_paths": image_path,
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "cfg_scale": cfg,
        "save_folder": save_folder,
        "num": num,
        "cnt": cnt,
    }, timeout=timeout)
    return

edit_sys_prompt = "Now I stand at the marker '1' in the first image. "

prompt = f"I turn {direction}. "
input_prompt = edit_sys_prompt + prompt + " What do I see?"
call_edit_model(input_image_paths, input_prompt, save_folder, orig_index, cnt, seed=edit_seed, steps=25, cfg=10.0)



# Overlay marker to a raw 2D image.
def plot(input_path, save_path, x, y):
    img = plt.imread(input_path)

    x = int(x / 100 * img.shape[1])
    y = int(y / 100 * img.shape[0])
    # scale = (img.shape[1] / 480 * img.shape[0] / 640)
    fontsize = 10
    # print(scale)
    # print(fontsize)
    # pad = 0.05 * (img.shape[1] / 480) * (img.shape[0] / 640)
    lw = 1.0
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.text(x, y, 1, color='black', fontsize=fontsize, ha='center', va='center',
            bbox=dict(boxstyle=f"circle,pad=0.05", facecolor='white', edgecolor='black', lw=lw))

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()