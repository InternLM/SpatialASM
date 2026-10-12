from openai import OpenAI
import base64
import sys
import os
import json
import re
import matplotlib.pyplot as plt
import ast
import requests
import sys

# Give your answer-only spatial-VQA sample (An example shown below)

# Question Index (You can set it to any integer)
orig_index = 1
# Paths of Input images
all_images = ["HoliSpatial-2M/Problem1_1.png", "HoliSpatial-2M/Problem1_2.png"]
# Input Question
question = "<image>\n<image>\nIn image B, the object described as \"rectangular window with white frame, featuring a decorative red border outline, set within a tiled wall, allowing bright light to enter\" is the window.\nIn image B, the object described as \"Small potted plant with green leaves, placed on white sink, in terracotta pot with visible soil and stems\" is the plant.\nThe bathroom area is in Image B.\nYou are positioned at window and face plant.\nIn which direction is the bathroom area relative to you?\nA. Left\nB. Right\nC. Front-Left\nD. Front-Right"
# Ground-truth answer
gt = "B"



argv = sys.argv[1:]
IP = argv[0]
save_folder = "SpatialASM-scaffolding-imgs"
os.makedirs(save_folder, exist_ok=True)

EDIT_SERVER_URL = f"http://{IP}:8000/edit"
_session = requests.Session()


api_keys = ""
client_reasoning = OpenAI(
    api_key="EMPTY",  
    base_url=f'http://{IP}:23333/v1'
)
client_captioning = OpenAI(
    api_key="EMPTY",  
    base_url=f'http://{IP}:23332/v1'
)

model_select_reasoning = "vllm--qwen-3.6-35BA3B"
model_select_captioning = "vllm--qwen-3.6-27B"


edit_sys_prompt = "Now I stand at the marker '1' in the first image. "

SYS_PROMPT_REASONING = "First think and then give the final answer. Put the option letter within <answer> and </answer>."

SYS_PROMPT_MARKING = (
"I gave you several images and a spatial reasoning question. You can now use an image-editing model to generate images of the scene from new perspectives based on the input images. You do not need to answer the spatial reasoning question I gave you, but you need to reason based on the question and give commands to the image-editing model to provide more useful information for answering this question.\n"
"For each command, you need to first select a source image from the input images. Then you need to select an area on the image and assign a facing direction, so that the image-editing model will generate the scene viewed from this area and specific facing direction.\n"
"Hints: To see the right side of the image, the area should be on the left side (x<=25) and 'Direction' should be 'Right'; To see the back side of the image, the area should be near the middle (35<=x<=65) on the x-axis and 'Direction' should be 'Backwards';  To see the left side of the image, the area should be on the right side (x>=75) and 'Direction' should be 'Left'.\n"
"Provide your answer in the following format: a list containing Dict(s):\n"
"[{'source_img': '<e.g. 1>', 'pixel_x': '<e.g. 22>', 'pixel_y': '<e.g. 63>', 'Direction': '<e.g. Right>'}, ...]\n"
"'source_img'  is the image number (the first image is numbered 1, and so on). 'pixel_x' and 'pixel_y' represent the assigned area, and we assume the width and height of the images are both 100 for simplicity. Direction is chosen from 'Left', Backwards', and 'Right'.\n"
"You can output an empty list if you believe there is no need for providing a new perspective.\n"
"First think and then give the final answer. The final answer should contain a short summary of your reasoning process (no more than 200 words), which only focus on how to derive the command without giving the answer of the original question, and the final list. The summary should be put within <summary> and </summary>, and then output the list within <answer> and </answer>.\n"
)

caption_sys_prompt = "Describe the 3D location of the objects in the image. The location include left, right, top, bottom, background, foreground, and so on. You can describe the location of each object separately and the relative location between objects as well (e.g. Object A is to the right of Object B). The answer should be no more than 200 words."

edit_seed = 42

def get_response_reason(image_paths, prompts, sys_prompt):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    content = []

    for img_path in image_paths:
        base64_imag = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_imag}"
            }
        })
    for prompt in prompts:
        content.append({
            "type": "text",
            "text": prompt
        })
    content.append({
        "type": "text",
        "text": sys_prompt
    })

    response = client_reasoning.chat.completions.create(
        model=model_select_reasoning,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.7,
        max_tokens=8192,
        presence_penalty=1.5, 
        extra_body={"top_p": 0.80, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    )
    return response.choices[0].message.content


def get_response_caption(image_paths, prompts, sys_prompt):
    # 内部函数：编码图像为Base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    content = []

    for img_path in image_paths:
        base64_imag = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_imag}"
            }
        })
    for prompt in prompts:
        content.append({
            "type": "text",
            "text": prompt
        })
    content.append({
        "type": "text",
        "text": sys_prompt
    })

    response = client_captioning.chat.completions.create(
        model=model_select_captioning,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0,
        max_tokens=8192,
        presence_penalty=1.5, 
        extra_body={"top_p": 0.80, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    )
    return response.choices[0].message.content


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

def clamp(x):
    return max(5, min(x, 95))

def extract(response):
    pattern = r"<answer>([^}]*)</answer>"
    # 使用 re.findall 查找所有匹配的内容
    matches = re.findall(pattern, response)
    if len(matches)==0:
        return response
    return matches[-1].strip()

def convert_str_list(s):
    s = extract(s)
    idx = s.find('[')
    idx_back = s.rfind(']')
    s = s[idx:idx_back+1]
    # print(s)
    s_list = ast.literal_eval(s)
    for d in s_list:
        for k, v in d.items():
            if isinstance(v, str) and v.isdigit():
                d[k] = int(v)
    return s_list

def judge(pred, gt):
    pred = extract(pred)
    if pred.lower()==gt.lower():
        return 1
    if pred.startswith(f"{gt}."):
        return 1
    else:
        return 0


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

def generate_metadata(all_images, question, gt):   
    marking_response = ""
    res1 = ""
    res2 = ""
    judge1 = None
    judge2 = None
    edit_row = None
    try:
        print(all_images, flush=True)
        print(question, flush=True)
        print("gt:", gt)
        res1 = get_response_reason(all_images, [question], SYS_PROMPT_REASONING)
        print(res1, flush=True)
        judge1 = judge(res1, gt)
        print(judge1, flush=True)

        marking_response = get_response_reason(all_images, [question], SYS_PROMPT_MARKING)
        print(marking_response, flush=True)
        edit_row = convert_str_list(marking_response)
        if len(edit_row)==0:
            metadata = {"id": orig_index, "ans_list": edit_row, "marking_response": marking_response, "response1": res1, "response2": res1, "gt": gt, "hit1": judge1, "hit2": judge1}
            return metadata

        cnt = 0
        directions = []
        for edit_command in edit_row:
            src_id = edit_command['source_img'] - 1
            drafted_path = f"{save_folder}/marked_{orig_index}_{cnt}.jpg"
            x = edit_command['pixel_x']
            y = edit_command['pixel_y']
            direction = edit_command['Direction'].lower()
            directions.append(direction)
            
            x = clamp(x)
            y = clamp(y)

            edit_command['Direction'] = direction

            plot(all_images[src_id], drafted_path, x, y)
            
            input_image_paths = [drafted_path]
            for i, image in enumerate(all_images):
                if i==src_id:
                    continue
                input_image_paths.append(image)
            
            prompt = f"I turn {direction}. "
            input_prompt = edit_sys_prompt + prompt + " What do I see?"
            call_edit_model(input_image_paths, input_prompt, save_folder, orig_index, cnt, seed=edit_seed, steps=25, cfg=10.0)
            cnt += 1
        
        original_image_tokens = ""
        for i in range(len(all_images)):
            original_image_tokens += f"<image{i+1}>"
        
        edit_prompts = ""
        original_len = len(all_images)
        for j in range(cnt):
            edit_prompts += f"Standing at the place marked as '1' in <image{original_len+2*j+1}>, and turning {directions[j]}, you will see <image{original_len+2*j+2}>.\n"
            all_images.append(f"{save_folder}/marked_{orig_index}_{j}.jpg")
            all_images.append(f"{save_folder}/edited_{orig_index}_{j}.png")
        

        aux_prompt = (
        f"I gave you several images{original_image_tokens} and a spatial reasoning problem only based on these image(s).\n"
        "To help you with the problem, I provide you with several new images of the scene seen in new perspectives.\n"
        f"{edit_prompts}"
        "Use the auxiliary images to help you solve the problem."
        )
        res2 = get_response_reason(all_images, [question + "\n" + aux_prompt], SYS_PROMPT_REASONING)
        print(res2, flush=True)
        judge2 = judge(res2, gt)

        metadata = {"id": orig_index, "ans_list": edit_row, "marking_response": marking_response, "response1": res1, "response2": res2, "gt": gt, "hit1": judge1, "hit2": judge2}
        return metadata
    
    except Exception as e:
        print(e, flush=True)
        return None

def process(initial_len, caption):
    def extract_summary(response):
        pattern = r"<summary>([^}]*)</summary>"
        # 使用 re.findall 查找所有匹配的内容
        matches = re.findall(pattern, response)
        if len(matches)==0:
            return response
        return matches[-1].strip()
    
    assert isinstance(caption['ans_list'], list)
    if len(caption['ans_list'])==0:
        # assert caption['hit1']==1
        return caption['response1']
    else:
        # assert caption['hit2']==1
        marking_cot = extract_summary(caption['marking_response'])
        marking_result = "\nTherefore, I need to imagine the views I will see in novel perspectives. Assume that all input images have the size (100, 100)."
        for j, edit_command in enumerate(caption['ans_list']):
            marking_result +=  (
                f" Assume I add a marker '1' at the position ({edit_command['pixel_x']}, {edit_command['pixel_y']}) in <image{edit_command['source_img']}>, I will get <image{initial_len+2*j+1}>.\n"
                f"At the marker '1' in <image{initial_len+2*j+1}>, if I turn {edit_command['Direction']}. I will see <image{initial_len+2*j+2}>:\n"
                f"{caption['caption'][j].strip()}\n"
                )
        marking_result += "Now I will rethink the problems based on input images and the imagined images.\n"
        return marking_cot + marking_result + caption['response2']
    
def main():
    # Get Metadata dictionary (correponds to Section 3.2 in the paper):
    # "ans_list" is v_k; "marking_response" is r_{plan}; "response1" is \hat{a}_0; "response2" is r_{plan} + \hat{a}; "hit1" and "hit2" describe whether the two reponses are correct.
    initial_len = len(all_images)
    ASM_trajectory_meta = generate_metadata(all_images, question, gt)

    # Caption Substitution: I_{edit, k} -> C_{edit, k}
    captions = []
    for j in range(len(ASM_trajectory_meta['ans_list'])):
        path_name = f"{save_folder}/edited_{orig_index}_{j}.png"
        assert os.path.exists(path_name)
        res = get_response_caption([path_name], [], caption_sys_prompt)
        print(res, flush=True)
        captions.append(res)
    ASM_trajectory_meta['caption'] = captions

    # Final ASM reasoning Trajectory (in ShareGPT format)
    conversations = []
    conversations.append({"from": "human", "value": question + SYS_PROMPT_REASONING})
    answer = process(initial_len, ASM_trajectory_meta)
    conversations.append({"from": "gpt", "value": answer})
    print(f"ASM Trajectory:{conversations}")

if __name__ == "__main__":
    main()