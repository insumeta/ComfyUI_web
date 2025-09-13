import json
import random
import pickle
import numpy as np
import gradio as gr
from PIL import Image

from api import get_prompt_images
from settings import COMFYUI_PATH

STYLE_LIST = [
    {"name": "Cinematic",
     "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"},
    {"name": "3D Model",
     "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"},
    {"name": "Anime",
     "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed"},
    {"name": "Digital Art",
     "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"},
    {"name": "Photographic",
     "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"},
    {"name": "Pixel art",
     "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"},
    {"name": "Fantasy art",
     "prompt": "ethereal fantasy concept art of {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"},
    {"name": "Neonpunk",
     "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"},
]

def get_styled_prompt(style_name: str, base_prompt: str) -> str:
    for style in STYLE_LIST:
        if style["name"].lower() == style_name.lower():
            return style["prompt"].replace("{prompt}", base_prompt)
    raise ValueError(f"Style '{style_name}' not found.")

def save_input_image(image):
    input_image = f"{COMFYUI_PATH}/input/sample_sketch.png"
    pillow_image = Image.fromarray(np.array(image["composite"]))
    pillow_image.save(input_image)

def process(positive_prompt, image, style, seed, guidance, prompt_guidance, style_guidance):
    with open("workflow.json", "r", encoding="utf-8") as f:
        prompt = json.load(f)

    prompt["3"]["inputs"]["seed"] = int(seed)
    prompt["13"]["inputs"]["strength"] = guidance
    prompt["6"]["inputs"]["text"] = get_styled_prompt(style, positive_prompt)
    
    prompt["6"]["inputs"]["prompt_guidance"] = prompt_guidance
    prompt["6"]["inputs"]["style_guidance"] = style_guidance

    save_input_image(image)
    images = get_prompt_images(prompt)
    return images

def random_seed():
    return str(random.randint(0, 2**32 - 1))

# Theme 강조색 초록(이지만 색깔 이상함)
custom_theme = gr.themes.Base(
    primary_hue="green",
    secondary_hue="green",
    font="Segoe UI"
)

# 초기 스케치패드 배경 검정(으로 하려고헀는데 안됨)
black_bg = Image.fromarray(np.zeros((1024, 1024, 4), dtype=np.uint8))

with gr.Blocks(theme=custom_theme, css="""
body { background-color: #0a0a0a; color: #ffffff; font-family: 'Segoe UI'; }
.gr-button { border-radius: 12px; padding: 10px 20px; }
textarea, select { background-color: #111111 !important; color: #ffffff !important; border-radius: 8px; border: 1px solid #333; }
""") as demo:

    with gr.Row():
        sketch = gr.Sketchpad(
            type="pil",
            value=black_bg,
            height=1024,
            width=1024,
            image_mode="RGBA",
            show_label=False,
            show_download_button=True,
            brush=gr.Brush(colors=["#00ff00"], default_size=4),
            layers=False
        )
        with gr.Column():
            prompt = gr.Textbox(
                label="Positive Prompt",
                placeholder="Enter prompt here...",
                interactive=True,
                max_lines=2
            )
            style = gr.Dropdown(
                label="Style",
                choices=[s["name"] for s in STYLE_LIST],
                value="Cinematic"
            )

            with gr.Row():
                seed = gr.Textbox(
                    label="Seed",
                    value=str(random.randint(0, 2**32 - 1))
                )
                dice_btn = gr.Button("🎲", elem_id="dice-button")

            guidance = gr.Slider(
                label="Sketch guidance",
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01
            )
            # 새로 추가된 guidance 슬라이더들(은 작동 안됨)
            prompt_guidance = gr.Slider(
                label="Prompt guidance",
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01
            )
            style_guidance = gr.Slider(
                label="Style guidance",
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01
            )

            submit_btn = gr.Button("Generate")

    output = gr.Gallery(label="Result")

    dice_btn.click(fn=random_seed, inputs=None, outputs=seed)

    submit_btn.click(
        fn=process,
        inputs=[prompt, sketch, style, seed, guidance, prompt_guidance, style_guidance],
        outputs=output
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
