# --- Imports ---
import gradio as gr
import json, os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.prompts import PromptTemplate

# --- Log Config ---
LOG_PATH = "pixelalchemy_log.json"

def log_prompt(prompt, image_path):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "image_path": image_path
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

#Log Viewing Functions-To see the history

def view_logs(n=5):
    logs = []
    try:
        with open(LOG_PATH, "r") as f:
            for line in f:
                logs.append(json.loads(line))
    except FileNotFoundError:
        return "No logs found yet."
    
    recent = logs[-n:]
    return "\n".join([f"{e['timestamp']} | {e['prompt']} | {e['image_path']}" for e in recent])

# --- Model loading ---
#Load Stable Diffusion model

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)


#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16).to("cuda")

#Load BLIP captioning model

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

#Lagnchain prompt template

template = PromptTemplate(
    input_variables=["concept"],
    template="Generate a creative image of {concept} "
)


# --- Helper functions ---
def describe_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# --- Logging function ---
def log_prompt(concept, prompt, image_path, flagged=True):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "concept": concept,
        "prompt": prompt,
        "image_path": image_path,
        "flagged": flagged
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- Read logs ---
def read_logs():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return [json.loads(line) for line in f]

# --- Display all logs ---
def display_logs():
    entries = read_logs()
    if not entries:
        return "No logs yet."
    html = "<table border='1' style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th>Timestamp</th><th>Prompt</th><th>Image</th></tr>"
    for e in entries[::-1]:
        html += f"<tr><td>{e['timestamp']}</td><td>{e['prompt']}</td>"
        html += f"<td><img src='{e['image_path']}' width='200'></td></tr>"
    html += "</table>"
    return html

# --- Display flagged logs only ---
def display_flagged():
    entries = [e for e in read_logs() if e.get("flagged")]
    if not entries:
        return "No flagged entries yet."
    html = "<table border='1' style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th>Timestamp</th><th>Prompt</th><th>Image</th></tr>"
    for e in entries[::-1]:
        html += f"<tr><td>{e['timestamp']}</td><td>{e['prompt']}</td>"
        html += f"<td><img src='{e['image_path']}' width='200'></td></tr>"
    html += "</table>"
    return html

# --- Image generation function ---

def generate_image(concept):
    prompt = template.format(concept=concept)
    image = pipe(prompt).images[0]

    image_path = f"output_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(image_path)

    # NEW: generate description
    description = describe_image(image)

    # Log with description
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "image_path": image_path,
        "description": description,
        "flagged": True
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return image, description


# --- Gradio Tabs ---

with gr.Blocks() as demo:
    with gr.Tab("Generate Image"):
      gr.Interface(
      fn=generate_image,
      inputs=gr.Textbox(label="Concept"),
      outputs=[gr.Image(label="Generated Image"), gr.Textbox(label="Description")],
      title="PixelAlchemy with Image Description"
      )
       

    with gr.Tab("Logs"):
        log_output = gr.HTML()
        refresh_btn = gr.Button("Refresh Logs")
        refresh_btn.click(display_logs, outputs=log_output)

    with gr.Tab("Flagged Outputs"):
        flagged_output = gr.HTML()
        refresh_flagged_btn = gr.Button("Refresh Flagged")
        refresh_flagged_btn.click(display_flagged, outputs=flagged_output)

demo.launch()

