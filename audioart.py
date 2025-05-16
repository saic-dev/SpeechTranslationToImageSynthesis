import gradio as gr
import torch
import gc
import whisper
from diffusers import StableDiffusionXLPipeline
torch.cuda.empty_cache()
gc.collect()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
try:
    whisper_model = whisper.load_model("base")
    whisper_model = whisper_model.to(device)
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")


try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype= torch_dtype,  
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    print("Stable Diffusion model loaded successfully")
except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
image_style_options = ["Sci-fi", "photorealistic", "low poly", "cinematic", "cartoon", "graffiti", "sketching"]
image_quality_options = ["High resolution", "Crystal clear", "heavy detailed", "Sharp & vibrant", "hyper detailed", "Cinematic masterpiece"]
render_options = ["Octane", "RenderMan", "V-Ray", "Cycles", "Eevee", "Redshift", "Corona", "Unreal Engine", "Unity HDRP"]
angle_options = ["Wide-angle shot", "Full shot", "Top-down view", "Extreme close-up","Telephoto zoom","Portrait framing", "Anamorphic widescreen", "Cinematic composition", "Dynamic action shot"]
lighting_options = ["Soft", "ambient", "ring light", "neon", "Natural", "Harsh", "Dramatic", "Backlit", "Studio"]
background_options = [ "Futuristic city", "Cyberpunk streets", "Alien planet", "Deep space", "Sky with clouds","Indoor futuristic lab", "Grand hall", "Neon-lit alley"]
device_options = ["Go Pro", "Iphone 15", "Canon EOS R5","Nikon Z7", "Sony F950", "Drone"]
emotion_options = ["Happy", "Sad", "Angry", "Mysterious", "Surprised", "Annoyed", "Neutral", "dreamy", "nostalgic"]


def process_audio(audio_file, image_style, image_quality, render, angle, lighting, background, device_type, emotion):
    if audio_file is None:
        return "Please record or upload audio first.", None, "settings"
    
    try:
        print(f"Processing audio file: {audio_file}")
        result = whisper_model.transcribe(audio_file)
        text_prompt = result["text"]
        print(f"Transcribed text: {text_prompt}")

        full_prompt = f"A stunning {image_style}, {image_quality} shot of {text_prompt} captured in {device_type} using {angle} and rendered by {render}, illuminated by {lighting} light, with {emotion} emotions in a {background} background setting."
        neg_prompt = "ugly, blurry, poor quality, deformed structure, very bad lighting, bad colouring, noise"
        print(f"Generating image with prompt: {full_prompt}")

        output = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            height=512,
            width=512,
            num_inference_steps=20
        )

        image = output.images[0]
        print("Image generated successfully")
        return text_prompt, image,"results"

    except Exception as e:
        print(f"Error in process_audio: {e}")
        return f"Error processing: {str(e)}", None,"settings"

with gr.Blocks(css="""
    body,.gradio-container {
        background: url('https://i.pinimg.com/originals/d6/e1/27/d6e12796914cde798323225515bd7868.gif') no-repeat center center fixed;
        background-size: cover;
    }
    .input-page { background: url('https://i.pinimg.com/originals/d6/e1/27/d6e12796914cde798323225515bd7868.gif'); background-size: cover; padding: 20px; }
    .settings-page { background: url('https://source.unsplash.com/random/1920x1080/?design'); background-size: cover; padding: 20px; }
}
""") as demo:
    gr.Markdown("# 🎨 Audio2Art")
    
    page_state = gr.State("input")
    
    with gr.Column(visible=True, elem_classes=["input-page"]) as input_page:
        audio_input = gr.Audio(type="filepath", label="🎤 Record Your Voice or Upload Audio")
        next_button = gr.Button("Next →")
    
    with gr.Column(visible=False, elem_classes=["settings-page"]) as settings_page:
        back_to_input = gr.Button("← Back to Input")
        style = gr.Dropdown(choices=image_style_options, label="🎨 Image Style", value="Sci-fi")
        quality = gr.Dropdown(choices=image_quality_options, label="🖼 Quality", value="High resolution")
        render = gr.Dropdown(choices=render_options, label="🖥 Render Engine")
        angle = gr.Dropdown(choices=angle_options, label="📷 Camera Angle")
        lighting = gr.Dropdown(choices=lighting_options, label="💡 Lighting", value="Soft")
        background = gr.Dropdown(choices=background_options, label="🌆 Background")
        device_type = gr.Dropdown(choices=device_options, label="📸 Device", value="Go Pro")
        emotion = gr.Dropdown(choices=emotion_options, label="😃 Emotion", value="Happy")
        generate_button = gr.Button("🚀 Generate Image")
    
    with gr.Column(visible=False) as results_page:
        output_text = gr.Textbox(label="📝 Transcribed Text")
        output_image = gr.Image(label="🖼 Generated Image")
        back_to_settings = gr.Button("← Back to Settings")
    
    def update_page(page):
        return (
            gr.update(visible=page == "input"),
            gr.update(visible=page == "settings"),
            gr.update(visible=page == "results"),
        )
    
    next_button.click(lambda: "settings", outputs=page_state).then(update_page, inputs=page_state, outputs=[input_page, settings_page, results_page])
    back_to_input.click(lambda: "input", outputs=page_state).then(update_page, inputs=page_state, outputs=[input_page, settings_page, results_page])
    generate_button.click(process_audio, inputs=[audio_input, style, quality, render, angle, lighting, background, device_type, emotion], outputs=[output_text, output_image, page_state]).then(update_page, inputs=page_state, outputs=[input_page, settings_page, results_page])
    back_to_settings.click(lambda: "settings", outputs=page_state).then(update_page, inputs=page_state, outputs=[input_page, settings_page, results_page])

demo.launch(share=True, debug=True)