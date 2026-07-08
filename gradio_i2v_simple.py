"""
Simple Gradio Web UI for HunyuanVideo-I2V
Generate videos from images and text prompts with an easy-to-use interface
"""

import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from loguru import logger
import gradio as gr
import random

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


# Global model instance
model_sampler = None


def load_model():
    """Load the HunyuanVideo-I2V model"""
    global model_sampler
    
    if model_sampler is not None:
        return "Model already loaded ✓"
    
    try:
        args = parse_args()
        models_root_path = Path(args.model_base)
        
        if not models_root_path.exists():
            return f"❌ Error: Model path does not exist: {models_root_path}"
        
        logger.info("Loading HunyuanVideo-I2V model...")
        model_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        logger.info("✓ Model loaded successfully!")
        
        return "✓ Model loaded successfully!"
    
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def generate_video_from_image(
    input_image,
    prompt,
    seed,
    stability_mode,
    infer_steps,
    flow_shift,
):
    """
    Generate video from image and prompt
    
    Args:
        input_image: PIL Image or file path
        prompt: Text prompt for video generation
        seed: Random seed (-1 for random)
        stability_mode: Enable stable mode
        infer_steps: Number of inference steps
        flow_shift: Flow shift value
    
    Returns:
        Video path or error message
    """
    
    global model_sampler
    
    if model_sampler is None:
        return None, "❌ Model not loaded. Please load the model first."
    
    try:
        # Handle seed
        if seed == -1:
            seed = None
        
        # Save image to temp file if needed
        if hasattr(input_image, 'save'):
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            input_image.save(temp_file.name)
            image_path = temp_file.name
        else:
            image_path = str(input_image)
        
        logger.info(f"Generating video from image: {image_path}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Seed: {seed}, Stability: {stability_mode}, Steps: {infer_steps}")
        
        # Set flow_shift based on stability mode
        if stability_mode:
            flow_shift = 7.0
        else:
            flow_shift = 17.0
        
        # Generate video
        start_time = time.time()
        
        outputs = model_sampler.predict(
            prompt=prompt if prompt else "High quality video",
            height=720,
            width=1280,
            video_length=129,
            seed=seed,
            negative_prompt="",
            infer_steps=infer_steps,
            guidance_scale=1.0,
            num_videos_per_prompt=1,
            flow_shift=flow_shift,
            batch_size=1,
            embedded_guidance_scale=6.0,
            i2v_mode=True,
            i2v_resolution="720p",
            i2v_image_path=image_path,
            i2v_condition_type="camera",
            i2v_stability=stability_mode,
        )
        
        samples = outputs['samples']
        sample = samples[0].unsqueeze(0)
        
        # Save video
        output_dir = "./gradio_i2v_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        prompt_text = (prompt[:50] if prompt else "video").replace(" ", "_").replace("/", "")
        video_path = f"{output_dir}/{time_flag}_{prompt_text}_seed{outputs['seeds'][0]}.mp4"
        
        save_videos_grid(sample, video_path, fps=24)
        
        gen_time = time.time() - start_time
        logger.info(f"✓ Video saved: {video_path}")
        logger.info(f"Generation time: {gen_time:.1f}s")
        
        status_msg = f"✓ Video generated in {gen_time:.1f}s\nSaved to: {video_path}"
        
        # Clean up temp file
        if hasattr(input_image, 'save'):
            try:
                os.unlink(image_path)
            except:
                pass
        
        return video_path, status_msg
    
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return None, f"❌ Error: {str(e)}"


def create_demo():
    """Create Gradio interface"""
    
    with gr.Blocks(title="HunyuanVideo-I2V: Image-to-Video Generator", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🎬 HunyuanVideo-I2V: Image-to-Video Generator
            
            **Generate professional videos from your images!**
            
            Upload an image and describe what should happen in the video. The AI will animate it for you.
            """
        )
        
        # Load model button
        with gr.Row():
            load_status = gr.Textbox(
                label="Model Status",
                value="⏳ Click 'Load Model' to start",
                interactive=False,
                scale=3
            )
            load_btn = gr.Button("Load Model", scale=1, variant="primary", size="lg")
        
        gr.Markdown("---")
        
        # Input section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    scale=1
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Settings")
                
                prompt = gr.Textbox(
                    label="Describe the video motion/action",
                    placeholder="e.g., 'Person walks forward with a smile'",
                    lines=3
                )
                
                with gr.Row():
                    stability = gr.Checkbox(
                        label="Stable Mode",
                        value=True,
                        info="Generate more stable, subtle motion"
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        info="-1 for random"
                    )
                
                infer_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Inference Steps",
                    info="More = better quality but slower"
                )
                
                flow_shift = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=7.0,
                    step=0.5,
                    label="Flow Shift",
                    info="Controls motion intensity"
                )
        
        # Generate button
        generate_btn = gr.Button(
            "🎬 Generate Video",
            variant="primary",
            size="lg",
            scale=1
        )
        
        gr.Markdown("---")
        
        # Output section
        with gr.Row():
            output_video = gr.Video(
                label="Generated Video",
                scale=2
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                scale=1
            )
        
        gr.Markdown(
            """
            ### 💡 Tips for Best Results:
            - **Keep prompts concise** and specific about the motion
            - **Use "Stable Mode"** for videos that maintain the image well
            - **Uncheck "Stable Mode"** for more dynamic, varied motion
            - **Higher inference steps** = better quality but takes longer
            - **Seed control** for reproducible results
            
            ### ⚙️ Requirements:
            - GPU with 60GB+ VRAM (80GB recommended)
            - First generation takes ~5-10 minutes to initialize
            - Subsequent generations are faster
            """
        )
        
        # Event handlers
        load_btn.click(
            load_model,
            outputs=load_status
        )
        
        generate_btn.click(
            generate_video_from_image,
            inputs=[input_image, prompt, seed, stability, infer_steps, flow_shift],
            outputs=[output_video, status_text]
        )
        
        return demo


if __name__ == "__main__":
    # Set environment variable
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    # Create and launch demo
    demo = create_demo()
    
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8081"))
    
    logger.info(f"Starting Gradio server on {server_name}:{server_port}")
    print(f"\n🌐 Open http://localhost:{server_port} in your browser\n")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        share=False
    )
