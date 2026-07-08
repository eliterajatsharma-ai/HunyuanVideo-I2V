# 🚀 Quick Start: HunyuanVideo-I2V (Image-to-Video)

Generate professional AI videos from your images! Free and open-source.

## 📋 Requirements

- **GPU**: NVIDIA with 60GB+ VRAM (80GB recommended)
- **OS**: Linux
- **CUDA**: 11.8 or 12.4
- **Python**: 3.11+

## 🎯 Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
cd HunyuanVideo-I2V

# 2. Create environment
conda create -n HunyuanVideo-I2V python==3.11.9
conda activate HunyuanVideo-I2V

# 3. Install PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install dependencies
pip install -r requirements.txt
pip install ninja
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# 5. Download models (~40GB)
pip install huggingface_hub[cli]
huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts

# 6. Download text encoders
cd ./ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./text_encoder_i2v
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```

## 🎬 Usage

### Option 1: Web Interface (Easiest - No Command Line!)

```bash
cd HunyuanVideo-I2V
python gradio_i2v_simple.py
```

Then open **http://localhost:8081** in your browser:

1. Click **"Load Model"** button
2. Upload your image
3. Describe what should happen ("Person walks forward", "Woman smiles", etc.)
4. Adjust settings (quality, stability, etc.)
5. Click **"Generate Video"**
6. Wait 5-10 minutes for first generation, 2-3 min for subsequent ones
7. Download your video! 🎉

### Option 2: Command Line

```bash
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "Woman smiles at camera" \
    --i2v-mode \
    --i2v-image-path ./your_image.jpg \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results
```

## 📊 Configuration Guide

| Setting | Purpose | Value |
|---------|---------|-------|
| `--infer-steps` | Video quality | 50 (recommended), higher = better |
| `--flow-shift` | Motion intensity | 7.0 (stable), 17.0 (dynamic) |
| `--i2v-stability` | Keep image faithful | Yes (stable) or No (dynamic) |
| `--seed` | Reproducible results | 0-9999 or -1 (random) |
| `--video-length` | Video frames | 65 (2s) or 129 (5s) |

## 💡 Pro Tips

### ✅ Best Practices
- **Keep prompts short**: "Person walks forward" not "A person is walking very slowly forward down a street..."
- **Be specific about motion**: Include action verbs and direction
- **Use Stable Mode** for realistic, faithful animation (default)
- **Uncheck Stable Mode** for creative, diverse motion
- **Test with low inference steps** (30) first to iterate faster

### ❌ Avoid These
- Very long, detailed prompts (causes unwanted transitions)
- Contradictory descriptions
- Requesting face changes (works better with body motion)
- Unrealistic physics

### 📝 Example Prompts
- "Woman smiles at camera"
- "Man waves hand up"
- "Child runs to the right"
- "Person nods head slowly"
- "Model walks forward confidently"
- "Actor looks surprised"
- "Person blinks eyes"

## 🎥 Output

Videos are saved to:
- **Web UI**: `./gradio_i2v_outputs/`
- **Command line**: `./results/`

Format: MP4, 720p (1280x720), 24 fps, 5 seconds (129 frames)

## 🚀 Speed & Performance

### Generation Time
- **First run**: ~5-10 minutes (model initialization + CUDA compilation)
- **Subsequent runs**: ~2-3 minutes per video on single 80GB GPU
- **With 8 GPUs**: ~30-45 seconds per video (with xDiT)

### Multi-GPU Inference

For faster generation with 8 GPUs:

```bash
ALLOW_RESIZE_FOR_SP=1 torchrun --nproc_per_node=8 \
    sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "Your prompt here" \
    --i2v-mode \
    --i2v-image-path ./image.jpg \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results \
    --ulysses-degree 8 \
    --ring-degree 1
```

## 💰 Cost (If Using GPU Cloud)

- **Google Colab**: FREE (limited GPU, queue)
- **Lambda Labs**: $1.50/hr (RTX 6000 Ada, 48GB)
- **Vast.ai**: $0.50-2/hr (various GPUs)
- **Per video**: ~$0.04-0.33 for 5-minute generation

## ⚠️ Common Issues & Fixes

### CUDA Out of Memory
```bash
# Enable CPU offload (slower but uses less GPU RAM)
--use-cpu-offload

# Or use lower resolution
--i2v-resolution 360p

# Or reduce inference steps
--infer-steps 30
```

### Float Point Exception
```bash
# Make sure CUDA 12.4+ is installed
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```

### Model Download Slow
```bash
# Use mirror (for China)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts
```

### Poor Video Quality
- Increase `--infer-steps` to 70-100
- Use clearer, well-lit input images
- Try different `--seed` values
- Make prompt more specific

## 📚 More Information

- **Official Repo**: https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
- **Research Paper**: https://arxiv.org/abs/2412.03603
- **Project Page**: https://aivideo.hunyuan.tencent.com
- **HuggingFace Models**: https://huggingface.co/tencent/HunyuanVideo-I2V

## 🎉 Ready to Create!

You now have everything to generate professional AI videos from images.

**Next steps:**
1. Follow installation guide above
2. Run `python gradio_i2v_simple.py` or use command line
3. Upload image + write prompt
4. Generate! ✨

**Questions?** Check the [GitHub Issues](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/issues) or [GitHub Discussions](https://github.com/Tencent-Hunyuan/HunyuanVideo)

Happy video creating! 🚀🎬
