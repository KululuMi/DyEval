# DyEval

Official implementation of the paper **"Interactive Visual Assessment for Text-to-Image Generation Models"**

DyEval is an interactive evaluation system for Text-to-Image (T2I) generation models that leverages Large Language Models (LLMs) to dynamically generate test cases and identify potential failures in T2I models.

## 🎯 Features

- **LLM-Powered Test Generation**: Automatically generates diverse test topics and inputs using LLMs
- **Multi-Model Support**: Test various T2I models including:
  - Stable Diffusion 1.4/1.5
  - Stable Diffusion 2.1
  - Stable Diffusion XL
  - Stable Diffusion 3
- **Interactive Web Interface**: User-friendly Streamlit-based interface for test execution and annotation
- **MiniInput Refinement**: Automatically refines test inputs when pass rate is low (<75%) to identify minimal failure cases
- **Hierarchical Testing**: Supports depth-based test topic generation with configurable maximum depth
- **Automatic Result Saving**: All test results, annotations, and metadata are automatically saved in JSON format
- **Bilingual Support**: Interface available in both English and Chinese

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key (for LLM-powered test generation)

### Python Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit torch torchvision diffusers openai open-clip-torch numpy pillow safetensors
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/KululuMi/DyEval.git
cd DyEval
```

### 2. Configure Environment

Edit `lmm-demo/pages/0_MiniInput0.py` or `lmm-demo/pages/1_Initialization.py` to set:

- **OpenAI API Key**: Replace `"OPENAI-TOKEN"` with your actual API key
- **Model Paths**: Update model paths in `experiment_data.py` according to your local setup
- **CUDA Device**: Adjust `torch.cuda.set_device()` to your preferred GPU

### 3. Download Models

Ensure you have the T2I models downloaded locally. Update the model paths in `experiment_data.py`:

```python
# Example paths (modify according to your setup)
model_paths = {
    'sd1_5': '/path/to/stable-diffusion-v1-5',
    'sd2_1': '/path/to/stable-diffusion-2-1-base',
    'sdxl': '/path/to/stable-diffusion-xl-base-1.0',
    'sd3': '/path/to/stable-diffusion-3-medium'
}
```

### 4. Run the Application

```bash
cd lmm-demo
bash run.sh
```

Or manually:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
STREAMLIT_SERVER_HEADLESS=true python -m streamlit run home.py --server.fileWatcherType none --server.address=127.0.0.1
```

The application will be available at `http://127.0.0.1:8501`

## 📖 Usage

### Initialization

1. **Enter User Information**:
   - Username
   - Contact information
   - Test theme (e.g., "different style in images", "spatial relationships")

2. **Configure Test Parameters**:
   - **Tested Model**: Select from available models (sd1_5, sd2_1, sdxl, sd3, etc.)
   - **Max Test Depth**: Maximum depth for hierarchical topic generation
   - **Topics per Test**: Number of topics to generate per iteration
   - **Test Input Count**: Number of test inputs per topic
   - **Image Count**: Number of images to generate per test input

### Testing Workflow

1. **Generate Test Topics**: Click "Test Topic Generation" to create test topics using LLM
2. **Select Topic**: Choose a topic from the generated list
3. **Generate Test Inputs**: Click "Test Input Generation" to create test prompts
4. **Generate Images**: Images are automatically generated for the selected input
5. **Annotate Images**: Label each image as "Pass", "Fail", or "Off-topic"
6. **MiniInput (if needed)**: If pass rate < 75%, the system automatically generates minimal inputs to identify failure cases

### Results

All test results are saved in the `save/` directory with the following structure:

```
save/
  └── {username}/
      ├── {username}.json  # User records
      ├── {username}_{theme}_{model}.json  # Test results
      └── {topic}/
          └── {model}/
              └── images/  # Generated images
```

## 📁 Project Structure

```
DyEval/
├── lmm-demo/
│   ├── home.py                 # Streamlit entry point
│   ├── experiment_data.py      # Core experiment data management
│   ├── run.sh                  # Launch script
│   ├── pages/
│   │   ├── 0_MiniInput0.py    # Main testing page with MiniInput
│   │   └── 1_Initialization.py # Basic testing page
│   └── save/                   # Test results directory
├── image_filter.py             # Image filtering utilities
└── README.md
```

## 🔧 Configuration

### Key Parameters

- **`max_test_depth`**: Controls the depth of hierarchical topic generation
- **`topics_per_test`**: Number of topics generated per iteration
- **`test_input_count`**: Number of test inputs per topic
- **`image_count`**: Number of images generated per input
- **Pass Rate Threshold**: 75% (triggers MiniInput refinement)

### Model Configuration

Edit `experiment_data.py` to configure model loading:

```python
def load_t2i_model(model_name='sd1_5', model_path='...', device='cuda'):
    # Model loading logic
    ...
```

## 📊 Output Format

Test results are saved as JSON files containing:

- Test metadata (username, theme, model, timestamps)
- Test topics hierarchy
- Test inputs and generated images
- Image annotations (Pass/Fail/Off-topic)
- Pass rates and statistics
- MiniInput refinement history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 Citation

If you use DyEval in your research, please cite:

```bibtex
@article{mi2024interactive,
  title={Interactive Visual Assessment for Text-to-Image Generation Models},
  author={Mi, Xiaoyue and Tang, Fan and Cao, Juan and Sheng, Qiang and Huang, Ziyao and Li, Peng and Liu, Yang and Lee, Tong-Yee},
  journal={arXiv preprint arXiv:2411.15509},
  year={2024}
}
```

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

[Add acknowledgments if applicable]

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Note**: This is the official implementation of the paper "Bug Finder: LLM-aided Dynamic Testing for Text-to-Image Models". For more details, please refer to the paper.

