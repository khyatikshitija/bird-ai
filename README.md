# 🐦 Bird Buddy AI

**An open-weight AI birdwatching companion that encourages people to explore nature.**

Bird Buddy AI helps users explore birds by uploading a photograph and receiving an AI-generated identification suggestion with a description of visible features. It also encourages outdoor activity through a simple daily nature challenge.

## ✨ Features

- 📸 Upload bird photographs in JPG, JPEG, or PNG format.
- 🤖 Generate bird identification suggestions using the open-weight SmolVLM model.
- 🌿 Explore visible bird colours, shapes, and markings.
- 🌳 Complete a daily outdoor observation challenge.
- 🔒 Run AI inference locally after downloading the model, without sending photos to a separate inference API through the local app.

## 🧠 AI Model

Bird Buddy AI uses [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct), an open-weight vision-language model hosted on Hugging Face.

The model analyses uploaded images and generates a text response. Bird identifications are suggestions, not guaranteed scientific classifications.

**Privacy note:** When running locally, images are processed on your computer. If using a hosted demo, images are processed by the hosting environment.

## 🛠️ Built With

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow

## 🚀 Run Locally

### 1. Download the repository

Download the ZIP file from the GitHub repository and extract it, or clone the repository using Git if you already have Git installed.

Open the extracted `bird-ai` folder in VS Code.

### 2. Create a virtual environment

In the VS Code terminal, run:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable inside `.venv` directly for the following commands.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Start the application

```powershell
python -m streamlit run app.py
```

The first run may take longer because the model needs to download. Processing speed depends on your computer and available hardware.

## 🌳 Why This Project?

Bird Buddy AI connects AI exploration with real-world nature discovery. It encourages users to step outside, observe local wildlife, and learn from their surroundings.

Using open-weight AI makes it possible to explore how vision-language models work and develop an experience without relying on a separate hosted inference API for local image analysis.

## 🔮 Future Improvements

- Improve bird identification using a specialised bird-classification model.
- Display multiple possible species when identification is uncertain.
- Add regional bird guides and habitat information.
- Improve inference speed on supported GPUs.
- Maintain a personal record of bird sightings.

## ⚠️ Disclaimer

AI-generated bird identifications may be inaccurate. Verify suggestions using a reliable bird field guide or wildlife resource.

## 📄 License

A suitable open-source license will be added after reviewing the project and model licensing requirements.
