# 🐦 Bird Buddy AI

**An open-weight AI birdwatching companion that encourages people to explore nature.**

Bird Buddy AI helps users explore the birds around them by uploading a photograph and receiving an AI-generated identification suggestion with a description of visible features. It also encourages outdoor activity through a simple daily nature challenge.

## ✨ Features

- 📸 Upload bird photographs in JPG, JPEG, or PNG format.
- 🤖 Generate bird identification suggestions using the open-weight SmolVLM model.
- 🌿 Explore visible bird colours, shapes, and markings.
- 🌳 Complete a daily outdoor observation challenge.
- 🔒 Run AI inference locally after downloading the model, without sending the photo to a separate inference API through this app.

## 🧠 AI Model

Bird Buddy AI uses [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct), an open-weight vision-language model hosted on Hugging Face.

The model analyses uploaded images and generates a text response. Its bird identifications are suggestions, not guaranteed scientific classifications.

## 🛠️ Built With

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd bird-ai
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

The first run may take longer because the model needs to download. Processing speed depends on your computer and available hardware.

## 🌳 Why This Project?

Bird Buddy AI connects AI exploration with real-world nature discovery. It encourages users to step outside, observe local wildlife, and learn from their surroundings.

Using open-weight AI makes it possible to explore how vision-language models work and adapt the experience without relying on a separate hosted inference API.

## 🔮 Future Improvements

- Improve bird identification with a specialised bird-classification model.
- Display several possible species when identification is uncertain.
- Add regional bird guides and habitat information.
- Improve inference speed on supported GPUs.
- Keep a personal record of bird sightings.

## ⚠️ Disclaimer

AI-generated bird identifications may be inaccurate. Verify suggestions using a reliable bird field guide or wildlife resource.

## 📄 License

Add a suitable open-source license before publishing the repository. Check the model's license and the licenses of your dependencies before redistributing model files or other assets.
