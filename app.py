import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Page settings
st.set_page_config(
    page_title="Bird Buddy AI",
    page_icon="🐦",
    layout="centered",
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Load the open-weight AI model once and reuse it
@st.cache_resource(show_spinner="Loading the open-weight AI model...")
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Use the available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        _attn_implementation="eager",
    )

    model = model.to(device)
    model.eval()

    return processor, model, device


# Main interface
st.title("🐦 Bird Buddy AI")
st.subheader("Your open-AI birdwatching companion")

st.write(
    "Step outside, photograph a bird, and use AI to "
    "explore the wildlife around you."
)

st.info(
    "The AI runs locally on your computer after the model "
    "has downloaded. Your photo is not sent to a separate "
    "AI inference API by this code."
)

st.header("📸 Identify a Bird")

uploaded_file = st.file_uploader(
    "Upload a bird photo",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Resize large photos to reduce processing time
        image.thumbnail((384, 384))

        st.image(
            image,
            caption="Your bird photo",
            use_container_width=True,
        )

        if st.button("🔍 Identify this bird", type="primary"):
            try:
                with st.spinner(
                    "The AI is examining your photo. "
                    "The first run may take longer..."
                ):
                    processor, model, device = load_model()

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": (
                                        "You are a careful birdwatching "
                                        "assistant. Examine this image. "
                                        "If a bird is visible, suggest "
                                        "its most likely common name and "
                                        "scientific name if known. "
                                        "Briefly describe visible colours, "
                                        "shape, and markings supporting "
                                        "your suggestion. If uncertain, "
                                        "say so and suggest a broader "
                                        "bird group. Do not invent "
                                        "details or a confidence score. "
                                        "Separate observations from "
                                        "uncertain guesses."
                                    ),
                                },
                            ],
                        }
                    ]

                    prompt = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                    )

                    inputs = processor(
                        text=[prompt],
                        images=[image],
                        return_tensors="pt",
                    )

                    # Run inference on the available device
                    inputs = {
                        key: value.to(device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in inputs.items()
                    }

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=60,
                            do_sample=False,
                        )

                    new_tokens = output[
                        :, inputs["input_ids"].shape[1]:
                    ]

                    answer = processor.batch_decode(
                        new_tokens,
                        skip_special_tokens=True,
                    )[0]

                st.subheader("🌿 AI's Bird Identification")
                st.markdown(answer)

                st.caption(
                    "AI predictions can be incorrect. Compare the "
                    "suggestion with a reliable bird field guide."
                )

            except Exception as error:
                st.error(
                    "Bird identification failed. Check the VS Code "
                    "terminal for the technical error."
                )
                print(f"Bird Buddy AI error: {error}")

    except Exception as error:
        st.error("Could not open this image. Try another JPG or PNG.")
        print(f"Image loading error: {error}")


# Outdoor challenge
st.divider()
st.header("🌳 Today's Nature Challenge")

st.write(
    "Spend 10 minutes outdoors observing birds quietly. "
    "Notice their colours, calls, and behaviour without "
    "disturbing them."
)

st.checkbox("I completed today's nature challenge!")

st.caption(
    "Bird Buddy AI | Open-weight AI, real-world discovery."
)
