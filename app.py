
import io
import sqlite3
from pathlib import Path
from datetime import date

import pandas as pd
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
DB_PATH = Path(__file__).resolve().parent / "bird_journal.db"


# Database functions
def init_journal_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bird_name TEXT NOT NULL,
                sighting_date TEXT NOT NULL,
                location TEXT,
                notes TEXT,
                name_corrected INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(sightings)"
            ).fetchall()
        }

        if "name_corrected" not in columns:
            conn.execute(
                """
                ALTER TABLE sightings
                ADD COLUMN name_corrected INTEGER NOT NULL DEFAULT 0
                """
            )

        conn.commit()


def get_sightings():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, bird_name, sighting_date, location, notes,
                   name_corrected
            FROM sightings
            ORDER BY sighting_date DESC, id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def save_sighting(
    bird_name,
    sighting_date,
    location,
    notes,
    name_corrected,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sightings
                (bird_name, sighting_date, location, notes, name_corrected)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                bird_name.strip(),
                sighting_date.isoformat(),
                location.strip(),
                notes.strip(),
                int(name_corrected),
            ),
        )
        conn.commit()


# Load the open-weight AI model once
@st.cache_resource(show_spinner="Loading the open-weight AI model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        _attn_implementation="eager",
    )

    model.to(device)
    model.eval()

    return processor, model, device


# Bird identification
def identify_bird(image):
    processor, model, device = load_model()

    bird_prompt = """
You are Bird Buddy AI, a careful bird identification assistant.

Identify the bird actually visible in the supplied image.
Do not assume a species from previous answers.

EXAMINE THE IMAGE:
- Determine whether a bird is clearly visible.
- Examine body size and shape, bill shape and length, head, crest,
  wings, tail, legs, and visible feather patterns.
- Describe only features that are actually visible.
- Do not invent colors, markings, or body parts.
- Being in flight does not by itself identify a species.

IMPORTANT COMPARISON:
- Hummingbirds usually have very small bodies and long, slender bills.
- Northern Cardinals have thicker, cone-shaped bills and prominent crests.
- Adult male Rufous Hummingbirds often have rusty-orange coloring.
  Females and immature birds can be greener.
- Do not identify a bird as a Northern Cardinal just because it is red
  or orange. Compare physical structure first.

ANSWER RULES:
- Choose one best-supported species if the evidence allows.
- Do not list the chosen species as its own alternative.
- Never repeat the same paragraph or section.
- Give at most two genuinely different plausible alternatives.
- If the image is unclear, explain the uncertainty.
- Confidence is a subjective estimate, not a measured probability.
- Keep the answer under 120 words.

Use exactly these four sections:

Most likely bird:
One species name, or "Uncertain bird identification."

Visible evidence:
Two or three short observations based only on the image.

Reasoning:
Briefly explain how the observations support the proposed identification.
Mention any important mismatch.

Alternative species and confidence:
Give up to two different plausible alternatives, or say "No reliable
alternative." State Low, Medium, or High confidence and briefly explain why.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": bird_prompt},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]

    answer = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    return answer


# Initialize the journal
init_journal_db()


# App heading
st.title("🐦 Bird Buddy AI")
st.caption("Your open-AI birdwatching companion")

st.write(
    "Upload a bird photograph, explore an AI-generated identification, "
    "and record the wildlife you discover."
)

st.info(
    "When running locally, the open-weight model performs inference "
    "on your computer after downloading. On a hosted deployment, "
    "inference runs on the hosting machine."
)


# Bird identification section
st.header("🔍 Identify a Bird")

uploaded_file = st.file_uploader(
    "Upload a bird photograph",
    type=["jpg", "jpeg", "png", "webp"],
    key="bird_photo_uploader",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((768, 768))

        st.image(
            image,
            caption="Your uploaded photograph",
            use_container_width=True,
        )

        if st.button(
            "Identify this bird",
            type="primary",
            key="identify_bird",
        ):
            with st.spinner("Examining the photograph..."):
                try:
                    st.session_state["latest_bird_answer"] = identify_bird(
                        image
                    )
                except Exception as error:
                    st.error("The model could not process this photograph.")
                    st.exception(error)

    except Exception as error:
        st.error("This image could not be opened. Please try another.")
        st.exception(error)


# Display identification
if st.session_state.get("latest_bird_answer"):
    st.subheader("🪶 AI Identification")
    st.markdown(st.session_state["latest_bird_answer"])

    st.caption(
        "AI identifications may be incorrect. Verify uncertain species "
        "with a reliable bird field guide."
    )


# Nature challenge
st.divider()
st.header("🌿 Touch Grass Challenge")

completed_challenge = st.checkbox(
    "I went outside and observed nature today.",
    key="nature_challenge",
)

if completed_challenge:
    st.success("Great work! Every outdoor observation counts. 🌱")


# Bird journal
st.divider()
st.header("📓 My Bird Journal")

with st.form("bird_journal_form"):
    bird_name = st.text_input(
        "Bird name",
        placeholder="Example: Rufous Hummingbird",
    )

    sighting_date = st.date_input(
        "Date observed",
        value=date.today(),
    )

    location = st.text_input(
        "Location (optional)",
        placeholder="Example: Garden or local park",
    )

    notes = st.text_area(
        "Notes (optional)",
        placeholder="Describe the bird and its behavior.",
    )

    name_corrected = st.checkbox(
        "I corrected the AI's bird identification"
    )

    save_button = st.form_submit_button(
        "Save sighting",
        type="primary",
    )

    if save_button:
        if not bird_name.strip():
            st.error("Please enter a bird name before saving.")
        else:
            save_sighting(
                bird_name,
                sighting_date,
                location,
                notes,
                name_corrected,
            )
            st.success("The bird sighting was saved successfully!")


# Journal statistics
st.divider()
st.header("📊 Journal Statistics")

sightings = get_sightings()

if sightings:
    journal_df = pd.DataFrame(sightings)

    total_sightings = len(journal_df)
    unique_birds = (
        journal_df["bird_name"].str.strip().str.lower().nunique()
    )
    corrected_names = int(journal_df["name_corrected"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sightings", total_sightings)
    col2.metric("Different bird names", unique_birds)
    col3.metric("Corrections recorded", corrected_names)

    st.subheader("Sightings over time")

    chart_df = journal_df.copy()
    chart_df["observation_date"] = pd.to_datetime(
        chart_df["sighting_date"]
    )

    chart_df = (
        chart_df.groupby("observation_date")
        .size()
        .reset_index(name="Sightings")
        .sort_values("observation_date")
        .set_index("observation_date")
    )

    st.line_chart(chart_df["Sightings"])

    st.subheader("Saved sightings")

    display_df = journal_df[
        [
            "bird_name",
            "sighting_date",
            "location",
            "notes",
            "name_corrected",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "bird_name": "Bird name",
            "sighting_date": "Date observed",
            "location": "Location",
            "notes": "Notes",
            "name_corrected": "Name corrected",
        }
    )

    display_df["Name corrected"] = display_df["Name corrected"].map(
        {0: "No", 1: "Yes"}
    )

    st.dataframe(display_df, use_container_width=True)

    csv_buffer = io.StringIO()
    display_df.to_csv(csv_buffer, index=False)

    st.download_button(
        "⬇️ Download my bird journal (CSV)",
        data=csv_buffer.getvalue(),
        file_name="my_bird_journal.csv",
        mime="text/csv",
        key="download_bird_journal",
    )

else:
    st.info(
        "Your journal is empty. Identify a bird and save your first "
        "sighting to see your statistics here."
    )


# Footer
st.divider()
st.caption("🐦 Bird Buddy AI · Observe nature, learn, and keep exploring.")
