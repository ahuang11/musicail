import os
import random
import subprocess
import sys
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Any, Dict, Optional, Set
from uuid import uuid4

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, OpenAI, PromptTemplate
from music21 import environment, instrument
from music21.converter import parse
from music21.stream import Part, Score, Stream
from PIL import Image, ImageChops, ImageOps

if sys.platform == "darwin":
    MSCORE_PATH = "/opt/homebrew/bin/mscore"
else:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    MSCORE_PATH = "/usr/bin/mscore"
environment.set("musescoreDirectPNGPath", MSCORE_PATH)


def is_subclass(obj: object) -> bool:
    """Checks if an object is a subclass of `instrument.Instrument`.

    Args:
        obj: The object to check.

    Returns:
        A boolean indicating whether the object is a subclass of `instrument.Instrument`.
    """
    try:
        return issubclass(obj, instrument.Instrument)
    except TypeError:
        return False

TEMPLATE = dedent(
    """
    Here's a challenge for you, my helpful chat assistant.

    Imagine you are a world-class, inspirational composer, {composer}.

    Compose a new, memorable song, with a duration of {minutes} minutes,
    based on this description: "{description}".

    The song should use these instruments: "{instruments}".

    Think carefully, step by step, critiquing and revising the composition
    as you go, basing it off musical theory, ensuring each part
    complements one another.

    When you are ready with a final composition, provide me the
    entire song using tinynotation wrapped in "```",
    with each instrument part on its own line, separated by "---".

    Each part must be formatted with a unique title first,
    instrument name second (do not modify instrument name),
    and music notes third, separated by ":".

    Be sure to add flair by using "r" for rest,
    "trip" for triplet, and "quad" for quadruplet.

    Your composition should be formatted like:
    ```
    Dawn of Vi : Viola : 4/4 c4 trip{{c8 d e}} trip{{f4 g a}} b-1
    ---
    Ya : PipeOrgan : 4/4 E4 r f# g=lastG trip{{b-8 a g}} c
    ```
    """
).strip()

DEFAULT_DESCRIPTION = (
    "Grandiose, suspenseful, captivating, dynamic, non-repetitive, "
    "multi-instrumental with emotional solos, anticipation, "
    "with a grand finale"
)

INSTRUMENT_OPTIONS = [
    var_name
    for var_name in dir(instrument)
    if is_subclass(getattr(instrument, var_name))
    and var_name not in {"Instrument", "Conductor"}
    and "Instrument" not in var_name
]


INSTRUCTIONS = """
    With MusicAIl, you *don't need* to be an expert in music theory
    to compose beautiful songs.

    Start by filling out the sidebar; you can use the OpenAI API or paste a
    composition output from an online LLM like ChatGPT, Bing Chat, or Bard,
    using the prompt template as inspiration. When you're ready, press run.

    Or you can manually add parts and enter musical notes in [tinynotation](
    https://web.mit.edu/music21/doc/usersGuide/usersGuide_16_tinyNotation.html).

    Valid notes can be as simple as `A B r C`, where `r` denotes a rest.
    You can also mix complex symbols with notes like
    `4/4 c4 trip{c8 d e} trip{f4 g a} b-1`.

    Add as many parts as you want--each part will be played simultaneously.
"""


def to_mp3(stream: Stream, key: str) -> None:
    """Converts a stream to MP3.

    Args:
        stream: The stream to convert.
    """
    temp_midi_path = stream.write("midi")
    with NamedTemporaryFile(suffix=".mp3") as temp_mp3_file:
        temp_mp3_path = temp_mp3_file.name
        subprocess.run([MSCORE_PATH, "-o", temp_mp3_path, temp_midi_path])
        st.audio(temp_mp3_path, format="audio/mpeg")


def play_stream_inputs(stream: Stream, key: str):
    """Displays a button that plays or pauses a stream when clicked.

    Args:
        stream: The stream to play.
        key: The key for the button.
    """
    label = "part" if "part" in key else "song"
    if st.button(f"🔊 Listen to this {label}.", key=key, use_container_width=True):
        if len(stream.flat.notes) == 0:
            st.error(f"The {label} is empty! Please first enter some musical notes.")
            return
        with st.spinner():
            to_mp3(stream, key=key)


@st.cache_data
def format_keys(part_id: str) -> Dict[str, str]:
    """Formats a set of keys based on a given part ID.

    Args:
        part_id (str): The ID of the part to format the keys for.

    Returns:
        dict: A dictionary of formatted keys.
    """
    return {
        "musical_notes": f"musical_notes_{part_id}",
        "instrument_name": f"instrument_name_{part_id}",
        "custom_label": f"custom_label_{part_id}",
    }


def read_states(part_id: str) -> Dict[str, Any]:
    """
    Reads a set of states based on a given part ID.

    Args:
        part_id (str): The ID of the part to read the states for.

    Returns:
        dict: A dictionary of states.
    """
    widget_keys = format_keys(part_id)
    return {key: st.session_state[value] for key, value in widget_keys.items()}


@st.cache_data
def name_part(instrument_name: str, custom_label: str) -> str:
    """Returns a formatted string representing a part's name.

    Args:
        instrument_name (str): The name of the instrument associated with the part.
        custom_label (str): A custom label for the part.

    Returns:
        str: A formatted string representing the name of the part.
    """
    return custom_label or instrument_name


def serialize_part(musical_notes: str, instrument_name: str, custom_label: str) -> Part:
    """
    Create a Music21 Part object based on the given parameters.

    Args:
        musical_notes (str): A string representation of the musical notes to be included in the part.
        instrument_name (str): The name of the instrument to use for the part.
        custom_label (str): A custom label to append to the instrument name to create the part name.

    Returns:
        Part: A Music21 Part object containing the specified musical notes and instrument information.
    """
    part = parse(musical_notes, format="tinyNotation") if musical_notes else Part()
    part.insert(0, getattr(instrument, instrument_name)())
    part.partName = name_part(instrument_name, custom_label)
    return part


def trim(im: Image) -> Image:
    """
    Trims an image by cropping it to the bounding box of the non-white pixels.
    Adds a 5 pixel white border around the resulting image.

    Args:
        im (Image): The image to trim.

    Returns:
        Image: The trimmed image with a 5 pixel white border.
    """

    if sys.platform == "darwin":
        im_gray = ImageOps.grayscale(im).convert("RGB")
        # Invert the image to convert the white pixels to black and vice versa
        im_inverted = ImageOps.invert(im_gray)

        # Create a mask of the non-white pixels
        im_mask = ImageChops.darker(im_inverted, ImageOps.invert(im_inverted))

        # Crop the image to the bounding box of the non-white pixels
        im_cropped = im.crop(im_mask.getbbox())
    else:
        im = im.convert("RGBA")
        im_cropped = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im_cropped.paste(im, mask=im)
    im_padded = ImageOps.expand(im_cropped, border=5, fill="white")
    return im_padded


def show_image(
    musical_notes: Optional[str] = None, stream: Optional[Stream] = None
) -> None:
    """Displays an image of a music score created from a `music21` stream object or a string of musical notes.

    Args:
        musical_notes (str, optional): A string of musical notes in `tinyNotation` format. Defaults to None.
        stream (music21.stream.Stream, optional): A `music21` stream object. Defaults to None.

    Returns:
        None: Displays the music score image using Streamlit's `st.image` function.

    Raises:
        ValueError: If `musical_notes` is an empty string.
    """
    if musical_notes:
        stream = parse(musical_notes, format="tinyNotation")
    elif stream:
        musical_notes = stream.flat.notes

    if len(musical_notes) == 0:
        st.error("The part is empty! Please first enter some musical notes.")
        return

    try:
        image_path = stream.write("musicxml.png")
    except Exception as e:
        st.error(f"Parsing error; cannot render the part: {e}.")
        return
    with Image.open(image_path) as part_image:
        return st.image(trim(part_image), use_column_width=True)


def create_part_inputs(
    part_id: Optional[str] = None,
    musical_notes: Optional[str] = None,
    instrument_name: Optional[str] = None,
    custom_label: Optional[str] = None,
) -> str:
    """
    Displays a UI for creating or editing a piano part.

    Args:
        part_id (str, optional): The unique ID of the part. If provided, the part
            will be loaded from session state. Defaults to None.
        musical_notes (str, optional): The musical notes to use for the part. If
            provided, the text area will be pre-populated with this value.
            Defaults to None.
        instrument_name (str, optional): The name of the instrument to use for the
            part. If provided, the select box will be set to this value. Defaults
            to None.
        custom_label (str, optional): A custom label to append to the part title.

    Returns:
        str: The ID of the created or edited part.
    """
    if part_id:
        states = read_states(part_id)
        musical_notes = states["musical_notes"]
        instrument_name = states["instrument_name"]
        custom_label = states["custom_label"]
    else:
        part_id = uuid4().hex
        musical_notes = (musical_notes or "").strip()
        instrument_name = (instrument_name or "").strip()
        custom_label = (custom_label or "").strip()
        st.session_state.part_ids.append(part_id)

    part_keys = format_keys(part_id)

    deletable_wrapper = st.empty()

    with deletable_wrapper.container():
        header = st.subheader(f"Part")
        with st.expander("Click to expand this part.", expanded=True):
            st.text_area(
                label="🎹 Enter some musical notes.",
                value=musical_notes,
                key=part_keys["musical_notes"],
            )

            if instrument_name and instrument_name in INSTRUMENT_OPTIONS:
                index = INSTRUMENT_OPTIONS.index(instrument_name)
            else:
                index = random.randint(0, len(INSTRUMENT_OPTIONS)) - 1
                instrument_name = INSTRUMENT_OPTIONS[index]

            st.selectbox(
                label="🥁 Choose an instrument to use.",
                options=INSTRUMENT_OPTIONS,
                index=index,
                key=part_keys["instrument_name"],
            )

            st.text_input(
                label="✏️ Label this part if desired.",
                value=custom_label,
                key=part_keys["custom_label"],
            )

            part_name = name_part(instrument_name, custom_label)
            header.subheader(part_name)

            part = serialize_part(musical_notes, instrument_name, custom_label)

            col1, col2 = st.columns(2)
            with col1:
                show = st.button(
                    "🎼 Show this part.",
                    key=f"part_show_{part_id}",
                    use_container_width=True,
                )
            if show:
                with st.spinner():
                    show_image(musical_notes=musical_notes)
            with col2:
                delete = st.button(
                    label="🗑️ Delete this part.",
                    key=f"delete_part_{part_id}",
                    use_container_width=True,
                )
            play_stream_inputs(part, key=f"part_player_{part_id}")

    if delete:
        st.session_state.part_ids.remove(part_id)
        deletable_wrapper.empty()
    return part_id


def create_song():
    """
    Creates a music Score object from the serialized parts in session state.

    Returns:
        A Score object containing all serialized parts.
    """
    song = Score()
    for part_id in st.session_state.part_ids:
        states = read_states(part_id)
        part = serialize_part(**states)
        if states["musical_notes"]:
            song.append(part)
    return song


def output_song(song: Score):
    """Outputs the current song in the selected format.

    Returns:
        None
    """
    format = st.session_state.output_format
    if format == "":
        return

    if format == "midi":
        temp_midi_path = song.write("midi")
        with open(temp_midi_path, "rb") as f:
            st.download_button(
                file_name="musicail.mid",
                label="💾 Click to download MIDI.",
                data=f.read(),
                use_container_width=True,
            )
    elif format == "png":
        st.markdown("💾 Right click and 'Save image as...' to download.")
        show_image(stream=song)
    elif format == "xml":
        temp_xml_path = song.write("musicxml")
        with open(temp_xml_path, "r") as f:
            contents = f.read()
            st.download_button(
                file_name="musicail.xml",
                label="💾 Click to download XML.",
                data=contents,
                use_container_width=True,
            )
            st.code(contents, language="xml")


def serialize_composition(composition: str) -> Set[str]:
    """serializeuate a music composition in a specific format and create input fields for each part.

    Args:
        composition: A string representing a music composition in the following format:
            ```
            <instrument_name>: <musical_notes>
            ---
            <instrument_name>: <musical_notes>
            ...
            ```
            where `---` separates different parts, `<instrument_name>` is the name of the instrument for the corresponding
            part, and `<musical_notes>` is a string of musical notes in the TinyNotation format.

    Returns:
        None.
    """
    clear_parts()
    try:
        composition = composition.strip()
        if "```" in composition:
            composition = composition.split("```")[1]
        # sometimes, models don't listen :(
        sep = "---"
        if sep not in composition:
            sep = "\n\n"
        if sep not in composition:
            sep = "\n"
        parts = composition.strip().split(sep)
    except ValueError:
        st.sidebar.error(
            f"The composition is not formatted correctly {composition}; "
            "please ensure it follows the example output."
        )
        return

    part_ids = set()
    for part in parts:
        if not part.strip():
            continue
        try:
            components = part.strip().split(":", maxsplit=3)
            custom_label, instrument_name, musical_notes = components[:3]
        except ValueError:
            st.sidebar.warning(f"{part} cannot be parsed; skipping...")
            continue
        part_id = create_part_inputs(
            musical_notes=musical_notes,
            instrument_name=instrument_name,
            custom_label=custom_label,
        )
        part_ids.add(part_id)
    return part_ids


def clear_parts():
    st.session_state.part_ids = []
    placeholder.empty()


# initialize session state

if "memory_stack" not in st.session_state:
    st.session_state.memory_stack = []

if "part_ids" not in st.session_state:
    st.session_state.part_ids = []

if "output_format" not in st.session_state:
    st.session_state.output_format = ""

# start laying out widgets

st.title("🎶 MusicAIl")
st.subheader("Music composition accessible to all with AI.")
st.markdown(INSTRUCTIONS)

# this must be up here for sidebar to use
placeholder = st.empty()
part_container = placeholder.container()

# create sidebar

st.sidebar.subheader("🤖 Compose a song with AI.")
composer = st.sidebar.text_input(
    "🧑‍🎤 Enter an inspirational composer.", value=""
)
description = st.sidebar.text_area(
    "📝 Describe a song to compose.",
    value=DEFAULT_DESCRIPTION,
)
instruments = st.sidebar.multiselect(
    label="🪗 Select the instruments that the AI should use.",
    options=INSTRUMENT_OPTIONS,
    default=["Piano", "Guitar", "BassDrum"],
)
minutes = st.sidebar.slider(
    label="Choose how long, in minutes, the song should be.",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.05,
)

prompt_template = PromptTemplate(
    template=TEMPLATE,
    input_variables=["composer", "minutes", "description", "instruments"],
)
prompt_inputs = dict(
    composer=composer,
    description=description,
    instruments=", ".join(instruments),
    minutes=minutes,
)
prompt = prompt_template.format(**prompt_inputs)

tab1, tab2 = st.sidebar.tabs(["Online LLMs", "OpenAI API"])
with tab2:
    st.warning(
        "⚠️ This feature may not work well due to extraneous output from the model. "
        "It's recommended to chat with an Online LLM and paste the final composition."
    )
    api_key = st.text_input("🔑 Paste in an OpenAI API key.", type="password")
    model_name = st.selectbox(
        label="Choose a model.",
        options=["gpt-3.5-turbo", "davinci", "curie", "babbage", "ada"],
        index=0,
    )
    temperature = st.slider(
        label="Choose how creative the AI should be.",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )
    max_tokens = st.slider(
        label="Set a threshold for the max tokens used.",
        min_value=1,
        max_value=4096,
        value=500,
        step=1,
    )
with tab1:
    pasted_output = st.text_area(
        "📋 Paste in a composition from online LLMs to serialize it. "
        "Ensure the output format follows the example output below!"
    )

serialized_part_ids = {}
if st.sidebar.button("🏃‍♀️ Clear all parts and run.", use_container_width=True):
    if pasted_output and api_key:
        st.sidebar.warning(
            "Both pasted output and API key are provided; " "will use pasted output."
        )
    if pasted_output:
        serialized_part_ids = serialize_composition(pasted_output)
    elif api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        llm_class = ChatOpenAI if model_name == "gpt-3.5-turbo" else OpenAI
        llm = llm_class(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
        with st.spinner():
            llm_chain = LLMChain(prompt=prompt_template, llm=llm)
            api_output = llm_chain.run(**prompt_inputs)
            serialized_part_ids = serialize_composition(api_output)

st.sidebar.markdown(f"💬 Here's a prompt template to copy:")
st.sidebar.code(prompt, language="text")

# create main
with part_container:
    for part_id in st.session_state.part_ids[:]:
        if part_id in serialized_part_ids:
            continue
        create_part_inputs(part_id)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🎻 Add a new part.", key="add_part", use_container_width=True):
        with part_container:
            part_id = create_part_inputs()
with col2:
    output = st.button(
        "📦 Package all parts.",
        key="output_song",
        use_container_width=True,
    )
with col3:
    clear = st.button(
        "🗑️ Clear all parts.",
        key="clear_parts",
        use_container_width=True,
    )

song = create_song()
play_stream_inputs(song, key="play_score")
if output:
    format = st.selectbox(
        label="🗄️ Select the output format.",
        options=["", "midi", "png", "xml", "tinynotation"],
        key="output_format",
    )
with st.spinner():
    output_song(song=song)


if clear:
    clear_parts()
