from dataclasses import dataclass
import os
import random
import streamlit as st
from uuid import uuid4
from textwrap import dedent
from music21.converter import parse
from music21.stream import Score, Part
from music21.midi.realtime import StreamPlayer
from music21 import instrument
from PIL import Image, ImageOps, ImageChops
from tempfile import NamedTemporaryFile
from music21 import environment
import subprocess
from langchain import PromptTemplate, LLMChain, OpenAI


def is_subclass(obj):
    try:
        return issubclass(obj, instrument.Instrument)
    except TypeError:
        return False


TEMPLATE = dedent(
    """
    You are a helpful chat assistant.

    Imagine you are the brilliant {composer}.

    Compose a {minutes} minutes long song, following this description: {description}.

    The song should include these instruments: {instruments}.

    Think about it step by step and ensure that each instrument part
    complements one another.

    Respond with ONLY the song using tinynotation, wrapped in ```.
    
    Each instrument part should be prefixed with instrument name as provided
    and on its own line separated by ---.

    Here's an example output:
    ```
    Violin: 4/4 c4 d b-1
    ---
    TenorSaxophone: 4/4 A B C
    ```
    """
).strip()
MSCORE_PATH = "/opt/homebrew/bin/mscore"
environment.set("musescoreDirectPNGPath", MSCORE_PATH)
INSTRUMENT_OPTIONS = [
    var_name
    for var_name in dir(instrument)
    if is_subclass(getattr(instrument, var_name))
    and var_name not in {"Instrument", "Conductor"}
]


INSTRUCTIONS = """
    With MusicAIl, you *don't need* to be an expert in music theory
    to compose beautiful songs.

    Start by adding a part and entering some musical notes in [tinynotation](
    https://web.mit.edu/music21/doc/usersGuide/usersGuide_16_tinyNotation.html).

    You can enter simple music notes such as `A B r C`, where `r` denotes a rest.
    
    Alternatively, you can mix complex symbols with notes like
    `4/4 c4 trip{c8 d e} trip{f4 g a} b-1`.

    Add as many parts as you want--each part will be played simultaneously.
    
    **Or, simply have AI compose a song for you by expanding the AI options!**

"""


def hear_stream(stream):
    player = StreamPlayer(stream)
    if player.pygame.mixer.music.get_busy():
        player.stop()
    else:
        player.play(blocked=False)


def hear_stream_inputs(stream, key):
    label = "part" if "part" in key else "song"
    if st.button(f"🔊 Play or pause {label}.", key=key, use_container_width=True):
        if len(stream.flat.notes) == 0:
            st.error(f"The {label} is empty! Please first enter some musical notes.")
            return
        hear_stream(stream)


@st.cache_data
def format_keys(part_id):
    return {
        "musical_notes": f"musical_notes_{part_id}",
        "instrument_name": f"instrument_name_{part_id}",
        "custom_label": f"custom_label_{part_id}",
    }


def read_states(part_id):
    widget_keys = format_keys(part_id)
    return {key: st.session_state[value] for key, value in widget_keys.items()}


@st.cache_data
def name_part(instrument_name, custom_label):
    return f"{instrument_name} {custom_label} Part"


def serialize_part(musical_notes, instrument_name, custom_label):
    part = parse(musical_notes, format="tinyNotation") if musical_notes else Part()
    part.insert(0, getattr(instrument, instrument_name)())
    part.partName = name_part(instrument_name, custom_label)
    return part


def trim(im):
    im_gray = ImageOps.grayscale(im)

    # Invert the image to convert the white pixels to black and vice versa
    im_inverted = ImageOps.invert(im_gray)

    # Create a mask of the non-white pixels
    im_mask = ImageChops.darker(im_inverted, ImageOps.invert(im_inverted))

    # Crop the image to the bounding box of the non-white pixels
    im_cropped = im.crop(im_mask.getbbox())
    im_padded = ImageOps.expand(im_cropped, border=5, fill="white")
    return im_padded


def show_image(musical_notes=None, stream=None):
    if musical_notes:
        if len(musical_notes) == 0:
            st.error("The part is empty! Please first enter some musical notes.")
            return
        stream = parse(musical_notes, format="tinyNotation")
    try:
        image_path = stream.write("musicxml.png")
    except Exception as e:
        st.error(f"Parsing error; cannot render the part: {e}.")
        return
    with Image.open(image_path) as part_image:
        return st.image(trim(part_image), use_column_width=True)


def create_part_inputs(part_id=None, musical_notes=None, instrument_name=None):
    if part_id:
        states = read_states(part_id)
        musical_notes = states["musical_notes"]
        instrument_name = states["instrument_name"]
        custom_label = states["custom_label"]
    else:
        part_id = uuid4().hex
        musical_notes = musical_notes or ""
        instrument_name = instrument_name or ""
        custom_label = ""
        st.session_state.part_ids.append(part_id)

    part_keys = format_keys(part_id)

    deletable_wrapper = st.empty()

    with deletable_wrapper.container():
        header = st.subheader(f"Piano Part")
        st.text_area(
            label="🎹 Enter some musical notes.",
            value=musical_notes,
            key=part_keys["musical_notes"],
        )

        if instrument_name:
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

        part_name = name_part(custom_label, instrument_name)
        header.subheader(part_name)

        part = serialize_part(musical_notes, instrument_name, custom_label)

        col1, col2, col3 = st.columns(3)
        with col1:
            hear_stream_inputs(part, key=f"part_player_{part_id}")
        with col2:
            show = st.button(
                "🎼 Show part.", key=f"part_show_{part_id}", use_container_width=True
            )
        if show:
            show_image(musical_notes=musical_notes)
        with col3:
            delete = st.button(
                label="🗑️ Delete part.",
                key=f"delete_part_{part_id}",
                use_container_width=True,
            )

    if delete:
        st.session_state.part_ids.remove(part_id)
        deletable_wrapper.empty()
    return part_id


def create_song():
    song = Score()
    for part_id in st.session_state.part_ids:
        states = read_states(part_id)
        part = serialize_part(**states)
        if states["musical_notes"]:
            song.append(part)
    return song


def output_song():
    format = st.session_state.output_format
    if format == "":
        return

    if format == "mp3":
        temp_midi_path = song.write("midi")
        with NamedTemporaryFile(suffix=".mp3") as temp_mp3_file:
            temp_mp3_path = temp_mp3_file.name
            subprocess.run([MSCORE_PATH, "-o", temp_mp3_path, temp_midi_path])
            st.text("💾 Click on ⋮ to download.")
            st.audio(temp_mp3_path, format="audio/mpeg")
    elif format == "midi":
        temp_midi_path = song.write("midi")
        with open(temp_midi_path, "rb") as f:
            st.download_button(
                file_name="musicail.mid",
                label="💾 Click to download MIDI.",
                data=f.read(),
                use_container_width=True,
            )
    elif format == "png":
        st.text("💾 Right click and 'Save image as...' to download.")
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


def eval_composition(composition):
    try:
        if "```" in composition:
            composition = composition.split("```")[1]
        parts = composition.strip().split("---")
    except ValueError:
        st.error(
            "The composition is not formatted correctly; "
            "please ensure it follows the example output."
        )
        return

    st.session_state.part_ids = []
    for part in parts:
        instrument_name, musical_notes = part.strip().split(": ")
        create_part_inputs(musical_notes=musical_notes, instrument_name=instrument_name)


st.title("🎶 MusicAIl")
st.subheader("Music composition accessible to all with AI.")
st.markdown(INSTRUCTIONS)

if "memory_stack" not in st.session_state:
    st.session_state.memory_stack = []

if "part_ids" not in st.session_state:
    st.session_state.part_ids = []

if "output_format" not in st.session_state:
    st.session_state.output_format = ""

part_container = st.container()

with part_container:
    for part_id in st.session_state.part_ids[:]:
        create_part_inputs(part_id)

with st.expander("🤖 Click to expand AI options."):
    composer = st.text_input("🧑‍🎤 Enter an inspirational composer.", value="Composer")
    description = st.text_area("📝 Describe a song to compose.")
    instruments = st.multiselect(
        label="🪗 Select the instruments that the AI should use.",
        options=INSTRUMENT_OPTIONS,
        default=["Piano"],
    )
    _, col, _ = st.columns([1, 80, 1])
    with col:
        minutes = st.slider(
            label="Choose how long, in minutes, the song should be.",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.05,
        )

    prompt_template = PromptTemplate(
        template=TEMPLATE,
        input_variables=["composer", "description", "instruments", "minutes"],
    )
    prompt_inputs = dict(
        composer=composer,
        description=description,
        instruments=", ".join(instruments),
        minutes=minutes,
    )
    prompt = prompt_template.format(**prompt_inputs)
    st.text(f"💬 The AI will use this as the prompt:\n\n{prompt}")

    llm = None
    tab1, tab2 = st.tabs(["OpenAI API", "Online LLMs"])
    with tab1:
        api_key = st.text_input("🔑 Paste in an OpenAI API key.", type="password")
        temperature = st.slider(
            label="Choose how creative the AI should be.",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            llm = OpenAI(temperature=temperature)
    with tab2:
        pasted_output = st.text_area(
            "📋 Paste in a composition from online LLMs, like "
            "from ChatGPT, Bing Chat, Bard, etc. Ensure the "
            "output format follows the example prompt!"
        )

    if st.button("🏃‍♀️ Run", use_container_width=True):
        if pasted_output:
            eval_composition(pasted_output)
        elif llm:
            llm_chain = LLMChain(prompt=prompt_template, llm=llm)
            api_output = llm_chain.run(**prompt_inputs)
            with part_container:
                eval_composition(api_output)

if st.button("🎻 Add new part.", key="add_part", use_container_width=True):
    with part_container:
        part_id = create_part_inputs()

st.divider()

col1, col2 = st.columns(2)
song = create_song()
with col1:
    hear_stream_inputs(song, key="song_player")
with col2:
    output = st.button("📁 Output song.", key="output_song", use_container_width=True)
if output:
    format = st.selectbox(
        label="🗄️ Select the output format.",
        options=["", "mp3", "midi", "png", "xml"],
        key="output_format",
    )
output_song()
