from dataclasses import dataclass
from enum import Enum
import streamlit as st
from uuid import uuid4
from typing import Optional
from music21.converter import parse
from music21.stream import Score, Part
from music21.midi.realtime import StreamPlayer
from music21 import instrument


def is_subclass(obj):
    try:
        return issubclass(obj, instrument.Instrument)
    except TypeError:
        return False


INSTRUMENT_OPTIONS = [
    var_name
    for var_name in dir(instrument)
    if is_subclass(getattr(instrument, var_name))
    and var_name not in {"Instrument", "Conductor"}
]


INSTRUCTIONS = """
    *With MusicAIl, you don't need to be an expert in music theory
    to create beautiful songs.*

    Simply, add a part and enter some musical notes in [tinynotation](
    https://web.mit.edu/music21/doc/usersGuide/usersGuide_16_tinyNotation.html).

    Input can be as simple as `A B C D E F`.

    Or something more complex like `4/4 c4 trip{c8 d e} trip{f4 g a} b-1`.
    
    Add as many parts as you want--each part will be played simultaneously.
"""


def create_player(stream, key):
    player = StreamPlayer(stream)
    label = "part" if "part" in key else "score"
    if st.button(f"🔊 Play or pause the {label}.", key=key, use_container_width=True):
        empty_part = len(stream) == 1 and label == "part"
        empty_score = len(stream) == 0 and label == "score"
        if empty_part or empty_score:
            st.error(f"The {label} is empty! Please first enter some musical notes.")
            return
        if player.pygame.mixer.music.get_busy():
            player.stop()
        else:
            player.play(blocked=False)


def format_keys(part_id):
    return {
        "musical_notes": f"musical_notes_{part_id}",
        "instrument_name": f"instrument_name_{part_id}",
        "custom_label": f"custom_label_{part_id}",
    }


def read_states(part_id):
    widget_keys = format_keys(part_id)
    return {key: st.session_state[value] for key, value in widget_keys.items()}


def name_part(instrument_name, custom_label):
    return f"{instrument_name} {custom_label} Part"


def serialize_part(musical_notes, instrument_name, custom_label):
    part = parse(musical_notes, format="tinyNotation") if musical_notes else Part()
    part.insert(0, getattr(instrument, instrument_name)())
    part.partName = name_part(instrument_name, custom_label)
    return part


def create_part_inputs(part_id=None):
    title = st.title(f"Piano Part")

    if part_id:
        states = read_states(part_id)
        musical_notes = states["musical_notes"]
        instrument_name = states["instrument_name"]
        custom_label = states["custom_label"]
    else:
        part_id = uuid4().hex
        musical_notes = ""
        instrument_name = "Piano"
        custom_label = ""

    part_keys = format_keys(part_id)

    with st.expander(label="Click to expand.", expanded=True):
        st.text_area(
            label="🎹 Enter some musical notes.",
            value=musical_notes,
            key=part_keys["musical_notes"],
        )

        index = INSTRUMENT_OPTIONS.index(instrument_name)
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
        title.title(part_name)

        part = serialize_part(musical_notes, instrument_name, custom_label)
        create_player(part, key=f"part_player_{part_id}")

    return part_id


def create_score():
    score = Score()
    for part_id in st.session_state.part_ids:
        states = read_states(part_id)
        part = serialize_part(**states)
        if states["musical_notes"]:
            score.append(part)
    return score


if "part_ids" not in st.session_state:
    st.session_state.part_ids = []

st.title("🎼 MusicAIl")
st.subheader("Music composition accessible to all with AI.")
st.markdown(INSTRUCTIONS)

col1, col2 = st.columns(2)
with col1:
    add_button = st.button("🎻 Add a new part.", key="add_part", use_container_width=True)
with col2:
    score = create_score()
    create_player(score, key="score_player")

for part_id in st.session_state.part_ids[:]:
    create_part_inputs(part_id)

if add_button:
    part_id = create_part_inputs()
    st.session_state.part_ids.append(part_id)
