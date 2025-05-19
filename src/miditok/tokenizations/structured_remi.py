"""REMI (Revamped MIDI) tokenizer."""

from __future__ import annotations

print("\n=== Loading structured_remi.py ===")

import logging
import copy
import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict
import sys # <<< Added import

import numpy as np

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
    ControlChange,
    TextMeta,
)
from symusic.core import NoteTickList
import symusic

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import (
    ADD_TRAILING_BARS,
    DEFAULT_VELOCITY,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
    USE_BAR_END_TOKENS,
)
from miditok.tokenizations.remi import REMI
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat
from miditok.utils.structured_utils import (
    get_instrument_type_from_program,
    get_relevant_ccs_for_instrument,
    normalize_track_name,
    get_instrument_type_from_track_name,
    extract_chord_events,
    TOKENIZER_EXPECTED_QUALITIES,
    TARGET_INSTRUMENT_FAMILIES,
    map_track_name_to_family,
    MUSIC21_AVAILABLE # Import the flag
)
import functools # <<< Add import

# --- Custom Event Types ---

class TrackNameEvent(Event):
    """Represents a track name"""
    def __init__(self, name: str, instrument_type: str = "unknown", time: int = 0):
        super().__init__("TrackName", name, time)
        self.instrument_type = instrument_type

    def __repr__(self):
        return f"TrackName(time={self.time}, value={self.value}, type={self.instrument_type})"

class GlobalChordRootEvent(Event):
    """Represents a global chord root"""
    def __init__(self, root_pc: int, time: int = 0):
        super().__init__("GlobalChordRoot", str(root_pc), time)

    def __repr__(self):
        return f"GlobalChordRoot(time={self.time}, value={self.value})"

class GlobalChordQualEvent(Event):
    """Represents a global chord quality"""
    def __init__(self, quality: str, time: int = 0):
        super().__init__("GlobalChordQual", quality, time)

    def __repr__(self):
        return f"GlobalChordQual(time={self.time}, value={self.value})"

class CCTypeEvent(Event):
    """Represents a CC type (number)"""
    def __init__(self, number: int, time: int = 0):
        super().__init__("CCType", str(number), time)

    def __repr__(self):
        return f"CCType(time={self.time}, value={self.value})"

class CCValueEvent(Event):
    """Represents a CC value"""
    def __init__(self, value: int, time: int = 0): # Value here is the binned value
        super().__init__("CCValue", str(value), time)

    def __repr__(self):
        return f"CCValue(time={self.time}, value={self.value})"

class RomanNumeralEvent(Event):
    """Represents a Roman Numeral analysis result"""
    def __init__(self, figure: str, time: int = 0):
        # Sanitize figure: replace slashes, sharps, flats etc. for token compatibility
        sanitized_figure = figure.replace('#', 'sharp').replace('b', 'flat').replace('/', 'slash')
        super().__init__("RomanNumeral", sanitized_figure, time)

    def __repr__(self):
        return f"RomanNumeral(time={self.time}, value={self.value})"

class NeoRiemannianEvent(Event):
    """Represents a Neo-Riemannian transformation"""
    def __init__(self, transformation: str, time: int = 0):
        # Transformation value should already be simple (P, L, R)
        super().__init__("NeoRiemannian", transformation, time)

    def __repr__(self):
        return f"NeoRiemannian(time={self.time}, value={self.value})"

# --- Predefined Sets for Vocabulary ---
# Common Roman Numeral figures (can be expanded)
# Need to sanitize these according to the RomanNumeralEvent logic
# Example: V/V -> VslashV, viio -> viio, V7 -> V7, iv6 -> iv6
COMMON_ROMAN_NUMERALS = {
    # Major Key (examples)
    "I", "ii", "iii", "IV", "V", "vi", "viio",
    "I6", "ii6", "iii6", "IV6", "V6", "vi6",
    "I64", "ii64", "iii64", "IV64", "V64", "vi64",
    "V7", "viio7", "VslashV", "V7slashV", "viioslashV",
    # Minor Key (examples)
    "i", "iio", "III", "iv", "V", "VI", "VII", "viio",
    "i6", "iio6", "III6", "iv6", "V6", "VI6", "VII6",
    "i64", "iio64", "III64", "iv64", "V64", "VI64", "VII64",
    "V7", "viio7", "III+", # Add augmented III
    # Others (Neapolitan, etc.)
    "flatII", "flatII6",
    # Mode mixture examples
    "iv", "flatVI", "flatVII", # in Major
    "VI", # Picardy in minor
}
COMMON_NEO_RIEMANNIAN = {"P", "L", "R"} # Basic P, L, R

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class StructuredREMI(REMI):
    r"""
    Structured REMI tokenizer.

    Extends REMI to include structured elements like track names, global chords,
    and instrument-aware Control Changes (CCs).

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object. **Note:** Ensure that
        `use_control_changes=False` in the config, as this class handles CCs directly.
    :param use_track_names: Add specific tokens for track names.
    :param use_global_chords: Add tokens representing global chord context.
    :param use_filtered_ccs: Add tokens for CCs, filtered based on instrument type.
    :param cc_filter_mode: How to filter CCs: 'instrument_aware', 'all',
        'explicit_list', 'none'.
    :param cc_list: List of CC numbers to use if `cc_filter_mode` is 'explicit_list'.
    :param cc_bins: Number of bins to quantize CC values into.
    :param chord_analysis_func: Optional function to perform chord analysis.
        Defaults to `extract_chord_events`.
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file.
    """
    print("\n=== Defining StructuredREMI class ===")

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        # New custom parameters
        use_track_names: bool = True,
        use_global_chords: bool = True,
        use_filtered_ccs: bool = True,
        cc_filter_mode: str = 'instrument_aware',
        cc_list: Optional[List[int]] = None,
        cc_bins: int = 128,
        chord_analysis_func: Optional[callable] = None,
        # <<< New Flags >>>
        use_roman_numerals: bool = True, # Default to False
        use_neo_riemannian: bool = True, # Default to False
        # Original REMI params (max_bar_embedding handled via config)
        params: str | Path | None = None,
    ) -> None:
        print("\n=== Initializing StructuredREMI instance ===")
        
        # --- Initialize logger FIRST --- 
        self.logger = logging.getLogger(__name__)
        # print(f"DEBUG: Logger name is {__name__}") # Optional print
        self.logger.setLevel(logging.WARNING)
        # --- End logger init ---
        
        # --- Store Custom Params Temporarily --- #
        # We store them first, then ensure they are in the config before super().__init__
        # and finally reload them from the config after super().__init__ in case a params file was loaded.
        _use_track_names = use_track_names
        _use_global_chords = use_global_chords
        _use_filtered_ccs = use_filtered_ccs
        _cc_filter_mode = cc_filter_mode
        _explicit_cc_list = set(cc_list) if cc_list is not None else set()
        _num_cc_bins = cc_bins
        # <<< Store new flags temporarily >>>
        _use_roman_numerals = use_roman_numerals
        _use_neo_riemannian = use_neo_riemannian

        # --- Music21 Requirement Check ---
        if (_use_global_chords or _use_roman_numerals or _use_neo_riemannian) and not MUSIC21_AVAILABLE:
            raise ImportError(
                "'use_global_chords', 'use_roman_numerals', or 'use_neo_riemannian' requires music21. "
                "Please install it (`pip install music21`)."
            )

        # --- Setup Config (Important: before super().__init__) --- #
        # Provide a default config if none is given and no params file is specified
        if tokenizer_config is None and params is None:
            tokenizer_config = TokenizerConfig(
                use_chords=True, # Base miditok chords off if using our global ones
                use_rests=True,
                use_tempos=True,
                use_time_signatures=True,
                use_programs=True,
                use_control_changes=False, # Ensure miditok CCs are OFF
                one_token_stream_for_programs=True, # Usually desired for structured approach
                num_velocities=64,
                beat_res={(0, 4): 8, (4, 12): 4},
                additional_params={
                    'tempo_range': (40, 250), 'num_tempos': 64,
                    'time_signature_range': {4: [4], 2: [2], 8: [3, 5, 6, 12]},
                    'max_bar_embedding': None, # REMI default
                    'use_bar_end_tokens': False,
                    'add_trailing_bars': False,
                    # Add flags to config so they are saved/loaded automatically
                    'use_track_names': True,
                    'use_global_chords': _use_global_chords,
                    'use_filtered_ccs': _use_filtered_ccs,
                    'cc_filter_mode': _cc_filter_mode,
                    'cc_list': list(_explicit_cc_list),
                    'num_cc_bins': _num_cc_bins,
                    # <<< Add new flags to default config >>>
                    'use_roman_numerals': _use_roman_numerals,
                    'use_neo_riemannian': _use_neo_riemannian,
                }
            )
        # If config is provided, ensure miditok CCs are off and store custom params
        elif tokenizer_config is not None:
            # Store/update custom params in additional_params for saving
            tokenizer_config.additional_params['use_track_names'] = _use_track_names
            tokenizer_config.additional_params['use_global_chords'] = _use_global_chords
            tokenizer_config.additional_params['use_filtered_ccs'] = _use_filtered_ccs
            tokenizer_config.additional_params['cc_filter_mode'] = _cc_filter_mode
            tokenizer_config.additional_params['cc_list'] = list(_explicit_cc_list)
            # Only set default if not already present in the passed config
            if 'num_cc_bins' not in tokenizer_config.additional_params:
                 tokenizer_config.additional_params['num_cc_bins'] = _num_cc_bins
                 #self.logger.debug(f"Setting 'num_cc_bins' to default value: {_num_cc_bins}")
            else:
                 # If present, use the value from the config and update _num_cc_bins
                 # so that the .get() fallback later uses the correct value if needed.
                 _num_cc_bins = tokenizer_config.additional_params['num_cc_bins']
                 #self.logger.debug(f"Using 'num_cc_bins' ({_num_cc_bins}) from provided tokenizer_config.")
            # <<< Store/update new flags in provided config >>>
            tokenizer_config.additional_params['use_roman_numerals'] = _use_roman_numerals
            tokenizer_config.additional_params['use_neo_riemannian'] = _use_neo_riemannian
        # If only params is provided, the loaded config might not have our custom params yet.
        # We'll handle loading them *after* super().__init__

        # --- Call Parent Init --- #
        # Needs to happen before logger is initialized if parent uses it, but after config mods
        # Note: Removed max_bar_embedding direct param, handled by config
        super().__init__(tokenizer_config, params)

        # --- Setup Post Parent Init --- # 
        # --- Initialize logger right after superclass init ---
        self.logger = logging.getLogger(__name__)
        # print(f"DEBUG: Logger name is {__name__}") # Optional print
        self.logger.setLevel(logging.WARNING)
        # --- End logger init ---

        # *** ADDED CHECK: Ensure additional_params exists after super init ***
        if not hasattr(self.config, 'additional_params') or self.config.additional_params is None:
            self.logger.warning("self.config.additional_params was missing or None after super().__init__(). Initializing to empty dict.")
            # Ensure the config object itself exists before trying to modify it
            if self.config is not None:
                try:
                    # Only try to set if the attribute exists but is None, or doesn't exist
                    if not hasattr(self.config, 'additional_params') or getattr(self.config, 'additional_params', None) is None:
                         setattr(self.config, 'additional_params', {})
                except AttributeError as e:
                     self.logger.error(f"Could not set additional_params on self.config object: {e}")
                     # Let the original error likely happen below if we couldn't fix it.
                     pass
            else:
                 self.logger.error("self.config itself is None after super().__init__(). Cannot proceed.")
                 # Raise an error or handle appropriately, as config is essential
                 raise AttributeError("self.config is None after parent initialization.")


        # Reload custom attributes from the *final* config (self.config)
        # Use the temporarily stored defaults (_use_track_names, _num_cc_bins, etc.) as fallbacks
        self.use_track_names = self.config.additional_params.get('use_track_names', _use_track_names)
        self.use_global_chords = self.config.additional_params.get('use_global_chords', _use_global_chords)
        self.use_filtered_ccs = self.config.additional_params.get('use_filtered_ccs', _use_filtered_ccs)
        self.cc_filter_mode = self.config.additional_params.get('cc_filter_mode', _cc_filter_mode)
        loaded_cc_list = self.config.additional_params.get('cc_list', list(_explicit_cc_list))
        self.explicit_cc_list = set(loaded_cc_list)
        # Load num_cc_bins from the final config, falling back to the initial default (_num_cc_bins = 32)
        # only if it's not present in the final config.
        loaded_num_bins = self.config.additional_params.get('num_cc_bins', _num_cc_bins) 
        if loaded_num_bins != _num_cc_bins:
             self.logger.info(f"Overriding initial num_cc_bins ({_num_cc_bins}) with value from loaded config: {loaded_num_bins}")
        self.num_cc_bins = loaded_num_bins
        # --- Force update the config object itself --- 
        self.config.additional_params['num_cc_bins'] = self.num_cc_bins
        self.logger.info(f"Final effective num_cc_bins set to: {self.num_cc_bins}") # Log the final value

        # <<< Reload new flags from final config >>>
        self.use_roman_numerals = self.config.additional_params.get('use_roman_numerals', _use_roman_numerals)
        self.use_neo_riemannian = self.config.additional_params.get('use_neo_riemannian', _use_neo_riemannian)
        # <<< Update config with final flags >>>
        self.config.additional_params['use_roman_numerals'] = self.use_roman_numerals
        self.config.additional_params['use_neo_riemannian'] = self.use_neo_riemannian
        self.logger.info(f"Final effective use_roman_numerals: {self.use_roman_numerals}")
        self.logger.info(f"Final effective use_neo_riemannian: {self.use_neo_riemannian}")

        # --- FORCE DISABLE problematic features --- #
        if self.use_roman_numerals:
             self.logger.warning("Temporarily forcing use_roman_numerals to False due to potential music21 errors.")
             self.use_roman_numerals = False
             self.config.additional_params['use_roman_numerals'] = False
        if self.use_neo_riemannian:
             self.logger.warning("Temporarily forcing use_neo_riemannian to False due to potential music21 errors.")
             self.use_neo_riemannian = False
             self.config.additional_params['use_neo_riemannian'] = False
        # --- END FORCE DISABLE ---

        # <<< Re-add chord analyzer assignment >>>
        # Assign after config is settled, using the function passed in __init__ if provided
        # If a custom function is passed, we can't easily pass the flags, so we only
        # modify the default case using extract_chord_events.
        if chord_analysis_func is not None:
             self.chord_analyzer = chord_analysis_func # Use the custom one directly
             self.logger.warning("Custom chord_analysis_func provided; cannot automatically pass analysis flags.")
        else:
             # Use functools.partial to preset the flags for the default extract_chord_events
             # The flags used here (self.use_roman_numerals etc.) are the potentially
             # modified ones (e.g., forced to False earlier in __init__)
             self.chord_analyzer = functools.partial(extract_chord_events,
                                                     analyze_roman_numerals=self.use_roman_numerals,
                                                     analyze_neo_riemannian=self.use_neo_riemannian)
             self.logger.info(f"Using default chord analyzer (extract_chord_events) with "
                              f"analyze_roman_numerals={self.use_roman_numerals}, "
                              f"analyze_neo_riemannian={self.use_neo_riemannian}")


        # CC Instrument Map Initialization (remains the same logic as before)
        self.cc_instrument_map: Dict[str, Set[int]] = {}
        instrument_types_for_cc = [
             "piano", "chromatic percussion", "organ", "guitar", "bass", "strings",
             "ensemble", "brass", "reed", "pipe", "synth lead", "synth pad",
             "synth effects", "ethnic", "percussive", "sound effects", "drum kit"
        ]
        # Use the potentially updated self attributes for defaults here
        default_relevant_ccs = get_relevant_ccs_for_instrument("unknown", extended=True)
        for inst_type in instrument_types_for_cc:
            self.cc_instrument_map[inst_type] = set(get_relevant_ccs_for_instrument(inst_type, extended=True))
        self.cc_instrument_map["unknown"] = set(default_relevant_ccs)
        self.cc_instrument_map["drum kit"] = set(get_relevant_ccs_for_instrument("percussion", extended=True))

        # Internal State
        self._current_track_instrument_type: str = "unknown"
        self._current_track_program: int = 0
        # Store valid chord qualities (use the imported constant)
        self.valid_chord_qualities = TOKENIZER_EXPECTED_QUALITIES.copy()

    def _tweak_config_before_creating_voc(self) -> None:
        # In case the tokenizer has been created without specifying any config or
        # params file path
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = None
        if "use_bar_end_tokens" not in self.config.additional_params:
            self.config.additional_params["use_bar_end_tokens"] = USE_BAR_END_TOKENS
        if "add_trailing_bars" not in self.config.additional_params:
            self.config.additional_params["add_trailing_bars"] = ADD_TRAILING_BARS

    def _compute_ticks_per_pos(self, ticks_per_beat: int) -> int:
        return ticks_per_beat // self.config.max_num_pos_per_beat

    def _compute_ticks_per_units(
        self, time: int, current_time_sig: Sequence[int], time_division: int
    ) -> tuple[int, int, int]:
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(time, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
        return ticks_per_bar, ticks_per_beat, ticks_per_pos

    def _add_new_bars(
        self,
        until_time: int,
        event_type: str,
        all_events: Sequence[Event],
        current_bar: int,
        bar_at_last_ts_change: int,
        tick_at_last_ts_change: int,
        tick_at_current_bar: int,
        current_time_sig: tuple[int, int],
        ticks_per_bar: int,
    ) -> tuple[int, int]:
        num_new_bars = (
            bar_at_last_ts_change
            + self._units_between(tick_at_last_ts_change, until_time, ticks_per_bar)
            - current_bar
        )
        for i in range(num_new_bars):
            current_bar += 1
            tick_at_current_bar = (
                tick_at_last_ts_change
                + (current_bar - bar_at_last_ts_change) * ticks_per_bar
            )
            if self.config.additional_params["use_bar_end_tokens"] and current_bar > 0:
                all_events.append(
                    Event(
                        type_="Bar",
                        value="End",
                        time=tick_at_current_bar - 1,
                        desc=0,
                    )
                )
            all_events.append(
                Event(
                    type_="Bar",
                    value=str(current_bar + i)
                    if self.config.additional_params["max_bar_embedding"] is not None
                    else "None",
                    time=tick_at_current_bar,
                    desc=0,
                )
            )
            # Add a TimeSignature token, except for the last new Bar token
            # if the current event is a TS
            if self.config.use_time_signatures and not (
                event_type == "TimeSig" and i + 1 == num_new_bars
            ):
                all_events.append(
                    Event(
                        type_="TimeSig",
                        value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        time=tick_at_current_bar,
                        desc=0,
                    )
                )
        return current_bar, tick_at_current_bar

    def _add_position_event(
        self,
        event: Event,
        all_events: list[Event],
        tick_at_current_bar: int,
        ticks_per_pos: int,
    ) -> None:
        pos_index = self._units_between(tick_at_current_bar, event.time, ticks_per_pos)
        # if event in TOKEN_TYPES_AC
        all_events.append(
            Event(
                type_="Position",
                value=pos_index,
                time=event.time,
                desc=event.time,
            )
        )

    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        current_bar = -1
        bar_at_last_ts_change = 0
        previous_tick = -1
        previous_note_end = 0
        tick_at_last_ts_change = tick_at_current_bar = 0

        # Determine time signature and compute ticks per entites
        current_time_sig, time_sig_time = TIME_SIGNATURE, 0

        # First look for a TimeSig token, if any is given at tick 0, to update
        # current_time_sig
        if self.config.use_time_signatures:
            for event in events:
                # There should be a TimeSig token at tick 0
                if event.type_ == "TimeSig":
                    current_time_sig = self._parse_token_time_signature(event.value)
                    time_sig_time = event.time
                    break
                if event.type_ in [
                    "Pitch",
                    "PitchDrum",
                    "Velocity",
                    "Duration",
                    "PitchBend",
                    "Pedal",
                ]:
                    break
        ticks_per_bar, ticks_per_beat, ticks_per_pos = self._compute_ticks_per_units(
            time_sig_time, current_time_sig, time_division
        )

        # Add the time events
        for ei, event in enumerate(events):
            if event.type_.startswith("ACTrack"):
                all_events.append(event)
                continue
            if event.time != previous_tick:
                # (Rest)
                if (
                    self.config.use_rests
                    and event.time - previous_note_end >= self._min_rest(ticks_per_beat)
                ):
                    previous_tick = previous_note_end
                    rest_values = self._time_ticks_to_tokens(
                        event.time - previous_tick, ticks_per_beat, rest=True
                    )
                    # Add Rest events and increment previous_tick
                    for dur_value, dur_ticks in zip(*rest_values):
                        all_events.append(
                            Event(
                                type_="Rest",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{event.time - previous_tick} ticks",
                            )
                        )
                        previous_tick += dur_ticks
                    # We update current_bar and tick_at_current_bar here without
                    # creating Bar tokens
                    real_current_bar = bar_at_last_ts_change + self._units_between(
                        tick_at_last_ts_change, previous_tick, ticks_per_bar
                    )
                    if real_current_bar > current_bar:
                        # In case we instantly begin with a Rest,
                        # we need to update current_bar
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (
                            real_current_bar - current_bar
                        ) * ticks_per_bar
                        current_bar = real_current_bar

                # Bar
                current_bar, tick_at_current_bar = self._add_new_bars(
                    event.time,
                    event.type_,
                    all_events,
                    current_bar,
                    bar_at_last_ts_change,
                    tick_at_last_ts_change,
                    tick_at_current_bar,
                    current_time_sig,
                    ticks_per_bar,
                )

                # Position
                if event.type_ != "TimeSig" and not event.type_.startswith("ACBar"):
                    self._add_position_event(
                        event, all_events, tick_at_current_bar, ticks_per_pos
                    )

                previous_tick = event.time

            # Update time signature time variables, after adjusting the time (above)
            if event.type_ == "TimeSig":
                bar_at_last_ts_change += self._units_between(
                    tick_at_last_ts_change, event.time, ticks_per_bar
                )
                tick_at_last_ts_change = event.time
                current_time_sig = self._parse_token_time_signature(event.value)
                ticks_per_bar, ticks_per_beat, ticks_per_pos = (
                    self._compute_ticks_per_units(
                        event.time, current_time_sig, time_division
                    )
                )
                # We decrease the previous tick so that a Position token is enforced
                # for the next event
                previous_tick -= 1

            all_events.append(event)
            # Adds a Position token if the current event is a bar-level attribute
            # control and the next one is at the same position, as the position token
            # wasn't added previously.
            if (
                event.type_.startswith("ACBar")
                and not events[ei + 1].type_.startswith("ACBar")
                and event.time == events[ei + 1].time
            ):
                self._add_position_event(
                    event, all_events, tick_at_current_bar, ticks_per_pos
                )

            # Update max offset time of the notes encountered
            previous_note_end = self._previous_note_end_update(event, previous_note_end)

        if (
            previous_note_end > previous_tick
            and self.config.additional_params["add_trailing_bars"]
        ):
            # there are some trailing bars
            _ = self._add_new_bars(
                previous_note_end,
                event.type_,
                all_events,
                current_bar,
                bar_at_last_ts_change,
                tick_at_last_ts_change,
                tick_at_current_bar,
                current_time_sig,
                ticks_per_bar,
            )
        return all_events

    @staticmethod
    def _previous_note_end_update(event: Event, previous_note_end: int) -> int:
        r"""
        Calculate max offset time of the notes encountered.

        uses Event field specified by event.type_ .
        """
        event_time = 0
        if event.type_ in {
            "Pitch",
            "PitchDrum",
            "PitchIntervalTime",
            "PitchIntervalChord",
        }:
            event_time = event.desc
        elif event.type_ in {
            "Program",
            "Tempo",
            "TimeSig",
            "Pedal",
            "PedalOff",
            "PitchBend",
            "Chord",
        }:
            event_time = event.time
        return max(previous_note_end, event_time)

    @staticmethod
    def _units_between(start_tick: int, end_tick: int, ticks_per_unit: int) -> int:
        return (end_tick - start_tick) // ticks_per_unit

    def _tokens_to_score(self, tokens: TokSequence | list[TokSequence], programs: list[tuple[int, bool]] | None = None) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i, tokens_i in enumerate(tokens):
            tokens[i] = tokens_i.tokens
        score = Score(self.time_division)
        dur_offset = 2 if self.config.use_velocities else 1

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes, time_signature_changes = [], []
        chord_lyrics = [] # List to store chord lyrics events

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        current_track = None
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0:
                if self.config.use_time_signatures:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type == "TimeSig":
                            time_signature_changes.append(
                                TimeSignature(
                                    0, *self._parse_token_time_signature(tok_val)
                                )
                            )
                            break
                        if tok_type in [
                            "Pitch",
                            "PitchDrum",
                            "Velocity",
                            "Duration",
                            "PitchBend",
                            "Pedal",
                        ]:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[-1]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, score.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)

            # Set tracking variables
            current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
            current_bar = -1
            bar_at_last_ts_change = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
            active_pedals = {}
            # Variables to reconstruct chord lyrics
            pending_chord_root = None
            pending_chord_time = -1

            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type.startswith("Program"):
                            current_program = int(tok_val)
                            if current_program == -1:
                                is_drum, current_program = True, 0
                            break
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )
            current_track_use_duration = (
                current_program in self.config.use_note_duration_programs
            )

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")

                # Reset pending chord if a new event occurs at a later time
                if pending_chord_root is not None and current_tick > pending_chord_time:
                    pending_chord_root = None
                    pending_chord_time = -1

                if token == "Bar_None":
                    current_bar += 1
                    if current_bar > 0:
                        current_tick = tick_at_current_bar + ticks_per_bar
                    tick_at_current_bar = current_tick
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                    real_current_bar = bar_at_last_ts_change + self._units_between(
                        tick_at_last_ts_change, current_tick, ticks_per_bar
                    )
                    if real_current_bar > current_bar:
                        # In case we instantly begin with a Rest,
                        # we need to update current_bar
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (
                            real_current_bar - current_bar
                        ) * ticks_per_bar
                        current_bar = real_current_bar
                elif tok_type == "Position":
                    if current_bar == -1:
                        # as this Position token occurs before any Bar token
                        current_bar = 0
                    current_tick = tick_at_current_bar + int(tok_val) * ticks_per_pos
                elif tok_type in {
                    "Pitch",
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                }:
                    if tok_type in {"Pitch", "PitchDrum"}:
                        pitch = int(tok_val)
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                    else:  # PitchIntervalChord
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                    if (
                        not self.config.pitch_range[0]
                        <= pitch
                        <= self.config.pitch_range[1]
                    ):
                        continue

                    # We update previous_pitch_onset and previous_pitch_chord even if
                    # the try fails.
                    if tok_type != "PitchIntervalChord":
                        previous_pitch_onset[current_program] = pitch
                    previous_pitch_chord[current_program] = pitch

                    try:
                        if self.config.use_velocities:
                            vel_type, vel = seq[ti + 1].split("_")
                        else:
                            vel_type, vel = "Velocity", DEFAULT_VELOCITY
                        if current_track_use_duration:
                            dur_type, dur = seq[ti + dur_offset].split("_")
                        else:
                            dur_type = "Duration"
                            dur = int(
                                self.config.default_note_duration * ticks_per_beat
                            )
                        if vel_type == "Velocity" and dur_type == "Duration":
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            new_note = Note(
                                current_tick,
                                dur,
                                pitch,
                                int(vel),
                            )
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                            previous_note_end = max(
                                previous_note_end, current_tick + dur
                            )
                    except IndexError:
                        # A well constituted sequence should not raise an exception
                        # However with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass
                elif tok_type == "Program":
                    current_program = int(tok_val)
                    current_track_use_duration = (
                        current_program in self.config.use_note_duration_programs
                    )
                    if (
                        not self.config.one_token_stream_for_programs
                        and self.config.program_changes
                    ):
                        if current_program != -1:
                            current_track.program = current_program
                        else:
                            current_track.program = 0
                            current_track.is_drum = True
                elif tok_type == "Tempo":
                    if si == 0:
                        tempo_changes.append(Tempo(current_tick, float(tok_val)))
                    previous_note_end = max(previous_note_end, current_tick)
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if (
                        num != current_time_sig.numerator
                        or den != current_time_sig.denominator
                    ):
                        current_time_sig = TimeSignature(current_tick, num, den)
                        if si == 0:
                            time_signature_changes.append(current_time_sig)
                        tick_at_last_ts_change = tick_at_current_bar  # == current_tick
                        bar_at_last_ts_change = current_bar
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, score.ticks_per_quarter
                        )
                        ticks_per_beat = self._tpb_per_ts[den]
                        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                                seq[ti + 1].split("_")[1]
                            ]
                            # Add instrument if it doesn't exist, can happen for the
                            # first tokens
                            new_pedal = Pedal(current_tick, duration)
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_prog not in active_pedals:
                        active_pedals[pedal_prog] = current_tick
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        new_pedal = Pedal(
                            active_pedals[pedal_prog],
                            current_tick - active_pedals[pedal_prog],
                        )
                        if self.config.one_token_stream_for_programs:
                            check_inst(pedal_prog)
                            tracks[pedal_prog].pedals.append(new_pedal)
                        else:
                            current_track.pedals.append(new_pedal)
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.config.one_token_stream_for_programs:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_track.pitch_bends.append(new_pitch_bend)

                # Handle GlobalChord* tokens
                elif tok_type == "GlobalChordRoot":
                    pending_chord_root = tok_val
                    pending_chord_time = current_tick
                    # We don't reset here if another root comes immediately

                elif tok_type == "GlobalChordQual":
                    if pending_chord_root is not None and current_tick == pending_chord_time:
                        try:
                            root_idx = int(pending_chord_root)
                            if 0 <= root_idx < 12:
                                root_name = PITCH_CLASS_NAMES[root_idx]
                                lyric_text = f"{root_name}:{tok_val}"
                                lyric = symusic.TextMeta(time=current_tick, text=lyric_text)
                                chord_lyrics.append(lyric)
                            else:
                                # Log warning for invalid root index
                                pass
                        except (ValueError, IndexError) as e:
                            # Log warning if conversion fails
                            pass
                        finally:
                            # Always reset after processing or attempting to process a Qual token
                            pending_chord_root = None
                            pending_chord_time = -1
                    else:
                        # Qual appeared without a matching Root at the same time, reset
                        pending_chord_root = None
                        pending_chord_time = -1

                # Implicitly reset pending chord if any other token type appears
                elif tok_type not in ["GlobalChordRoot", "GlobalChordQual"]:
                     if pending_chord_root is not None:
                         # This case should be covered by the time check at the start of the loop,
                         # but added for extra safety.
                         pending_chord_root = None
                         pending_chord_time = -1

            # Add current_inst to score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        # Add global events to the score
        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes

        # Add collected chord lyrics to the first track if it exists
        if score.tracks:
            # Sort lyrics by time just in case they were added out of order
            # although current_tick should maintain order
            chord_lyrics.sort(key=lambda x: x.time)
            score.tracks[0].lyrics.extend(chord_lyrics)

        return score

    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        This version adds TrackName, GlobalChord*, and CC* tokens.

        :return: the vocabulary as a list of string.
        """
        print("\n=== Creating Base Vocabulary ===")
        vocab = []

        # --- DEBUG: Check additional_params before using them ---
        config_params = self.config.additional_params
        print(f"DEBUG [VocabBuild]: self.config.additional_params = {config_params}")
        if config_params is None:
             print("ERROR [VocabBuild]: additional_params is None! Using defaults.")
             config_params = {} # Use empty dict to avoid error, but log the issue

        # Bar
        if config_params.get("max_bar_embedding") is not None:
            vocab += [
                f"Bar_{i}"
                for i in range(config_params["max_bar_embedding"])
            ]
        else:
            vocab += ["Bar_None"]
        if config_params.get("use_bar_end_tokens", False):
            vocab.append("Bar_End")

        # NoteOn/NoteOff/Velocity/Duration (handled by parent method)
        super()._add_note_tokens_to_vocab_list(vocab)

        # Position
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab += [f"Position_{i}" for i in range(num_positions)]

        # Add additional tokens (Tempo, Chord, Rest, Program) handled by parent
        super()._add_additional_tokens_to_vocab_list(vocab)

        # --- Add Custom Tokens --- #
        # Read flags directly from config, as self attributes might not be set yet
        config_params = self.config.additional_params
        print(f"DEBUG [VocabBuild Custom]: Reading flags from config_params: {config_params}") # DEBUG

        _use_track_names = config_params.get('use_track_names', False)
        _use_global_chords = config_params.get('use_global_chords', False)
        _use_filtered_ccs = config_params.get('use_filtered_ccs', False)
        print(f"DEBUG [VocabBuild Custom]: use_track_names={_use_track_names}, use_global_chords={_use_global_chords}, use_filtered_ccs={_use_filtered_ccs}") # DEBUG

        # TrackName (Add tokens for predefined families)
        if _use_track_names:
            vocab += [f"TrackName_{family}" for family in TARGET_INSTRUMENT_FAMILIES]

        # Global Chords
        if _use_global_chords:
            chord_roots = [str(i) for i in range(12)]
            # Define canonical chord qualities using the imported constant
            chord_qualities = list(TOKENIZER_EXPECTED_QUALITIES)
            self.valid_chord_qualities = set(chord_qualities) # Store for validation (self is available here)
            vocab += [f"GlobalChordRoot_{r}" for r in chord_roots]
            vocab += [f"GlobalChordQual_{q}" for q in chord_qualities]

        # Filtered CCs
        if _use_filtered_ccs:
            num_cc_bins = config_params.get('num_cc_bins', 128) # Get bins from config
            # CC Types (0-127)
            vocab += [f"CCType_{i}" for i in range(128)]
            # CC Values (Binned)
            vocab += [f"CCValue_{i}" for i in range(num_cc_bins)]

        # <<< Add Roman Numeral Tokens >>>
        _use_roman_numerals = config_params.get('use_roman_numerals', True)
        if _use_roman_numerals:
            # Use the predefined set, assuming figures are already sanitized
            vocab += [f"RomanNumeral_{figure}" for figure in COMMON_ROMAN_NUMERALS]
            vocab.append("RomanNumeral_None") # Add a token for cases where analysis fails

        # <<< Add Neo-Riemannian Tokens >>>
        _use_neo_riemannian = config_params.get('use_neo_riemannian', True)
        if _use_neo_riemannian:
            vocab += [f"NeoRiemannian_{trans}" for trans in COMMON_NEO_RIEMANNIAN]
            vocab.append("NeoRiemannian_None") # Add a token for no transformation

        # --- End Custom Tokens --- #

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        This version includes transitions for TrackName, GlobalChord*, and CC* tokens.

        :return: the token types transitions dictionary.
        """
        dic: dict[str, set[str]] = {}

        # Standard REMI token types
        if self.config.use_programs:
            first_note_token_type = (
                "Pitch" if self.config.program_changes else "Program"
            )
            dic["Program"] = {"Pitch"}
        else:
            first_note_token_type = "Pitch"

        dic["Bar"] = {"Position"}
        if self.config.additional_params["use_bar_end_tokens"]:
             # Bar_End only used in _add_time_events(), not planned in transitions
             # dic["Bar_End"] = {"Bar"}
             # If Bar_End is used, Bar can be followed by Bar_End
             # dic["Bar"].add("Bar_End") # This depends on exact implementation
             pass # Assuming Bar_End is handled implicitly or not part of graph logic

        dic["Position"] = {first_note_token_type}

        note_like_tokens = {first_note_token_type, "Position", "Bar"}

        if self.config.use_velocities:
            dic["Pitch"] = {"Velocity"}
            dic["Velocity"] = ({"Duration"} if self.config.using_note_duration_tokens else note_like_tokens.copy())
        elif self.config.using_note_duration_tokens:
            dic["Pitch"] = {"Duration"}
        else:
            dic["Pitch"] = note_like_tokens.copy()

        if self.config.using_note_duration_tokens:
            dic["Duration"] = note_like_tokens.copy()

        # Add standard additional tokens to the graph
        if self.config.use_pitch_intervals: self._add_pitch_interval_rules_to_graph(dic, first_note_token_type, note_like_tokens)
        if self.config.program_changes: self._add_program_change_rules_to_graph(dic, first_note_token_type)
        if self.config.use_chords: self._add_chord_rules_to_graph(dic, first_note_token_type)
        if self.config.use_tempos: self._add_tempo_rules_to_graph(dic, first_note_token_type)
        if self.config.use_time_signatures: self._add_time_signature_rules_to_graph(dic, first_note_token_type)
        if self.config.use_sustain_pedals: self._add_pedal_rules_to_graph(dic, first_note_token_type)
        if self.config.use_pitch_bends: self._add_pitch_bend_rules_to_graph(dic, first_note_token_type)
        if self.config.use_rests: self._add_rest_rules_to_graph(dic, first_note_token_type)
        if self.config.use_pitchdrum_tokens: self._add_pitch_drum_rules_to_graph(dic)

        # --- Add Custom Token Transitions --- #
        base_musical_event_starters = {first_note_token_type, "Position", "Bar"}
        if self.config.use_rests: base_musical_event_starters.add("Rest")
        if self.config.use_tempos: base_musical_event_starters.add("Tempo")
        if self.config.use_time_signatures: base_musical_event_starters.add("TimeSig") # Though chords/trackname might follow TS better
        if self.config.use_pitch_bends: base_musical_event_starters.add("PitchBend")

        # --- Read flags from config ---
        config_params = self.config.additional_params
        _use_track_names = config_params.get('use_track_names', False) # Default False if not found
        _use_global_chords = config_params.get('use_global_chords', False)
        _use_filtered_ccs = config_params.get('use_filtered_ccs', False)
        # --- End read flags ---

        can_precede_track_name = {"Bar", "Position", "TimeSig", "Tempo", "Rest"} # Events that can come before a track name
        can_follow_track_name = {first_note_token_type, "Position"} # Events that can come after a track name
        if self.config.use_programs: can_follow_track_name.add("Program") # Program usually follows track name

        can_precede_global_chord = {"Bar", "TimeSig"} # Events that can come before global chord root
        can_follow_global_chord = {first_note_token_type, "Position"} # Events that can come after global chord quality

        can_precede_cc = { # Events that can come before CCType
             "Pitch", "Velocity", "Duration", "Rest", "GlobalChordQual",
             "Position" # Allows CCs at start of bar/pos
        }
        if self.config.use_pitch_bends: can_precede_cc.add("PitchBend")

        can_follow_cc = { # Events that can come after CCValue
             "Pitch", "Velocity", "Duration", "Rest",
             "Position", "Bar", # Allows moving time/bar after CC
             "CCType", # Allow consecutive CCs
             "GlobalChordRoot", # Allow new chord after CC
        }
        if self.config.use_pitch_bends: can_follow_cc.add("PitchBend")


        if _use_track_names: # Use the local variable read from config
            dic["TrackName"] = can_follow_track_name
            for key in can_precede_track_name:
                 if key in dic: # Ensure key exists (e.g., Rest might be off)
                     dic[key].add('TrackName')
                 elif key not in dic and key != "TrackName": # Initialize if missing, except self-loops
                      dic[key] = {'TrackName'}
            # Ensure TrackName is a key if not added above
            if "TrackName" not in dic: dic["TrackName"] = set()

        if _use_global_chords: # Use the local variable
            dic["GlobalChordRoot"] = {"GlobalChordQual"}
            dic["GlobalChordQual"] = can_follow_global_chord.copy() # What can follow a chord quality
            if _use_filtered_ccs: dic["GlobalChordQual"].add("CCType") # CC can follow chord qual

            for key in can_precede_global_chord:
                 if key in dic:
                     dic[key].add('GlobalChordRoot')
                 elif key not in dic and key != "GlobalChordRoot":
                      dic[key] = {'GlobalChordRoot'}
            # Ensure GlobalChord keys exist
            if "GlobalChordRoot" not in dic: dic["GlobalChordRoot"] = set()
            if "GlobalChordQual" not in dic: dic["GlobalChordQual"] = set()

        if _use_filtered_ccs: # Use the local variable
            dic["CCType"] = {"CCValue"}
            dic["CCValue"] = can_follow_cc.copy()

            for key in can_precede_cc:
                 if key in dic:
                     dic[key].add('CCType')
                 elif key not in dic and key != "CCType": # Initialize if missing
                     dic[key] = {'CCType'}

            # Ensure CC keys exist
            if "CCType" not in dic: dic["CCType"] = set()
            if "CCValue" not in dic: dic["CCValue"] = set()

            # Allow CCType to follow CCValue for consecutive CCs
            if "CCValue" in dic: dic["CCValue"].add("CCType")

        # --- Final cleanup: Ensure all custom types used are keys in the graph --- #
        custom_types = []
        if _use_track_names: custom_types.append("TrackName") # Use local variable
        if _use_global_chords: custom_types.extend(["GlobalChordRoot", "GlobalChordQual"]) # Use local variable
        if _use_filtered_ccs: custom_types.extend(["CCType", "CCValue"]) # Use local variable

        for type_name in custom_types:
             if type_name not in dic:
                 dic[type_name] = set()
        # --- End Custom Token Transitions --- #

        return dic

    # --- Helper methods for graph creation (copied from MusicTokenizer) ---
    # These are needed because the main method calls them. Alternatively, refactor
    # the main method to not call these helpers, but copying is easier for now.
    def _add_pitch_interval_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str, note_like_tokens: set[str]) -> None:
        for token_type in ("PitchIntervalTime", "PitchIntervalChord"):
            dic[token_type] = (
                {"Velocity"}
                if self.config.use_velocities
                else {"Duration"}
                if self.config.using_note_duration_tokens
                else note_like_tokens.copy() | {"PitchIntervalTime", "PitchIntervalChord"}
            )
            if self.config.use_programs and self.config.one_token_stream_for_programs:
                if "Program" in dic: dic["Program"].add(token_type)
                else: dic["Program"] = {token_type}
            else:
                # Add interval token to lists of possible next tokens for relevant types
                for key in ["Pitch", "Velocity", "Duration", "Position"]:
                    if key in dic: dic[key].add(token_type)

    def _add_program_change_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        key = (
            "Duration" if self.config.using_note_duration_tokens
            else "Velocity" if self.config.use_velocities
            else first_note_token_type
        )
        if key in dic: dic[key].add("Program")
        else: dic[key] = {"Program"}

        if self.config.additional_params.get("use_bar_end_tokens", False):
            if "Program" in dic: dic["Program"].add("Bar")
            else: dic["Program"] = {"Bar"}
        # Add program to allowed followers of structural tokens if program changes allowed
        for structural_token in ["Position", "Rest", "Tempo", "TimeSig", "Bar"]:
             if structural_token in dic:
                 dic[structural_token].add("Program")
                 if "Program" in dic: dic["Program"].add(structural_token)
                 else: dic["Program"] = {structural_token}

    def _add_chord_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        dic["Chord"] = {first_note_token_type}
        for key in ["Position", "Program", "Tempo", "TimeSig", "Rest"]:
            if key in dic: dic[key].add("Chord")
        if self.config.use_pitch_intervals:
             if "Chord" in dic: dic["Chord"] |= {"PitchIntervalTime", "PitchIntervalChord"}
             else: dic["Chord"] = {"PitchIntervalTime", "PitchIntervalChord"}

    def _add_tempo_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        dic["Tempo"] = {first_note_token_type, "Position", "Bar"}
        for key in ["Position", "Program", "Duration", "Velocity", "Pitch", "Rest"]:
            if key in dic: dic[key].add("Tempo")

        if self.config.use_chords: dic["Tempo"].add("Chord")
        if self.config.use_rests: dic["Tempo"].add("Rest")
        if self.config.use_pitch_intervals: dic["Tempo"] |= {"PitchIntervalTime", "PitchIntervalChord"}
        if self.config.use_time_signatures: dic["Tempo"].add("TimeSig")

    def _add_time_signature_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        # Assuming TimeSig only follows Bar for simplicity, adjust if needed
        if "Bar" in dic: dic["Bar"].add("TimeSig")
        else: dic["Bar"] = {"TimeSig"}

        dic["TimeSig"] = {first_note_token_type, "Position", "Bar"}
        if self.config.use_chords: dic["TimeSig"].add("Chord")
        if self.config.use_rests: dic["TimeSig"].add("Rest")
        if self.config.use_pitch_intervals: dic["TimeSig"] |= {"PitchIntervalTime", "PitchIntervalChord"}
        if self.config.use_tempos: dic["Tempo"].add("TimeSig") # Allow Tempo -> TimeSig

    def _add_pedal_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        pedal_starts = {"Position", "Rest", "Tempo", "TimeSig"}
        pedal_ends = {first_note_token_type, "Position", "Bar", "Pedal"}

        if self.config.sustain_pedal_duration:
            dic["Pedal"] = {"Duration"}
            # Allow Pedal to start after note-related events or structure
            for key in {"Pitch", "Velocity", "Duration"}.union(pedal_starts):
                 if key in dic: dic[key].add("Pedal")
            # Duration must be able to precede Pedal if used
            if "Duration" in dic: dic["Duration"].add("Pedal")
            else: dic["Duration"] = {"Pedal"}
        else:
            dic["Pedal"] = pedal_ends.copy()
            dic["PedalOff"] = pedal_ends.copy() | {"PedalOff"}
            dic["Pedal"].add("PedalOff") # Pedal can be followed by PedalOff
            for key in pedal_starts:
                if key in dic:
                     dic[key].add("Pedal")
                     dic[key].add("PedalOff")

            # Allow Pedal/PedalOff after note-related events
            for note_key in ["Pitch", "Velocity", "Duration"]:
                 if note_key in dic:
                     dic[note_key].add("Pedal")
                     dic[note_key].add("PedalOff")

        # Tokens that can follow a rest
        if self.config.use_chords:
            dic["Pedal"].add("Chord")
            if not self.config.sustain_pedal_duration:
                 dic["PedalOff"].add("Chord")
                 if "Chord" in dic: dic["Chord"].update(["Pedal", "PedalOff"])
                 else: dic["Chord"] = {"Pedal", "PedalOff"}
            elif "Chord" in dic: dic["Chord"].add("Pedal")
            else: dic["Chord"] = {"Pedal"}

        if self.config.use_rests:
            dic["Pedal"].add("Rest")
            if not self.config.sustain_pedal_duration:
                 dic["PedalOff"].add("Rest")

        if self.config.use_pitch_intervals:
            intervals = {"PitchIntervalTime", "PitchIntervalChord"}
            if self.config.sustain_pedal_duration:
                if "Duration" in dic: dic["Duration"].update(intervals)
                else: dic["Duration"] = intervals.copy()
            else:
                dic["Pedal"].update(intervals)
                dic["PedalOff"].update(intervals)

    def _add_pitch_bend_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        dic["PitchBend"] = {first_note_token_type, "Position", "Bar"}
        # Allow PitchBend after various events
        pb_preceders = {"Position", "Tempo", "TimeSig", "Rest", "Pitch", "Velocity", "Duration"}
        if self.config.use_sustain_pedals:
            pb_preceders.add("Pedal")
            if not self.config.sustain_pedal_duration: pb_preceders.add("PedalOff")

        if self.config.use_programs and not self.config.program_changes:
            if "Program" in dic: dic["Program"].add("PitchBend")
            else: dic["Program"] = {"PitchBend"}
        else:
             for key in pb_preceders:
                 if key in dic: dic[key].add("PitchBend")

        if self.config.use_chords: dic["PitchBend"].add("Chord")
        if self.config.use_rests: dic["PitchBend"].add("Rest")

    def _add_rest_rules_to_graph(self, dic: dict[str, set[str]], first_note_token_type: str) -> None:
        dic["Rest"] = {"Rest", first_note_token_type, "Position", "Bar"}
        rest_preceders = {
            "Duration" if self.config.using_note_duration_tokens
            else "Velocity" if self.config.use_velocities
            else first_note_token_type,
            "Rest", "Bar", "Position", "TimeSig", "Tempo"
        }
        if self.config.use_sustain_pedals:
             rest_preceders.add("Pedal")
             if not self.config.sustain_pedal_duration: rest_preceders.add("PedalOff")
        if self.config.use_pitch_bends: rest_preceders.add("PitchBend")

        for key in rest_preceders:
            if key in dic: dic[key].add("Rest")

        # Tokens that can follow a rest
        if self.config.use_chords: dic["Rest"].add("Chord")
        if self.config.use_tempos: dic["Rest"].add("Tempo")
        if self.config.use_time_signatures: dic["Rest"].add("TimeSig")
        if self.config.use_sustain_pedals:
             dic["Rest"].add("Pedal")
             if not self.config.sustain_pedal_duration: dic["Rest"].add("PedalOff")
        if self.config.use_pitch_intervals: dic["Rest"] |= {"PitchIntervalTime", "PitchIntervalChord"}

    def _add_pitch_drum_rules_to_graph(self, dic: dict[str, set[str]]) -> None:
        if "Pitch" in dic: # Ensure Pitch key exists
             dic["PitchDrum"] = dic["Pitch"].copy()
             for key, values in dic.items():
                 if "Pitch" in values:
                     dic[key].add("PitchDrum")
        else: # Handle case where Pitch might not be in graph? (Shouldn't happen for REMI)
             self.logger.warning("'Pitch' token type missing from graph during PitchDrum rule addition.")
             dic["PitchDrum"] = set() # Initialize empty 

    # --- Custom Event Generation Logic --- #

    def _score_to_events(self, score: Score) -> TokSequence:
        # <<< REMOVE DEBUG PRINT AT TOP >>>
        #print("!*!*!* EXECUTING StructuredREMI._score_to_events *!*!*!")
        #sys.stdout.flush()
        # <<< END REMOVED DEBUG >>>

        #print("\n=== Starting _score_to_events in StructuredREMI ===")

        # Sort tracks
        score.tracks.sort(key=lambda x: (x.program, x.is_drum))

        all_events: List[Event] = [] # Use a list to collect events

        # Global events (tempo, time sig, chords)
        if self.config.use_tempos:
            #print("\nProcessing tempo events...")
            tempo_events = self._tempo_events(score.tempos)
            #print(f"Generated {len(tempo_events)} tempo events")
            all_events += tempo_events

        if self.config.use_time_signatures:
            #print("\nProcessing time signature events...")
            ts_events = self._time_signature_events(score.time_signatures)
            #print(f"Generated {len(ts_events)} time signature events")
            all_events += ts_events

        if self.use_global_chords:
            #print("\n=== Processing Global Chords ===")
            chord_events_data = self.chord_analyzer(score)
            #print(f"Found {len(chord_events_data)} chord events")

            # Log chord distribution
            #chord_qualities = {}
            #chord_roots = {}
            filtered_chord_count = 0
            time_zero_skipped = 0
            for chord_data in chord_events_data:
                # Filter time-0 chords
                time = chord_data.get('time')

                # <<< Remove Checkpoint prints >>>
                #print("DEBUG: CHECKPOINT 1 - Inside loop")
                #sys.stdout.flush()
                #time = chord_data.get('time')
                #print("DEBUG: CHECKPOINT 2 - Got time value")
                #sys.stdout.flush()
                # <<< End Removed Prints >>>

                if time == 0:
                    #print(f"DEBUG: Skipping time 0 chord: {chord_data}")
                    #sys.stdout.flush()
                    time_zero_skipped += 1
                    continue # Skip chords exactly at time 0

                quality = chord_data.get('quality', 'other')
                if quality not in self.valid_chord_qualities:
                    #print(f"Remapping quality {quality} to 'other' (valid qualities: {sorted(list(self.valid_chord_qualities))})")
                    quality = 'other'
                root_pc = chord_data.get('root_pc')
                if root_pc is not None:
                    #print(f"Adding chord event at time {time}: root={root_pc}, quality={quality}")
                    root_event = GlobalChordRootEvent(root_pc=root_pc, time=time)
                    qual_event = GlobalChordQualEvent(quality=quality, time=time)
                    #print(f"Created events: {root_event}, {qual_event}")
                    all_events.extend([root_event, qual_event])
                    filtered_chord_count += 1
                #else:
                    #print(f"Skipping chord event due to missing root_pc or time: {chord_data}")
            print(f"Skipped {time_zero_skipped} chord events occurring exactly at time 0.")
            #print(f"Added {filtered_chord_count} chord events (Root+Qual pairs) to all_events.")

        # Track-specific events (notes, CCs, track names)
        #print("\n=== Starting Track Event Processing ===")
        total_track_event_count = 0
        for track_idx, track in enumerate(score.tracks):
            #print(f"\n--- Processing Track {track_idx} (Program: {track.program}, Name: {track.name}) ---")
            #sys.stdout.flush()

            #print(f"Program: {track.program}, Name: {track.name}")
            # Set current track program and type for CC filtering
            self._current_track_program = track.program
            mapped_family = map_track_name_to_family(track.name)
            self._current_track_instrument_type = get_instrument_type_from_program(track.program)
            #print(f"Mapped to family: {mapped_family}, instrument type: {self._current_track_instrument_type}")

            track_events = []
            if self.use_track_names:
                track_events.append(TrackNameEvent(
                    name=mapped_family,
                    instrument_type=self._current_track_instrument_type,
                    time=0
                ))

            program_events = self._program_events([track])
            #print(f"Generated {len(program_events)} program events")
            track_events += program_events

            note_events = self._note_events(track.notes)
            #print(f"Generated {len(note_events)} note events")
            track_events += note_events

            if self.config.use_sustain_pedals:
                pedal_events = self._pedal_events(track.pedals)
                #print(f"Generated {len(pedal_events)} pedal events")
                track_events += pedal_events

            if self.config.use_pitch_bends:
                pb_events = self._pitch_bend_events(track.pitch_bends)
                #print(f"Generated {len(pb_events)} pitch bend events")
                track_events += pb_events

            if self.use_filtered_ccs and track.controls:
                #print(f"\nProcessing {len(track.controls)} CC events")
                #print(f"Filter mode: {self.cc_filter_mode}")
                cc_events = self._create_control_change_events_filtered(track.controls)
                #print(f"Generated {len(cc_events)} CC events")
                track_events += cc_events
            else:
                print("--- DEBUG (StructuredREMI): Skipping filtered CCs ---")
        
            # Events should be sorted by time before being processed by _add_time_events.
            # MusicTokenizer._score_to_tokens handles sorting after collecting all events.
            # track_events.sort(key=lambda x: x.time) # Sorting here is redundant as _score_to_tokens sorts later
        
            print(f"--- DEBUG (StructuredREMI): Exiting _create_track_events for track '{track.name}', returning {len(track_events)} total events ---")
            all_events += track_events
            #print(f"Total events for track {track_idx}: {len(track_events)}")

        #print(f"\n=== Finished Processing All Tracks: Added {total_track_event_count} track events ===")
        #sys.stdout.flush()

        # --- Remove logs around sorting as this method might not be called ---
        #event_counts_before = defaultdict(int)
        #for event in all_events:
        #    event_counts_before[event.type_] += 1
        #print("\n=== Event Counts Before Sorting (StructuredREMI version) ===")
        #for event_type, count in sorted(event_counts_before.items()):
        #    print(f"  {event_type}: {count}")
        #sys.stdout.flush()

        # Sort all events by time, then priority, then type
        all_events.sort(key=lambda x: (x.time, self._event_priority.get(x.type_, 10), x.value))
        #print(f"\n=== Final Event Count (after collecting all, StructuredREMI version): {len(all_events)} ===")
        #sys.stdout.flush()

        #print("\n=== First 150 Events After Sorting (StructuredREMI version) ===")
        #for i, event in enumerate(all_events[:150]):
        #     print(f"  {i}: {event}")
        #if len(all_events) > 150:
        #     print(f"  ... ({len(all_events) - 150} more events)")
        #sys.stdout.flush()
        # --- END Remove logs --- 

        # Reset internal track state before processing (important if tokenizer is reused)
        self._current_track_instrument_type = "unknown"
        self._current_track_program = 0

        # Add time events (Bars, Positions, Rests)
        events_with_time = self._add_time_events(all_events, score.ticks_per_quarter)
        #print(f"Final event count after adding time events: {len(events_with_time)}")

        # Create TokSequence
        tok_sequence = TokSequence(events=events_with_time)

        # Add sos/eos tokens
        self.add_sos_eos_to_tokseq(tok_sequence)

        # BPE / Unigram encoding
        if self.has_bpe_model:
            tok_sequence.tokens = self._apply_bpe(tok_sequence.tokens)

        # Convert events to ids
        tok_sequence.convert_to_ids()

        # Print final token sequence
        #print("\n=== Final Token Sequence ===")
        #print("First 50 tokens:")
        #for i, token in enumerate(tok_sequence.tokens[:50]):
        #    print(f"{i}: {token}")
        #if len(tok_sequence.tokens) > 50:
        #    print(f"... ({len(tok_sequence.tokens) - 50} more tokens)")

        return tok_sequence

    def _create_control_change_events_filtered(self, controls: List[ControlChange]) -> List[Event]:
        events = []
        if not controls:
            #print("No CC events to process")
            return events

        allowed_ccs: Set[int]
        if self.cc_filter_mode == 'all':
            allowed_ccs = set(range(128))
            #print("Using 'all' CC filter mode - allowing all CCs")
        elif self.cc_filter_mode == 'explicit_list':
            allowed_ccs = self.explicit_cc_list
            #print(f"Using explicit CC list: {sorted(list(allowed_ccs))}")
        elif self.cc_filter_mode == 'instrument_aware':
            allowed_ccs = self.cc_instrument_map.get(self._current_track_instrument_type,
                                                     self.cc_instrument_map["unknown"])
            #print(f"Using instrument-aware CC filtering for type {self._current_track_instrument_type}")
            #print(f"Allowed CCs for this instrument: {sorted(list(allowed_ccs))}")
        else: # 'none' or invalid mode
            #print(f"Invalid CC filter mode '{self.cc_filter_mode}' - no CCs will be processed")
            return events

        #print(f"Processing {len(controls)} CC events")
        cc_types_seen = defaultdict(int)
        cc_values_seen = defaultdict(list)

        for control in controls:
            #print(f"Processing CC: number={control.number}, value={control.value}, time={control.time}")
            if control.number in allowed_ccs:
                cc_types_seen[control.number] += 1
                value = min(max(control.value, 0), 127)
                cc_values_seen[control.number].append(value)
                binned_value = int((value / 127.0) * (self.num_cc_bins - 1))
                #print(f"CC {control.number} value {value} binned to {binned_value} (using {self.num_cc_bins} bins)")
                events.append(CCTypeEvent(number=control.number, time=control.time))
                events.append(CCValueEvent(value=binned_value, time=control.time))
            #else:
                #print(f"Skipping CC {control.number} (not in allowed set)")

        #print("\nCC Type Usage Summary:")
        #for cc_type, count in sorted(cc_types_seen.items()):
        #    print(f"CC {cc_type}: {count} times")
        #    values = cc_values_seen[cc_type]
        #    print(f"  Value range: {min(values)}-{max(values)}, avg: {sum(values)/len(values):.1f}")

        #print(f"\nGenerated {len(events)} CC events (type+value pairs)")
        return events

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: np.ndarray,
        ticks_per_quarter: int,
        ticks_bars: Sequence[int] | None = None,
        ticks_beats: Sequence[int] | None = None,
        attribute_controls_indexes: Mapping[int, Sequence[int] | bool] | None = None,
    ) -> list[Event]:
        """Create track events by manually iterating through notes, pedals, bends, and adding filtered CCs."""
        print(f"\n--- DEBUG (StructuredREMI): Entering _create_track_events for track '{track.name}' (Notes: {len(track.notes)}, Pedals: {len(track.pedals)}, Bends: {len(track.pitch_bends)}, CCs: {len(track.controls)}) --- ")
        
        track_events = []
        program = track.program if not track.is_drum else -1
        # Note events (Generate only Pitch events)
        print(f"--- DEBUG (StructuredREMI): Generating Note events ({len(track.notes)} notes) ---")
        for note in track.notes:
            # We just need the basic Pitch event with time and duration (in desc).
            # Subsequent steps (_add_time_events) will handle Velocity/Duration tokens if needed by the config.
            track_events.append(
                Event(type_="Pitch", value=note.pitch, time=note.time, desc=note.duration)
            )
            # print(f"    Generated Pitch event at time {note.time}: {note.pitch} (dur: {note.duration})") # Optional debug print
        
        # Pedal events
        if self.config.use_sustain_pedals:
            print(f"--- DEBUG (StructuredREMI): Generating Pedal events ({len(track.pedals)} pedals) ---")
            for pedal in track.pedals:
                 # REMI uses Pedal tokens. Store duration in desc if sustain_pedal_duration is true
                 pedal_desc = pedal.duration if self.config.sustain_pedal_duration else 0
                 track_events.append(
                    # For REMI, Pedal value might indicate ON state, not duration. Need to confirm.
                    # Let's assume None for now and duration in desc if configured.
                    Event(type_="Pedal", value=None, time=pedal.time, desc=pedal_desc)
                 )
        
        # Pitch bend events
        if self.config.use_pitch_bends:
            print(f"--- DEBUG (StructuredREMI): Generating Pitch Bend events ({len(track.pitch_bends)} bends) ---")
            for bend in track.pitch_bends:
                track_events.append(
                    Event(type_="PitchBend", value=bend.value, time=bend.time)
                )
        
        # --- Add filtered CC events --- 
        if self.use_filtered_ccs and track.controls:
            print(f"--- DEBUG (StructuredREMI): Adding filtered CCs ({len(track.controls)} controls) ---")
            cc_events = self._create_control_change_events_filtered(track.controls)
            print(f"--- DEBUG (StructuredREMI): Generated {len(cc_events)} filtered CC events ---")
            track_events.extend(cc_events)
        else:
            print("--- DEBUG (StructuredREMI): Skipping filtered CCs ---")
        
        # Events should be sorted by time before being processed by _add_time_events.
        # MusicTokenizer._score_to_tokens handles sorting after collecting all events.
        # track_events.sort(key=lambda x: x.time) # Sorting here is redundant as _score_to_tokens sorts later
        
        print(f"--- DEBUG (StructuredREMI): Exiting _create_track_events for track '{track.name}', returning {len(track_events)} total events ---")
        return track_events

    def _create_global_events(self, score: Score) -> List[Event]:
        """Create global events including our custom chord events."""
        #print("\n=== Starting _create_global_events in StructuredREMI ===")

        global_events = super()._create_global_events(score)
        #print(f"Base global events: {len(global_events)}")

        if self.use_global_chords:
            #print("\nAnalyzing global chords...")
            #print(f"Chord analyzer: {self.chord_analyzer}")
            #print(f"Score time division: {score.ticks_per_quarter}")
            #print(f"Score tracks: {len(score.tracks)}")

            chord_events_data = self.chord_analyzer(score)
            #print(f"Found {len(chord_events_data)} chord events")

            # Log chord distribution
            #chord_qualities = {}
            #chord_roots = {}
            filtered_chord_count = 0
            time_zero_skipped = 0

            for chord_data in chord_events_data:
                #print("DEBUG: CHECKPOINT 1 - Inside loop")
                #sys.stdout.flush()
                time = chord_data.get('time')

                # <<< Remove Checkpoint prints >>>
                #print("DEBUG: CHECKPOINT 2 - Got time value")
                #sys.stdout.flush()
                # <<< End Removed Prints >>>

                if time == 0:
                    #print(f"DEBUG: Skipping time 0 chord: {chord_data}")
                    #sys.stdout.flush()
                    time_zero_skipped += 1
                    continue

                #print(f"\nProcessing chord: {chord_data}")
                quality = chord_data.get('quality', 'other')
                if quality not in self.valid_chord_qualities:
                    #print(f"Remapping quality {quality} to 'other' (valid qualities: {sorted(list(self.valid_chord_qualities))})")
                    quality = 'other'
                root_pc = chord_data.get('root_pc')
                if root_pc is not None and time is not None:
                    #print(f"Adding chord event at time {time}: root={root_pc}, quality={quality}")
                    root_event = GlobalChordRootEvent(root_pc=root_pc, time=time)
                    qual_event = GlobalChordQualEvent(quality=quality, time=time)
                    #print(f"Created events: {root_event}, {qual_event}")
                    global_events.extend([root_event, qual_event])
                    filtered_chord_count += 1
                #else:
                    #print(f"Skipping chord event due to missing root_pc or time: {chord_data}")
            print(f"\nSkipped {time_zero_skipped} chord events occurring exactly at time 0.")
            #print(f"Added {filtered_chord_count} chord events (Root+Qual pairs) to global_events.")

        #print(f"\nTotal global events: {len(global_events)}")
        return global_events

    def _sort_events(self, events: List[Event]) -> None:
        """Sort events by time, then by priority, then by type.

        This method is called by the parent class's _score_to_tokens method.

        Args:
            events: List of events to sort
        """
        # Define event priority (lower number = higher priority)
        event_priority = {
            "Tempo": 0,
            "TimeSig": 1,
            "Bar": 2,
            "Position": 3,
            "Program": 4,
            "TrackName": 5,
            "GlobalChordRoot": 6,
            "GlobalChordQual": 7,
            "CCType": 8,
            "CCValue": 9,
            "Pitch": 10,
            "Velocity": 11,
            "Duration": 12,
            "Pedal": 13,
            "PedalOff": 14,
            "PitchBend": 15,
            "Rest": 16,
        }

        # Sort events by time, then by priority, then by type
        events.sort(key=lambda e: (e.time, event_priority.get(e.type_, 100), e.type_))