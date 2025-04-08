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
)

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
from ..utils.structured_utils import (
    get_instrument_type_from_program,
    get_relevant_ccs_for_instrument,
    normalize_track_name,
    get_instrument_type_from_track_name,
    extract_chord_events,
    TOKENIZER_EXPECTED_QUALITIES,
    TARGET_INSTRUMENT_FAMILIES,
    map_track_name_to_family
)

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

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


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
        cc_bins: int = 32,
        chord_analysis_func: Optional[callable] = None,
        # Original REMI params (max_bar_embedding handled via config)
        params: str | Path | None = None,
    ) -> None:
        print("\n=== Initializing StructuredREMI instance ===")
        # --- Store Custom Params Temporarily --- #
        # We store them first, then ensure they are in the config before super().__init__
        # and finally reload them from the config after super().__init__ in case a params file was loaded.
        _use_track_names = use_track_names
        _use_global_chords = use_global_chords
        _use_filtered_ccs = use_filtered_ccs
        _cc_filter_mode = cc_filter_mode
        _explicit_cc_list = set(cc_list) if cc_list is not None else set()
        _num_cc_bins = cc_bins
        # Store chord analyzer, default to imported one
        self.chord_analyzer = chord_analysis_func if chord_analysis_func is not None else extract_chord_events

        # --- Setup Config (Important: before super().__init__) --- #
        # Provide a default config if none is given and no params file is specified
        if tokenizer_config is None and params is None:
            tokenizer_config = TokenizerConfig(
                use_chords=False, # Base miditok chords off if using our global ones
                use_rests=True,
                use_tempos=True,
                use_time_signatures=True,
                use_programs=True,
                use_control_changes=False, # Ensure miditok CCs are OFF
                one_token_stream_for_programs=True, # Usually desired for structured approach
                num_velocities=32,
                beat_res={(0, 4): 8, (4, 12): 4},
                additional_params={
                    'tempo_range': (40, 250), 'num_tempos': 64,
                    'time_signature_range': {4: [4], 2: [2], 8: [3, 5, 6, 12]},
                    'max_bar_embedding': None, # REMI default
                    'use_bar_end_tokens': False,
                    'add_trailing_bars': False,
                    # Add flags to config so they are saved/loaded automatically
                    'use_track_names': _use_track_names,
                    'use_global_chords': _use_global_chords,
                    'use_filtered_ccs': _use_filtered_ccs,
                    'cc_filter_mode': _cc_filter_mode,
                    'cc_list': list(_explicit_cc_list),
                    'num_cc_bins': _num_cc_bins,
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
            tokenizer_config.additional_params['num_cc_bins'] = _num_cc_bins
        # If only params is provided, the loaded config might not have our custom params yet.
        # We'll handle loading them *after* super().__init__

        # --- Call Parent Init --- #
        # Note: Removed max_bar_embedding direct param, handled by config
        super().__init__(tokenizer_config, params)

        # --- Setup Post Parent Init --- #
        self.logger = logging.getLogger(__name__)
        print(f"DEBUG: Logger name is {__name__}")
        self.logger.setLevel(logging.DEBUG)

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


        # Reload custom attributes from config's additional_params.
        # This handles the case where tokenizer_config was None but params was provided,
        # or updates the attributes if both were provided.
        self.use_track_names = self.config.additional_params.get('use_track_names', _use_track_names)
        self.use_global_chords = self.config.additional_params.get('use_global_chords', _use_global_chords)
        self.use_filtered_ccs = self.config.additional_params.get('use_filtered_ccs', _use_filtered_ccs)
        self.cc_filter_mode = self.config.additional_params.get('cc_filter_mode', _cc_filter_mode)
        # Ensure loaded cc_list is converted back to a set
        loaded_cc_list = self.config.additional_params.get('cc_list', list(_explicit_cc_list))
        self.explicit_cc_list = set(loaded_cc_list)
        self.num_cc_bins = self.config.additional_params.get('num_cc_bins', _num_cc_bins)

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

    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
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
        self._add_note_tokens_to_vocab_list(vocab)

        # Position
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab += [f"Position_{i}" for i in range(num_positions)]

        # Add additional tokens (Tempo, Chord, Rest, Program) handled by parent
        self._add_additional_tokens_to_vocab_list(vocab)

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
            num_cc_bins = config_params.get('num_cc_bins', 32) # Get bins from config
            # CC Types (0-127)
            vocab += [f"CCType_{i}" for i in range(128)]
            # CC Values (Binned)
            vocab += [f"CCValue_{i}" for i in range(num_cc_bins)]
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

        # Add links with other features
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
        print("\n=== Starting _score_to_events in StructuredREMI ===")
        print(f"Score tracks: {len(score.tracks)}")
        print(f"Score time division: {score.ticks_per_quarter}")
        print(f"Score end time: {score.end()}")
        print(f"Tokenizer config: {self.config}")

        # Sort tracks
        score.tracks.sort(key=lambda x: (x.program, x.is_drum))

        all_events: List[Event] = [] # Use a list to collect events

        # Global events (tempo, time sig, chords)
        if self.config.use_tempos: 
            print("\nProcessing tempo events...")
            tempo_events = self._tempo_events(score.tempos)
            print(f"Generated {len(tempo_events)} tempo events")
            all_events += tempo_events

        if self.config.use_time_signatures:
            print("\nProcessing time signature events...")
            ts_events = self._time_signature_events(score.time_signatures)
            print(f"Generated {len(ts_events)} time signature events")
            all_events += ts_events

        if self.use_global_chords:
            print("\n=== Processing Global Chords ===")
            chord_events_data = self.chord_analyzer(score)
            print(f"Found {len(chord_events_data)} chord events")
            
            # Log chord distribution
            chord_qualities = {}
            chord_roots = {}
            for chord_data in chord_events_data:
                quality = chord_data.get('quality', 'other')
                root_pc = chord_data.get('root_pc')
                chord_qualities[quality] = chord_qualities.get(quality, 0) + 1
                if root_pc is not None:
                    chord_roots[root_pc] = chord_roots.get(root_pc, 0) + 1
            
            print(f"Chord quality distribution: {chord_qualities}")
            print(f"Chord root distribution: {chord_roots}")
            
            for chord_data in chord_events_data:
                print(f"\nProcessing chord: {chord_data}")
                quality = chord_data.get('quality', 'other')
                if quality not in self.valid_chord_qualities: 
                    print(f"Remapping quality {quality} to 'other' (valid qualities: {sorted(list(self.valid_chord_qualities))})")
                    quality = 'other'
                root_pc = chord_data.get('root_pc')
                time = chord_data.get('time')
                if root_pc is not None and time is not None:
                    print(f"Adding chord event at time {time}: root={root_pc}, quality={quality}")
                    root_event = GlobalChordRootEvent(root_pc=root_pc, time=time)
                    qual_event = GlobalChordQualEvent(quality=quality, time=time)
                    print(f"Created events: {root_event}, {qual_event}")
                    all_events.extend([root_event, qual_event])
                else:
                    print(f"Skipping chord event due to missing root_pc or time: {chord_data}")

        # Track-specific events (notes, CCs, track names)
        for track_idx, track in enumerate(score.tracks):
            print(f"\n=== Processing Track {track_idx} ===")
            print(f"Program: {track.program}, Name: {track.name}")
            # Set current track program and type for CC filtering
            self._current_track_program = track.program
            # Map track name to family first
            mapped_family = map_track_name_to_family(track.name)
            # Determine instrument type based on program (more reliable for CCs than name)
            self._current_track_instrument_type = get_instrument_type_from_program(track.program)
            print(f"Mapped to family: {mapped_family}, instrument type: {self._current_track_instrument_type}")

            track_events = []
            # Add TrackName event if enabled (at the start of track events)
            if self.use_track_names:
                 track_events.append(TrackNameEvent(
                     name=mapped_family,
                     instrument_type=self._current_track_instrument_type,
                     time=0
                 ))

            # Add program events
            program_events = self._program_events([track])
            print(f"Generated {len(program_events)} program events")
            track_events += program_events

            # Add note events
            note_events = self._note_events(track.notes)
            print(f"Generated {len(note_events)} note events")
            track_events += note_events

            # Add pedal events
            if self.config.use_sustain_pedals:
                pedal_events = self._pedal_events(track.pedals)
                print(f"Generated {len(pedal_events)} pedal events")
                track_events += pedal_events

            # Add pitch bend events
            if self.config.use_pitch_bends:
                pb_events = self._pitch_bend_events(track.pitch_bends)
                print(f"Generated {len(pb_events)} pitch bend events")
                track_events += pb_events

            # Add our custom CC events
            if self.use_filtered_ccs and track.controls:
                print(f"\nProcessing {len(track.controls)} CC events")
                print(f"Filter mode: {self.cc_filter_mode}")
                cc_events = self._create_control_change_events_filtered(track.controls)
                print(f"Generated {len(cc_events)} CC events")
                track_events += cc_events

            # Append track events to all events
            all_events += track_events
            print(f"Total events for track {track_idx}: {len(track_events)}")

        # Sort all events by time, then priority, then type
        all_events.sort(key=lambda x: (x.time, self._event_priority.get(x.type_, 10), x.value))
        print(f"\n=== Final Event Count: {len(all_events)} ===")

        # Reset internal track state before processing (important if tokenizer is reused)
        self._current_track_instrument_type = "unknown"
        self._current_track_program = 0

        # Add time events (Bars, Positions, Rests)
        events_with_time = self._add_time_events(all_events, score.ticks_per_quarter)
        print(f"Final event count after adding time events: {len(events_with_time)}")

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
        print("\n=== Final Token Sequence ===")
        print("First 50 tokens:")
        for i, token in enumerate(tok_sequence.tokens[:50]):
            print(f"{i}: {token}")
        if len(tok_sequence.tokens) > 50:
            print(f"... ({len(tok_sequence.tokens) - 50} more tokens)")

        return tok_sequence

    def _create_control_change_events_filtered(self, controls: List[ControlChange]) -> List[Event]:
        events = []
        if not controls: 
            print("No CC events to process")
            return events

        allowed_ccs: Set[int]
        if self.cc_filter_mode == 'all':
            allowed_ccs = set(range(128))
            print("Using 'all' CC filter mode - allowing all CCs")
        elif self.cc_filter_mode == 'explicit_list':
            allowed_ccs = self.explicit_cc_list
            print(f"Using explicit CC list: {sorted(list(allowed_ccs))}")
        elif self.cc_filter_mode == 'instrument_aware':
            allowed_ccs = self.cc_instrument_map.get(self._current_track_instrument_type,
                                                     self.cc_instrument_map["unknown"])
            print(f"Using instrument-aware CC filtering for type {self._current_track_instrument_type}")
            print(f"Allowed CCs for this instrument: {sorted(list(allowed_ccs))}")
        else: # 'none' or invalid mode
            print(f"Invalid CC filter mode '{self.cc_filter_mode}' - no CCs will be processed")
            return events

        print(f"Processing {len(controls)} CC events")
        cc_types_seen = defaultdict(int)
        cc_values_seen = defaultdict(list)
        
        for control in controls:
            print(f"Processing CC: number={control.number}, value={control.value}, time={control.time}")
            if control.number in allowed_ccs:
                # Track CC type usage
                cc_types_seen[control.number] += 1
                
                # Ensure value is in valid MIDI range
                value = min(max(control.value, 0), 127)
                # Track CC values
                cc_values_seen[control.number].append(value)
                # Bin the value
                binned_value = int((value / 127.0) * (self.num_cc_bins - 1))
                print(f"CC {control.number} value {value} binned to {binned_value} (using {self.num_cc_bins} bins)")

                events.append(CCTypeEvent(number=control.number, time=control.time))
                events.append(CCValueEvent(value=binned_value, time=control.time))
            else:
                print(f"Skipping CC {control.number} (not in allowed set)")

        print("\nCC Type Usage Summary:")
        for cc_type, count in sorted(cc_types_seen.items()):
            print(f"CC {cc_type}: {count} times")
            values = cc_values_seen[cc_type]
            print(f"  Value range: {min(values)}-{max(values)}, avg: {sum(values)/len(values):.1f}")

        print(f"\nGenerated {len(events)} CC events (type+value pairs)")
        return events 

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: Optional[np.ndarray] = None,
        ticks_per_quarter: Optional[int] = None,
        ticks_bars: Optional[np.ndarray] = None,
        ticks_beats: Optional[np.ndarray] = None,
        attribute_controls_indexes: Optional[Dict[int, Union[List[int], bool]]] = None,
    ) -> List[Event]:
        """Create track events including our custom events.
        
        Args:
            track: The track to create events for
            ticks_per_beat: Array of ticks per beat
            ticks_per_quarter: Number of ticks per quarter
            ticks_bars: Array of bar tick positions
            ticks_beats: Array of beat tick positions
            attribute_controls_indexes: Dictionary mapping track indices to control change indexes
            
        Returns:
            List of Event objects for this track
        """
        track_events = []
        
        # First try to use parent's _create_track_events if available
        if hasattr(super(), '_create_track_events'):
            try:
                parent_events = super()._create_track_events(
                    track=track,
                    ticks_per_beat=ticks_per_beat,
                    time_division=ticks_per_quarter,
                    ticks_bars=ticks_bars,
                    ticks_beats=ticks_beats,
                    add_track_attribute_controls=False,
                    bar_idx_attribute_controls=None
                )
                track_events.extend(parent_events)
            except Exception as e:
                print(f"Warning: Parent _create_track_events failed: {e}")
                # Fall back to individual event creation
                pass
        
        # If parent method failed or doesn't exist, create events individually
        if not track_events:
            # Program events
            if hasattr(super(), '_program_events'):
                program_events = super()._program_events([track])
                track_events.extend(program_events)
            
            # Note events
            if hasattr(super(), '_note_events'):
                note_events = super()._note_events(track.notes)
                track_events.extend(note_events)
            
            # Pedal events
            if self.config.use_sustain_pedals and hasattr(super(), '_pedal_events'):
                pedal_events = super()._pedal_events(track.pedals)
                track_events.extend(pedal_events)
            
            # Pitch bend events
            if self.config.use_pitch_bends and hasattr(super(), '_pitch_bend_events'):
                pitch_bend_events = super()._pitch_bend_events(track.pitch_bends)
                track_events.extend(pitch_bend_events)
        
        # Add our custom CC events
        if self.use_filtered_ccs and track.controls:
            print(f"\nProcessing {len(track.controls)} CC events")
            print(f"Filter mode: {self.cc_filter_mode}")
            cc_events = self._create_control_change_events_filtered(track.controls)
            print(f"Generated {len(cc_events)} CC events")
            track_events.extend(cc_events)
            
        return track_events 

    def _create_global_events(self, score: Score) -> List[Event]:
        """Create global events including our custom chord events."""
        print("\n=== Starting _create_global_events in StructuredREMI ===")
        
        # Get base global events from parent class
        global_events = super()._create_global_events(score)
        print(f"Base global events: {len(global_events)}")

        # Add our custom chord events
        if self.use_global_chords:
            print("\nAnalyzing global chords...")
            print(f"Chord analyzer: {self.chord_analyzer}")
            print(f"Score time division: {score.ticks_per_quarter}")
            print(f"Score tracks: {len(score.tracks)}")
            
            chord_events_data = self.chord_analyzer(score)
            print(f"Found {len(chord_events_data)} chord events")
            
            # Log chord distribution
            chord_qualities = {}
            chord_roots = {}
            for chord_data in chord_events_data:
                quality = chord_data.get('quality', 'other')
                root_pc = chord_data.get('root_pc')
                chord_qualities[quality] = chord_qualities.get(quality, 0) + 1
                if root_pc is not None:
                    chord_roots[root_pc] = chord_roots.get(root_pc, 0) + 1
            
            print(f"Chord quality distribution: {chord_qualities}")
            print(f"Chord root distribution: {chord_roots}")
            
            for chord_data in chord_events_data:
                print(f"\nProcessing chord: {chord_data}")
                quality = chord_data.get('quality', 'other')
                if quality not in self.valid_chord_qualities: 
                    print(f"Remapping quality {quality} to 'other' (valid qualities: {sorted(list(self.valid_chord_qualities))})")
                    quality = 'other'
                root_pc = chord_data.get('root_pc')
                time = chord_data.get('time')
                if root_pc is not None and time is not None:
                    print(f"Adding chord event at time {time}: root={root_pc}, quality={quality}")
                    root_event = GlobalChordRootEvent(root_pc=root_pc, time=time)
                    qual_event = GlobalChordQualEvent(quality=quality, time=time)
                    print(f"Created events: {root_event}, {qual_event}")
                    global_events.extend([root_event, qual_event])
                else:
                    print(f"Skipping chord event due to missing root_pc or time: {chord_data}")

        print(f"\nTotal global events: {len(global_events)}")
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