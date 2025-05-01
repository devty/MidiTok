"""MuMIDI tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from symusic import Note, Score, Tempo, Track, ControlChange

from miditok.classes import Event, TokSequence, TokenizerConfig
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import detect_chords, get_bars_ticks, get_score_ticks_per_beat
from miditok.utils.structured_utils import (
    map_track_name_to_family,
    get_instrument_type_from_program,
    get_relevant_ccs_for_instrument,
    TARGET_INSTRUMENT_FAMILIES,
    get_program_from_instrument_family,
)
from typing import Set, Dict, List, Optional
import logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

# Default number of bins for CC values if not specified
DEFAULT_CC_BINS = 128

class MuMIDI(MusicTokenizer):
    r"""
    MuMIDI tokenizer.

    Introduced with `PopMAG (Ren et al.) <https://arxiv.org/abs/2008.07703>`_,
    this tokenization made for multitrack tasks and uses embedding pooling. Time is
    represented with *Bar* and *Position* tokens. The key idea of MuMIDI is to represent
    all tracks in a single token sequence. At each time step, *Track* tokens preceding
    note tokens indicate their track. MuMIDI also include a "built-in" and learned
    positional encoding. As in the original paper, the pitches of drums are distinct
    from those of all other instruments.

    **MODIFIED VERSION:** This version replaces `Program` tokens with `TrackName`
    (representing instrument families) and adds optional support for filtered
    Continuous Controller (CC) events (`CCType`, `CCValue`), inspired by StructuredREMI.

    Each pooled token will be a list of the form (index: Token type):

    * 0: Pitch / PitchDrum / Position / Bar / TrackName / CCType / CCValue / (Chord) / (Rest);
    * 1: BarPosEnc;
    * 2: PositionPosEnc;
    * (-3 / 3: Tempo);
    * -2: Velocity;
    * -1: Duration.

    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can
    be very delicate. Hence, we do not recommend this tokenization for generation with
    small models.

    **Notes:**

    * Tokens are first sorted by time, then potentially by a priority heuristic (e.g., TrackName before Pitch), then pitch/type.
    * Tracks are processed based on their derived instrument family. Merging might occur if tracks map to the same family.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path | None = None,
        **kwargs,
    ) -> None:
        # Initialize logger FIRST
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING) # Default level

        # Store Custom Params Temporarily
        # Get from kwargs or set defaults, these will be stored in config later
        _use_track_names = kwargs.get('use_track_names', True)
        _use_filtered_ccs = kwargs.get('use_filtered_ccs', True)
        _cc_filter_mode = kwargs.get('cc_filter_mode', 'instrument_aware')
        _explicit_cc_list = set(kwargs.get('cc_list', []))
        _num_cc_bins = kwargs.get('num_cc_bins', DEFAULT_CC_BINS)

        # Setup Config (BEFORE super().__init__)
        # Handle the case where no config or params file is provided
        if tokenizer_config is None and params is None:
             tokenizer_config = TokenizerConfig(
                 # Base MuMIDI defaults (adjust as needed)
                 use_programs=False, # Disable original program handling
                 use_chords=False, # MuMIDI doesn't use chords by default
                 use_rests=False,
                 use_tempos=True, # Keep tempo
                 use_time_signatures=False,
                 use_sustain_pedals=False,
                 use_pitch_bends=False,
                 use_velocities=True, # MuMIDI uses velocities
                 use_note_duration_programs=[0,1,2,3], # Default to using durations for all
                 one_token_stream_for_programs=True, # Core MuMIDI concept
                 program_changes=False,
                 # Default beat res, potentially adjust
                 beat_res={(0, 4): 8, (4, 12): 4},
                 num_velocities=128, # MuMIDI usually uses 128
                 additional_params={
                     'max_bar_embedding': 256, # Default from original MuMIDI
                     # Add new flags to default config
                     'use_track_names': _use_track_names,
                     'use_filtered_ccs': _use_filtered_ccs,
                     'cc_filter_mode': _cc_filter_mode,
                     'cc_list': list(_explicit_cc_list),
                     'num_cc_bins': _num_cc_bins,
                 }
             )
        # If config IS provided, ensure programs are off and store/update custom params
        elif tokenizer_config is not None:
            tokenizer_config.use_programs = False # Force off
            # Store/update custom params in additional_params for saving/loading
            tokenizer_config.additional_params['use_track_names'] = _use_track_names
            tokenizer_config.additional_params['use_filtered_ccs'] = _use_filtered_ccs
            tokenizer_config.additional_params['cc_filter_mode'] = _cc_filter_mode
            tokenizer_config.additional_params['cc_list'] = list(_explicit_cc_list)
            tokenizer_config.additional_params.setdefault('num_cc_bins', _num_cc_bins)
            # Ensure use_programs is False if use_track_names is True
            if tokenizer_config.additional_params.get('use_track_names', False):
                 tokenizer_config.use_programs = False


        # Call parent __init__ AFTER config setup
        super().__init__(tokenizer_config=tokenizer_config, params=params, **kwargs)

        # Setup Post Parent Init
        # Reload custom attributes from the *final* config (self.config)
        # Use the temporarily stored defaults as fallbacks
        # Ensure additional_params exists
        if not hasattr(self.config, 'additional_params') or self.config.additional_params is None:
            self.logger.warning("self.config.additional_params missing after super().__init__(). Initializing.")
            self.config.additional_params = {}

        self.use_track_names = self.config.additional_params.get('use_track_names', _use_track_names)
        self.use_filtered_ccs = self.config.additional_params.get('use_filtered_ccs', _use_filtered_ccs)
        self.cc_filter_mode = self.config.additional_params.get('cc_filter_mode', _cc_filter_mode)
        loaded_cc_list = self.config.additional_params.get('cc_list', list(_explicit_cc_list))
        self.explicit_cc_list = set(loaded_cc_list)
        self.num_cc_bins = self.config.additional_params.get('num_cc_bins', _num_cc_bins)
        # --- Force update the config object itself in case defaults were used ---
        self.config.additional_params['use_track_names'] = self.use_track_names
        self.config.additional_params['use_filtered_ccs'] = self.use_filtered_ccs
        self.config.additional_params['cc_filter_mode'] = self.cc_filter_mode
        self.config.additional_params['cc_list'] = list(self.explicit_cc_list)
        self.config.additional_params['num_cc_bins'] = self.num_cc_bins
        self.logger.info(f"Initialized Enhanced MuMIDI with: use_track_names={self.use_track_names}, "
                         f"use_filtered_ccs={self.use_filtered_ccs} (mode={self.cc_filter_mode}, bins={self.num_cc_bins})")

        # Initialize CC Instrument Map (similar to StructuredREMI)
        self.cc_instrument_map: Dict[str, Set[int]] = {}
        # Use families directly as keys
        instrument_types_for_cc = list(TARGET_INSTRUMENT_FAMILIES) + ["unknown"] # Use families
        default_relevant_ccs = get_relevant_ccs_for_instrument("unknown", extended=True) # Default for unknown

        for family in instrument_types_for_cc:
            # Map family to a representative program to get instrument type for CCs
            # This assumes get_program_from_instrument_family exists and works
            program, is_drum = get_program_from_instrument_family(family)
            instrument_type = get_instrument_type_from_program(program)
            self.cc_instrument_map[family] = set(get_relevant_ccs_for_instrument(instrument_type, extended=True))

        # Ensure 'unknown' family exists
        if "unknown" not in self.cc_instrument_map:
             self.cc_instrument_map["unknown"] = set(default_relevant_ccs)

        # Internal State
        self._current_track_family: str = "unknown"
        self._current_track_instrument_type: str = "unknown" # Derived from family

    def _tweak_config_before_creating_voc(self) -> None:
        # Original MuMIDI tweaks
        self.config.use_rests = False
        self.config.use_time_signatures = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        # self.config.use_programs = True # <<< Changed: Disable original programs
        self.config.use_pitch_intervals = True # MuMIDI uses pitch intervals? Check original paper if needed. Let's assume no for now.
        self.config.use_pitch_intervals = False # Let's disable pitch intervals for now
        self.config.one_token_stream_for_programs = True # Keep this core concept
        self.config.program_changes = False # Not applicable with TrackNames
        self._disable_attribute_controls() # MuMIDI doesn't use these

        # <<< New Tweaks >>>
        # Ensure use_programs is False if use_track_names is True (redundant check)
        if self.config.additional_params.get('use_track_names', True):
             self.config.use_programs = False
        else:
            # If track names are disabled, we need fallback to original program logic
            self.logger.warning("`use_track_names` is False. Falling back to original Program logic. Ensure `use_programs` is True in config.")
            self.config.use_programs = True # Enable original program logic


        # Ensure Duration config is compatible (all or none)
        # This part seems okay to keep, assuming duration logic remains per-track/family
        if self.config.use_note_duration_programs is not None and len(self.config.use_note_duration_programs) > 0:
             # If some programs were specified, warn and apply to all *families* potentially?
             # For now, let's assume if use_note_duration_programs is used, it means ALL families.
             # A more granular control per-family seems overly complex for MuMIDI.
             # Let's keep the simpler logic: durations are used for *all* tracks or *none*.
             # We just need to ensure the config reflects this if it was set otherwise.
             if len(self.config.programs) > 0: # Check if original programs list was non-empty
                 self.config.use_note_duration_programs = list(range(-1, 128)) # Assume apply to all possible programs/families
                 warn(
                    "Original `use_note_duration_programs` might conflict with family-based approach. "
                    "Assuming note duration tokens apply to all tracks if enabled.",
                    stacklevel=2,
                 )
        # If durations are generally enabled (default True), ensure the list covers all families implicitly.
        # This is handled by the `config.using_note_duration_tokens` property check later.


        # Add max_bar_embedding if missing (MuMIDI default)
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = 128

        # <<< Update vocab_types_idx >>>
        self.vocab_types_idx = {
            "Pitch": 0,
            "PitchDrum": 0,
            "Position": 0,
            "Bar": 0,
            # "Program": 0, # <<< REMOVED
            "BarPosEnc": 1,
            "PositionPosEnc": 2,
        }
        # Add TrackName if used
        if self.config.additional_params.get('use_track_names', True):
             self.vocab_types_idx["TrackName"] = 0
        else:
             # Fallback to original Program if track names disabled
             self.vocab_types_idx["Program"] = 0

        # Add CC types if used
        if self.config.additional_params.get('use_filtered_ccs', True):
             self.vocab_types_idx["CCType"] = 0
             self.vocab_types_idx["CCValue"] = 0

        # Add optional token types, adjusting indices based on what's present
        current_neg_idx = -1
        if self.config.using_note_duration_tokens:
             self.vocab_types_idx["Duration"] = current_neg_idx
             current_neg_idx -= 1
        if self.config.use_velocities:
             self.vocab_types_idx["Velocity"] = current_neg_idx
             current_neg_idx -= 1
        if self.config.use_chords: # Although MuMIDI typically doesn't use chords
             self.vocab_types_idx["Chord"] = 0
        if self.config.use_rests: # Although MuMIDI typically doesn't use rests
             self.vocab_types_idx["Rest"] = 0
        if self.config.use_tempos:
             # Place tempo after BarPosEnc/PositionPosEnc if they exist
             tempo_idx = 3 if 2 in self.vocab_types_idx.values() else -abs(current_neg_idx) # Place as 3 or next negative
             self.vocab_types_idx["Tempo"] = tempo_idx
             # Make sure index 3 is available if used
             if tempo_idx == 3 and 3 in self.vocab_types_idx.values() and self.vocab_types_idx["Tempo"] != 3:
                  self.logger.warning("Potential index collision for Tempo token. Adjusting.")
                  # Find next available positive index or use negative
                  tempo_idx = max(v for v in self.vocab_types_idx.values() if v >= 0) + 1
                  self.vocab_types_idx["Tempo"] = tempo_idx


    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        """
        Create the time events from a list of global and track events.

        Unused here.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """

    def _track_to_tokens(
        self, track: Track, ticks_per_beat: np.ndarray = None, track_family: str = "unknown"
    ) -> list[Event]:
        r"""
        Convert a track (``symusic.Track``) into a sequence of Events.

        Generates Note events (Pitch, Velocity, Duration) bundled into a single Event
        and potentially CC events (CCType, CCValue) based on the tokenizer's
        configuration. The track's instrument family is used for CC filtering.

        :param track: track object to convert.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            time signature denominator section. (Not directly used in MuMIDI base).
        :param track_family: The determined instrument family for this track.
        :return: sequence of corresponding Events.
        """
        del ticks_per_beat # Unused in base MuMIDI logic here
        # Make sure the notes are sorted first by their onset (start) times, second by
        # pitch: notes.sort(key=lambda x: (x.start, x.pitch)) (done in preprocess_score)

        events = []
        tpb = self.time_division # Needed for duration calculation

        # Determine track_id based on config (family or program)
        _using_track_names = self.config.additional_params.get('use_track_names', True)
        track_id = track_family if _using_track_names else (track.program if not track.is_drum else -1)


        # Note Events: Bundle Pitch, Velocity, Duration
        for note in track.notes:
            # Determine Velocity value
            velocity_val = note.velocity
            if not self.config.use_velocities:
                velocity_val = None # Or use default? Let's use None to signify absence

            # Determine Duration token string
            duration_str = None
            if self.config.using_note_duration_tokens:
                duration_ticks = note.end - note.start
                # Handle potential errors if duration is not in the map (e.g., too long)
                try:
                    duration_str = self._tpb_ticks_to_tokens[tpb][duration_ticks]
                except KeyError:
                    # Handle unseen duration. Use closest? Use default? Log warning?
                    # For now, let's use the shortest duration as fallback
                    shortest_duration = self.durations[0]
                    duration_str = ".".join(map(str, shortest_duration))
                    self.logger.warning(
                        f"Note duration {duration_ticks} not found in tpb map {tpb}. "
                        f"Using shortest duration token '{duration_str}' as fallback. "
                        f"Time={note.start}, Pitch={note.pitch}"
                    )

            # Create a string representation of the data to store in desc
            # Format: "velocity|duration_str|track_id"
            desc_str = f"{velocity_val}|{duration_str if duration_str else ''}|{track_id}"
            
            # Create a single combined Pitch/Note Event with serialized desc
            note_event = Event(
                type_="Pitch" if not track.is_drum else "PitchDrum",
                value=note.pitch,
                time=note.start,
                desc=desc_str  # Store as string instead of tuple
            )
            events.append(note_event)


        # Adds chord tokens if specified (though typically not used in MuMIDI)
        if self.config.use_chords and not track.is_drum:
             # This requires ticks_per_beat, which MuMIDI doesn't inherently use.
             # If chords are enabled, ensure get_score_ticks_per_beat is called earlier.
             self.logger.warning("Using chords with MuMIDI requires careful handling of ticks_per_beat.")
             # Placeholder: Assume ticks_per_beat is available if this branch is taken
             # ticks_per_beat_arr = get_score_ticks_per_beat(score) # Needs score access
             # chords = detect_chords(...)
             # ... add chord events ...
             pass # Skipping full chord implementation for now


        # Add Filtered CC Events if enabled
        if self.use_filtered_ccs and track.controls:
             cc_events = self._create_control_change_events_filtered(track.controls, track_family)
             # Assign the track_id to the desc field for sorting/grouping CC events
             for cc_event in cc_events:
                 cc_event.desc = str(track_id)  # Store as string for consistency
             events.extend(cc_events)

        return events


    def _create_control_change_events_filtered(self, controls: List[ControlChange], track_family: str) -> List[Event]:
        """
        Generates CCType and CCValue events based on filtering rules.

        Adapted from StructuredREMI.

        :param controls: List of ControlChange objects from the track.
        :param track_family: The instrument family of the current track.
        :return: List of CCType and CCValue Events.
        """
        events = []
        if not controls:
            return events

        allowed_ccs: Set[int]
        # Get program and is_drum status from family
        program, _ = get_program_from_instrument_family(track_family)
        # Get instrument type using only the program number
        inst_type = get_instrument_type_from_program(program)

        if self.cc_filter_mode == 'all':
            allowed_ccs = set(range(128))
        elif self.cc_filter_mode == 'explicit_list':
            allowed_ccs = self.explicit_cc_list
        elif self.cc_filter_mode == 'instrument_aware':
             # Use the family directly if mapped, otherwise fallback via instrument type
             allowed_ccs = self.cc_instrument_map.get(track_family, self.cc_instrument_map.get(inst_type, self.cc_instrument_map["unknown"]))
        else: # 'none' or invalid mode
            return events

        num_bins = self.config.additional_params.get('num_cc_bins', DEFAULT_CC_BINS)

        for control in controls:
            if control.number in allowed_ccs:
                value = min(max(control.value, 0), 127)
                binned_value = int((value / 127.0) * (num_bins - 1))
                # Create CCType and CCValue events sequentially at the same time
                events.append(Event(type_="CCType", value=control.number, time=control.time))
                events.append(Event(type_="CCValue", value=binned_value, time=control.time))

        return events


    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        Handles TrackName mapping and CC filtering. Constructs the pooled token sequence.

        :param score: the :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` representing the score.
        """
        del attribute_controls_indexes # Not used by MuMIDI
        # Check bar embedding limit, update if needed (same as original)
        bar_ticks = get_bars_ticks(score, only_notes_onsets=True)
        max_bar_embedding = self.config.additional_params.get("max_bar_embedding")
        if max_bar_embedding is not None and max_bar_embedding < len(bar_ticks):
            score = score.clip(
                0, bar_ticks[max_bar_embedding]
            )
            msg = (
                f"miditok: {type(self).__name__} cannot tokenize entirely this file "
                f"as it contains {len(bar_ticks)} bars whereas the limit of the "
                f"tokenizer is {max_bar_embedding}. "
                "It is therefore clipped to "
                f"{max_bar_embedding} bars."
            )
            warn(msg, stacklevel=2)


        # --- Generate Events per Track ---
        all_track_events: List[Event] = []
        # Determine if using TrackNames or fallback Programs
        _using_track_names = self.config.additional_params.get('use_track_names', True)

        for track in score.tracks:
             track_id: str | int # Store family name or program number
             track_family = "unknown" # Default

             if _using_track_names:
                 track_family = map_track_name_to_family(track.name)
                 track_id = track_family
                 # Skip track if family is None or handle as 'unknown'? Let's map to unknown.
                 if track_family is None:
                      self.logger.debug(f"Track '{track.name}' did not map to a known family. Using 'unknown'.")
                      track_family = "unknown"
                      track_id = "unknown"

                 # Find the earliest event time in the track (notes or controls)
                 note_times = [n.time for n in track.notes]
                 control_times = [c.time for c in track.controls]
                 all_times = note_times + control_times
                 first_event_time = min(all_times) if all_times else 0

                 track_name_event = Event(type_="TrackName", value=track_family, time=first_event_time, desc=str(track_family))
                 all_track_events.append(track_name_event)

             else: # Fallback to using Program numbers
                 track_id = track.program if not track.is_drum else -1
                 # Skip tracks with programs outside the config list if not using track names
                 if track_id not in self.config.programs:
                      continue
                 # Need Program event? MuMIDI original logic implies Program tokens are inserted
                 # *during* the main loop based on `current_track`. Let's stick to that.
                 # The desc field will carry the program number.

             # Generate note and CC events for the track
             track_events = self._track_to_tokens(track, track_family=track_family) # Pass family for CC filtering

             all_track_events.extend(track_events)


        # --- Event Priority Definition --- (Lower number = higher priority)
        event_priority = {
            # Structural
            "Bar": 0,
            "Position": 1,
            # Global Context (if added later)
            # ...
            # Track Identifier (TrackName or Program - handled implicitly by loop)
            "TrackName": 2,
            # "Program": 2, # If using programs
            # Control Changes
            "CCType": 3,
            "CCValue": 4,
            # Note Components
            "Pitch": 5, "PitchDrum": 5,
            "Velocity": 6,
            "Duration": 7,
            # Others
            "Tempo": 8, # Tempo relatively low priority?
        }
        # --- Sort All Events ---
        # Sort by time, then priority, then value (for stable sort within same time/priority)
        all_track_events.sort(key=lambda x: (x.time, event_priority.get(x.type_, 10), x.value))


        # --- Build Token Sequence ---
        tokens: list[list[str | None]] = [] # Use None for missing optional slots
        ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
        ticks_per_bar = score.ticks_per_quarter * 4 # Assume 4/4

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track_id: str | int = "unknown" if _using_track_names else -2 # Track family or program
        if not _using_track_names: current_track_id = -2 # Initial invalid program

        current_tempo_idx = 0
        # <<< Re-apply Fix: Quantize initial tempo >>>
        raw_initial_tempo = score.tempos[0].tempo if score.tempos else 120.0 # Default tempo
        if self.config.use_tempos and self.tempos is not None and len(self.tempos) > 0:
             # Quantize the initial tempo to the closest value in self.tempos
             current_tempo = min(self.tempos, key=lambda x: abs(x - raw_initial_tempo))
        else:
             current_tempo = raw_initial_tempo # Keep raw if tempos not used

        # Placeholder strings for optional token types when not present
        velocity_none = None
        if self.config.use_velocities and self.velocities is not None and len(self.velocities) > 0:
            # Find the velocity value closest to the default (DEFAULT_VELOCITY)
            quantized_default_velocity = min(self.velocities, key=lambda x: abs(x - DEFAULT_VELOCITY))
            velocity_none = f"Velocity_{quantized_default_velocity}"

        duration_none = None # Initial value
        if self.config.using_note_duration_tokens:
             # Find a default duration token string - typically the smallest value
             default_dur_val = self.durations[0]
             duration_none = f'Duration_{".".join(map(str, default_dur_val))}'


        # Pre-calculate vocabulary sizes for index mapping
        num_pos_idx = 0
        num_neg_idx = 0
        if self.vocab_types_idx:
             num_pos_idx = max((idx for idx in self.vocab_types_idx.values() if idx >= 0), default=0) + 1
             num_neg_idx = abs(min((idx for idx in self.vocab_types_idx.values() if idx < 0), default=-1))
        num_token_types = num_pos_idx + num_neg_idx

        # --- Main loop through sorted events ---
        for event in all_track_events:
            event_time = event.time
            event_type = event.type_
            event_value = event.value
            # event_track_id = event.desc # We'll parse desc later if needed

            # <<< Extract track_id, velocity, duration from desc based on type >>>
            event_track_id = None
            note_velocity = None
            note_duration_str = None

            if event_type in ["Pitch", "PitchDrum"]:
                # Try to unpack bundled info from desc (string format)
                try:
                    if isinstance(event.desc, str) and "|" in event.desc:
                        # Parse the serialized string format
                        desc_parts = event.desc.split("|")
                        note_velocity_str = desc_parts[0]
                        # Correctly handle 'None' string before int conversion
                        note_velocity = int(note_velocity_str) if note_velocity_str and note_velocity_str != 'None' else None
                        note_duration_str = desc_parts[1] if desc_parts[1] else None
                        event_track_id = desc_parts[2]
                        # Convert track_id back to int if it's meant to be a program number
                        if not _using_track_names and isinstance(event_track_id, str) and event_track_id.lstrip('-').isdigit():
                            event_track_id = int(event_track_id)
                    # Removed the else block that attempted tuple unpacking
                    else:
                        # If desc is not the expected string format, raise an error to be caught
                        raise TypeError(f"event.desc has unexpected format: {type(event.desc)}, value: {event.desc}")

                except (TypeError, ValueError, IndexError) as e:
                    self.logger.warning(f"Could not unpack desc for event {event} (Error: {e}). Using defaults.")
                    # Fallback if desc is not as expected
                    event_track_id = "unknown" if _using_track_names else -1

            elif event_type in ["CCType", "CCValue", "TrackName"] or (_using_track_names is False and event_type == "Program"):
                 # These events store track_id directly in desc
                 event_track_id = event.desc
                 # Convert track_id back to int if needed for program numbers
                 if not _using_track_names and isinstance(event_track_id, str) and event_track_id.lstrip('-').isdigit():
                     event_track_id = int(event_track_id)
            # Add other event types (like Tempo) if they need track association stored in desc


            # (Tempo) Update tempo before processing event time
            if self.config.use_tempos and current_tempo_idx + 1 < len(score.tempos):
                for tempo_change in score.tempos[current_tempo_idx + 1 :]:
                    if tempo_change.time <= event_time:
                        # <<< Fix: Quantize tempo to closest value in vocab >>>
                        # Get raw tempo
                        raw_tempo = round(tempo_change.tempo, 2)
                        # Find the closest tempo in the configured tempos (self.tempos)
                        closest_tempo = min(self.tempos, key=lambda x: abs(x - raw_tempo))
                        current_tempo = closest_tempo # Use the quantized tempo
                        current_tempo_idx += 1
                    else:
                        break # Tempo change is later

            # --- Process Time ---
            # Check if time has changed, handle Bar/Position tokens
            if event_time != current_tick:
                 new_bar = event_time // ticks_per_bar
                 # (New Bar)
                 if new_bar > current_bar:
                      # Add missing Bar tokens
                      for bar_idx in range(current_bar + 1, new_bar + 1):
                           bar_time = bar_idx * ticks_per_bar
                           bar_token_list = [None] * num_token_types # Initialize with None
                           # Slot 0: Bar
                           bar_token_list[self.vocab_types_idx["Bar"]] = "Bar_None"
                           # Slot 1: BarPosEnc
                           bar_pos_enc = f"BarPosEnc_{bar_idx}"
                           bar_token_list[self.vocab_types_idx["BarPosEnc"]] = bar_pos_enc
                           # Slot 2: PositionPosEnc
                           pos_pos_enc = "PositionPosEnc_None"
                           bar_token_list[self.vocab_types_idx["PositionPosEnc"]] = pos_pos_enc
                           # Optional Tempo
                           if self.config.use_tempos:
                               tempo_idx = self.vocab_types_idx["Tempo"]
                               list_idx = tempo_idx if tempo_idx >= 0 else num_token_types + tempo_idx
                               # <<< Use quantized current_tempo >>>
                               bar_token_list[list_idx] = f"Tempo_{current_tempo}"
                            # Optional Velocity/Duration (use None placeholders for Bar)
                           if self.config.use_velocities:
                               vel_idx = self.vocab_types_idx["Velocity"]
                               list_idx = vel_idx if vel_idx >= 0 else num_token_types + vel_idx
                               bar_token_list[list_idx] = velocity_none # Placeholder for non-note events
                           if self.config.using_note_duration_tokens:
                               dur_idx = self.vocab_types_idx["Duration"]
                               list_idx = dur_idx if dur_idx >= 0 else num_token_types + dur_idx
                               bar_token_list[list_idx] = duration_none # Placeholder for non-note events

                           tokens.append(bar_token_list)
                      current_bar = new_bar
                      current_pos = -1 # Reset position within the new bar
                      # Reset current_track_id as we are in a new bar
                      # current_track_id = "unknown" if _using_track_names else -2 # Resetting here might be too early if event at bar start defines it


                 # (New Position) Calculate and add Position token if needed
                 pos_index = int((event_time % ticks_per_bar) / ticks_per_sample)
                 if pos_index != current_pos: # Only add if position changed since last event or bar start
                      current_pos = pos_index
                      current_tick = event_time # Update current_tick to match the new position's time

                      pos_token_list = [None] * num_token_types
                      # Slot 0: Position
                      pos_token_list[self.vocab_types_idx["Position"]] = f"Position_{current_pos}"
                      # Slot 1: BarPosEnc
                      bar_pos_enc = f"BarPosEnc_{current_bar}"
                      pos_token_list[self.vocab_types_idx["BarPosEnc"]] = bar_pos_enc
                      # Slot 2: PositionPosEnc
                      pos_pos_enc = f"PositionPosEnc_{current_pos}"
                      pos_token_list[self.vocab_types_idx["PositionPosEnc"]] = pos_pos_enc
                      # Optional Tempo
                      if self.config.use_tempos:
                           tempo_idx = self.vocab_types_idx["Tempo"]
                           list_idx = tempo_idx if tempo_idx >= 0 else num_token_types + tempo_idx
                           # <<< Use quantized current_tempo >>>
                           pos_token_list[list_idx] = f"Tempo_{current_tempo}"
                        # Optional Velocity/Duration (use None placeholders for Position)
                      if self.config.use_velocities:
                            vel_idx = self.vocab_types_idx["Velocity"]
                            list_idx = vel_idx if vel_idx >= 0 else num_token_types + vel_idx
                            pos_token_list[list_idx] = velocity_none
                      if self.config.using_note_duration_tokens:
                            dur_idx = self.vocab_types_idx["Duration"]
                            list_idx = dur_idx if dur_idx >= 0 else num_token_types + dur_idx
                            pos_token_list[list_idx] = duration_none

                      tokens.append(pos_token_list)
                      # Reset current track ID as we are at a new position
                      current_track_id = "unknown" if _using_track_names else -2

                 else: # Same position index, but time might have advanced slightly within the tick window
                      current_tick = event_time # Update tick anyway


            # --- Process Event Type ---
            # Add TrackName or Program Token if the track context changes *for the current event*
            if event_track_id is not None and event_track_id != current_track_id:
                 current_track_id = event_track_id
                 # Check if the current event *is* the TrackName/Program event itself
                 # If so, we don't need to add a *separate* TrackName token step
                 id_token_type = "TrackName" if _using_track_names else "Program"
                 if event_type != id_token_type:
                     # Add TrackName or Program Token as a separate step BEFORE the current event
                     track_token_list = [None] * num_token_types
                     id_token_str = f"{id_token_type}_{current_track_id}"
                     track_token_list[self.vocab_types_idx[id_token_type]] = id_token_str
                     track_token_list[self.vocab_types_idx["BarPosEnc"]] = f"BarPosEnc_{current_bar}"
                     track_token_list[self.vocab_types_idx["PositionPosEnc"]] = f"PositionPosEnc_{current_pos}"
                     if self.config.use_tempos:
                         tempo_idx = self.vocab_types_idx["Tempo"]
                         list_idx = tempo_idx if tempo_idx >= 0 else num_token_types + tempo_idx
                         # <<< Use quantized current_tempo >>>
                         track_token_list[list_idx] = f"Tempo_{current_tempo}"
                     if self.config.use_velocities:
                         vel_idx = self.vocab_types_idx["Velocity"]
                         list_idx = vel_idx if vel_idx >= 0 else num_token_types + vel_idx
                         track_token_list[list_idx] = velocity_none
                     if self.config.using_note_duration_tokens:
                         dur_idx = self.vocab_types_idx["Duration"]
                         list_idx = dur_idx if dur_idx >= 0 else num_token_types + dur_idx
                         track_token_list[list_idx] = duration_none
                     tokens.append(track_token_list)


            # --- Create the token list for the current event ---
            # Skip if event type is Velocity or Duration, as they are now handled with Pitch
            if event_type in ["Velocity", "Duration"]:
                 continue

            event_token_list = [None] * num_token_types

            # Add positional encodings and tempo (always present)
            event_token_list[self.vocab_types_idx["BarPosEnc"]] = f"BarPosEnc_{current_bar}"
            event_token_list[self.vocab_types_idx["PositionPosEnc"]] = f"PositionPosEnc_{current_pos}"
            if self.config.use_tempos:
                 tempo_idx = self.vocab_types_idx["Tempo"]
                 list_idx = tempo_idx if tempo_idx >= 0 else num_token_types + tempo_idx
                 # <<< Use quantized current_tempo >>>
                 event_token_list[list_idx] = f"Tempo_{current_tempo}"


            # Add the primary event token (Pitch, CCType, TrackName etc.) to slot 0
            primary_token_str = f"{event_type}_{event_value}"
            # Ensure the type exists in the vocab map (might fail for PAD etc if not handled)
            if event_type in self.vocab_types_idx:
                 event_token_list[self.vocab_types_idx[event_type]] = primary_token_str
            else:
                 self.logger.warning(f"Event type '{event_type}' not found in vocab_types_idx. Skipping primary token.")
                 continue # Skip this event if its primary type isn't known

            # Add optional Velocity/Duration based on event type
            if event_type in ["Pitch", "PitchDrum"]:
                 # Use the velocity/duration extracted from the event's desc field
                 if self.config.use_velocities:
                     vel_idx = self.vocab_types_idx["Velocity"]
                     list_idx = vel_idx if vel_idx >= 0 else num_token_types + vel_idx
                     if note_velocity is not None:
                          # <<< Re-apply Fix: Quantize extracted velocity >>>
                          quantized_velocity = min(self.velocities, key=lambda x: abs(x - note_velocity))
                          event_token_list[list_idx] = f"Velocity_{quantized_velocity}"
                     else: # Should only happen if use_velocities=False, but as safeguard
                          event_token_list[list_idx] = velocity_none
                 if self.config.using_note_duration_tokens:
                     dur_idx = self.vocab_types_idx["Duration"]
                     list_idx = dur_idx if dur_idx >= 0 else num_token_types + dur_idx
                     if note_duration_str is not None:
                          # note_duration_str already contains the token value string (e.g., "0.1.4")
                          event_token_list[list_idx] = f"Duration_{note_duration_str}"
                     else: # Should only happen if using_note_duration_tokens=False
                          event_token_list[list_idx] = duration_none

            elif event_type in ["CCType", "CCValue", "TrackName", "Program", "Position", "Bar", "Tempo"]:
                 # Use None placeholders for V/D for non-note events
                 if self.config.use_velocities:
                     vel_idx = self.vocab_types_idx["Velocity"]
                     list_idx = vel_idx if vel_idx >= 0 else num_token_types + vel_idx
                     event_token_list[list_idx] = velocity_none
                 if self.config.using_note_duration_tokens:
                     dur_idx = self.vocab_types_idx["Duration"]
                     list_idx = dur_idx if dur_idx >= 0 else num_token_types + dur_idx
                     event_token_list[list_idx] = duration_none
            # Add handling for other event types if necessary (e.g., Chord, Rest)

            # Append the constructed token list
            tokens.append(event_token_list)

        # --- Finalization ---
        # Remove the previous error log about broken association
        # self.logger.error("MuMIDI _score_to_tokens needs fixing: Velocity/Duration association with Pitch is broken.")


        tok_sequence = TokSequence(tokens=tokens)
        self.complete_sequence(tok_sequence) # Adds EOS/PAD if needed based on config
        return tok_sequence

    def _tokens_to_score(
        self,
        tokens: TokSequence,
        _: None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        Handles TrackName mapping and CC filtering based on the enhanced format.

        :param tokens: MuMIDI token sequence to convert.
        :param _: unused placeholder.
        :return: the ``symusic.Score`` object.
        """
        # Check if tokens is empty
        if not tokens or not tokens.tokens:
            return Score(self.time_division)

        score = Score(self.time_division)
        _using_track_names = self.config.additional_params.get('use_track_names', True)
        num_bins = self.config.additional_params.get('num_cc_bins', DEFAULT_CC_BINS)

        # Determine indices for optional data from self.vocab_types_idx
        # Need to map public index (e.g., -1) to list index
        num_pos_idx = 0
        num_neg_idx = 0
        if self.vocab_types_idx:
             num_pos_idx = max((idx for idx in self.vocab_types_idx.values() if idx >= 0), default=0) + 1
             num_neg_idx = abs(min((idx for idx in self.vocab_types_idx.values() if idx < 0), default=-1))
        num_token_types = num_pos_idx + num_neg_idx

        def get_list_index(public_idx: int) -> int | None:
            if public_idx >= 0:
                return public_idx
            else:
                list_idx = num_token_types + public_idx
                return list_idx if 0 <= list_idx < num_token_types else None

        vel_list_idx = get_list_index(self.vocab_types_idx.get("Velocity")) if self.config.use_velocities else None
        dur_list_idx = get_list_index(self.vocab_types_idx.get("Duration")) if self.config.using_note_duration_tokens else None
        tempo_list_idx = get_list_index(self.vocab_types_idx.get("Tempo")) if self.config.use_tempos else None

        # Get the first tempo token
        first_tempo = self.default_tempo
        if self.config.use_tempos and tempo_list_idx is not None:
            try:
                tempo_token = tokens.tokens[0][tempo_list_idx]
                if tempo_token is not None and tempo_token.startswith("Tempo_"):
                    first_tempo = float(tempo_token.split("_")[1])
            except (IndexError, ValueError, TypeError):
                self.logger.warning("Could not parse first tempo token. Using default.")
        score.tempos.append(Tempo(0, first_tempo))

        # track_data: key is track_id (family or program), value is list of events (Note, CC)
        tracks_data: Dict[str | int, Dict[str, list]] = {}
        current_tick = 0
        current_bar = -1
        # Determine initial track id based on config
        current_track_id: str | int = "unknown" if _using_track_names else 0 # Default
        ticks_per_beat = score.ticks_per_quarter # Assume 4/4 for beat calculations

        # Store pending CCType to pair with CCValue
        pending_cc_type: int | None = None

        for time_step in tokens.tokens:
            # Ensure timestep has the expected structure (list)
            if not isinstance(time_step, list) or len(time_step) != num_token_types:
                self.logger.warning(f"Malformed time step found: {time_step}. Skipping.")
                continue

            # Extract primary token (Slot 0)
            primary_token_str = time_step[0]
            if primary_token_str is None:
                # This might happen with PAD tokens or errors
                continue

            try:
                tok_type, tok_val = primary_token_str.split("_", 1)
            except ValueError:
                # Handle cases like Bar_None or potentially invalid tokens
                if primary_token_str == "Bar_None":
                    tok_type = "Bar"
                    tok_val = "None"
                else:
                    self.logger.warning(f"Skipping invalid token format: {primary_token_str}")
                    continue

            # --- Track Time --- #
            if tok_type == "Bar":
                current_bar += 1
                current_tick = current_bar * ticks_per_beat * 4
                pending_cc_type = None # Reset pending CC on new bar
            elif tok_type == "Position":
                if current_bar == -1: current_bar = 0
                try:
                     # MuMIDI pos index = tick within bar / ticks_per_sample
                     # To get tick: bar_start_tick + pos_index * ticks_per_sample
                     # ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
                     ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
                     current_tick = (current_bar * ticks_per_beat * 4) + int(int(tok_val) * ticks_per_sample)
                except ValueError:
                     self.logger.warning(f"Invalid Position value: {tok_val}. Skipping time update.")
                     continue
                pending_cc_type = None # Reset pending CC on new position

            # --- Track Identifier --- #
            elif tok_type == "TrackName":
                current_track_id = tok_val
                pending_cc_type = None # Reset pending CC on track change
            elif tok_type == "Program": # Fallback if not using track names
                 try:
                      current_track_id = int(tok_val)
                 except ValueError:
                      self.logger.warning(f"Invalid Program value: {tok_val}. Using 0.")
                      current_track_id = 0
                 pending_cc_type = None # Reset pending CC on track change

            # --- Note Event --- #
            elif tok_type in {"Pitch", "PitchDrum"}:
                pitch = int(tok_val)

                # Extract Velocity
                vel = DEFAULT_VELOCITY
                if self.config.use_velocities and vel_list_idx is not None:
                    vel_token = time_step[vel_list_idx]
                    if vel_token is not None and vel_token.startswith("Velocity_"):
                        try: vel = int(vel_token.split("_")[1])
                        except (ValueError, IndexError): pass # Keep default
                    # Allow None velocity? For now, use default if token invalid.

                # Extract Duration
                # --- MODIFICATION: Ensure default duration is int ticks ---
                default_duration_ticks = int(round(self.config.default_note_duration * ticks_per_beat)) # Calculate default in ticks and ensure int
                duration = default_duration_ticks # Default
                # --- END MODIFICATION ---
                if self.config.using_note_duration_tokens and dur_list_idx is not None:
                     dur_token = time_step[dur_list_idx]
                     if dur_token is not None and dur_token.startswith("Duration_"):
                          dur_val_str = dur_token.split("_", 1)[1]
                          try:
                              # --- MODIFICATION: Ensure lookup result is int ---
                              duration = int(self._tpb_tokens_to_ticks[ticks_per_beat][dur_val_str])
                              # --- END MODIFICATION ---
                          except KeyError:
                               self.logger.warning(f"Unknown duration token '{dur_val_str}' at tick {current_tick}. Using default ticks: {default_duration_ticks}.")
                               duration = default_duration_ticks # Use int default ticks
                     # Allow None duration? For now, use default if token invalid.


                # Add note to the correct track's data
                if current_track_id not in tracks_data:
                    tracks_data[current_track_id] = {'notes': [], 'controls': []}
                # --- MODIFICATION: Ensure duration passed to Note() is int ---
                tracks_data[current_track_id]['notes'].append(Note(current_tick, int(duration), pitch, vel))
                # --- END MODIFICATION ---
                pending_cc_type = None # Reset pending CC after a note

            # --- CC Events --- #
            elif tok_type == "CCType":
                try:
                     pending_cc_type = int(tok_val)
                except ValueError:
                     self.logger.warning(f"Invalid CCType value: {tok_val}. Ignoring CC event.")
                     pending_cc_type = None
            elif tok_type == "CCValue":
                if pending_cc_type is not None:
                    try:
                        binned_value = int(tok_val)
                        # Un-bin value: value = floor(binned_value * 128 / num_bins)
                        cc_value = int((binned_value / (num_bins - 1)) * 127) if num_bins > 1 else (127 if binned_value > 0 else 0)
                        cc_value = min(max(cc_value, 0), 127) # Clamp to 0-127

                        # Add control change to the correct track's data
                        if current_track_id not in tracks_data:
                             tracks_data[current_track_id] = {'notes': [], 'controls': []}
                        tracks_data[current_track_id]['controls'].append(
                             ControlChange(current_tick, pending_cc_type, cc_value)
                        )
                    except ValueError:
                        self.logger.warning(f"Invalid CCValue: {tok_val}. Ignoring CC event.")
                    finally:
                        pending_cc_type = None # Reset after processing or error
                # else: Ignore CCValue if no CCType was pending

            # --- Decode Tempo --- #
            if self.config.use_tempos and tempo_list_idx is not None:
                tempo_token = time_step[tempo_list_idx]
                if tempo_token is not None and tempo_token.startswith("Tempo_"):
                    try:
                        tempo_val = float(tempo_token.split("_")[1])
                        # Add tempo change only if different from the last one
                        if not score.tempos or tempo_val != score.tempos[-1].tempo:
                             # Add tempos associated with the *next* tick start if possible?
                             # MuMIDI tempo seems associated with the Bar/Position step.
                             # Let's add it at current_tick.
                             score.tempos.append(Tempo(current_tick, tempo_val))
                    except (ValueError, IndexError):
                         self.logger.warning(f"Invalid Tempo token: {tempo_token}. Skipping.")

        # --- Create symusic.Track objects --- #
        for track_id, track_data in tracks_data.items():
            program = 0
            is_drum = False
            track_name = f"Track_{track_id}"

            if _using_track_names:
                 # track_id is the family name (string)
                 track_name = track_id # Use family name as track name
                 try:
                     program, is_drum = get_program_from_instrument_family(track_id)
                 except ValueError:
                      self.logger.warning(f"Unknown instrument family: {track_id}. Using default Piano.")
                      program = 0 # Default to piano
                      is_drum = False
            else:
                 # track_id is the program number (int)
                 program = track_id
                 if program == -1:
                     is_drum = True
                     program = 0 # symusic uses program 0 for drums
                     track_name = "Drums"
                 else:
                     track_name = MIDI_INSTRUMENTS[program]["name"]

            # Skip empty tracks?
            if not track_data['notes'] and not track_data['controls']:
                continue

            new_track = Track(name=track_name, program=program, is_drum=is_drum)
            new_track.notes = sorted(track_data['notes'], key=lambda x: (x.time, x.pitch))
            new_track.controls = sorted(track_data['controls'], key=lambda x: (x.time, x.number))
            score.tracks.append(new_track)

        # Sort tracks and tempo changes
        score.tracks.sort(key=lambda x: (x.is_drum, x.program))
        score.tempos.sort(key=lambda x: x.time)
        # Remove duplicate tempos at the same time, keeping the last one
        if len(score.tempos) > 1:
            unique_tempos = []
            last_tempo = score.tempos[0]
            unique_tempos.append(last_tempo)
            for i in range(1, len(score.tempos)):
                if score.tempos[i].time > last_tempo.time:
                    unique_tempos.append(score.tempos[i])
                    last_tempo = score.tempos[i]
                elif score.tempos[i].time == last_tempo.time: # Same time, replace last
                     unique_tempos[-1] = score.tempos[i]
                     last_tempo = score.tempos[i]
            score.tempos = unique_tempos


        return score

    def __getitem__(self, item: str | tuple[int, str]) -> int:
        r"""
        Convert a token (string) to its corresponding id.

        Overridden for MuMIDI to handle both tuple `(vocab_idx, token)` and simple
        `token` lookups (trying slot 0 or special tokens).

        :param item: token to convert.
        :return: the corresponding token id.
        """
        if isinstance(item, str):
            # Handle simple string lookup attempts
            # 1. Try base class lookup (handles special tokens correctly if they are in _vocab_base)
            try:
                return super().__getitem__(item)
            except KeyError:
                # 2. If not found by base class (e.g., not a registered special token),
                #    try assuming it belongs to the primary vocabulary slot (index 0).
                try:
                    # Ensure vocab structure is list-like and index 0 exists
                    if isinstance(self.vocab, list) and len(self.vocab) > 0:
                        return self.vocab[0][item]
                    else:
                        # This case should ideally not happen if vocab is built correctly
                        raise KeyError(f"MuMIDI vocabulary structure invalid or empty, cannot lookup token '{item}' in slot 0.")
                except (KeyError, TypeError) as e2:
                    # Raise error if not found in slot 0 either
                    raise KeyError(
                        f"Token '{item}' not found via base lookup or in vocabulary slot 0."
                    ) from e2
                except IndexError:
                     raise KeyError(
                        f"MuMIDI vocabulary structure invalid (index 0 out of bounds), cannot lookup token '{item}'."
                    )

        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], str):
            # Handle tuple lookup: (vocab_idx, token)
            try:
                vocab_idx, token = item
                # Adjust negative index for list access if necessary
                num_vocabs = len(self.vocab) if isinstance(self.vocab, list) else 0
                if num_vocabs == 0:
                    raise KeyError(f"MuMIDI vocabulary is empty, cannot lookup token '{token}' in slot {vocab_idx}.")

                list_idx = vocab_idx if vocab_idx >= 0 else num_vocabs + vocab_idx

                # Check if the calculated list_idx is valid
                if not (0 <= list_idx < num_vocabs):
                     raise IndexError(f"Vocabulary index {vocab_idx} (resolved to {list_idx}) is out of bounds for MuMIDI vocab size {num_vocabs}.")

                return self.vocab[list_idx][token]
            except (KeyError, TypeError) as e:
                # Provide context for tuple lookup failure
                raise KeyError(f"Error accessing token '{item[1]}' in vocab slot {item[0]}: {e}") from e
            except IndexError as e: # Catch index errors from list_idx calculation or access
                 raise IndexError(f"Error accessing vocab slot {item[0]} for token '{item[1]}': {e}") from e
            except Exception as e: # Catch other potential errors
                raise ValueError(f"Invalid item or error during MuMIDI tuple lookup: {item}. Error: {e}") from e
        else:
            raise TypeError(f"Invalid item type or format for MuMIDI vocabulary lookup: {item}. Expected str or (int, str).")

    def _create_base_vocabulary(self) -> list[list[str]]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        For **Enhanced MuMIDI**, the structure is defined by `self.vocab_types_idx`.
        Special tokens are added to all sub-vocabularies.

        :return: the vocabulary as a list of lists of strings.
        """
        # Determine the number of distinct vocabulary sets needed based on vocab_types_idx
        max_pos_idx = 0
        min_neg_idx = 0
        if self.vocab_types_idx:
             max_pos_idx = max((idx for idx in self.vocab_types_idx.values() if idx >= 0), default=0)
             min_neg_idx = min((idx for idx in self.vocab_types_idx.values() if idx < 0), default=-1)
        num_vocabs = max_pos_idx + 1 + abs(min_neg_idx)
        vocab = [[] for _ in range(num_vocabs)]

        # Helper function to add tokens to the correct list based on vocab_types_idx
        def add_to_vocab(token_type: str, tokens: list[str]):
            if token_type in self.vocab_types_idx:
                idx = self.vocab_types_idx[token_type]
                list_idx = idx if idx >= 0 else num_vocabs + idx
                if 0 <= list_idx < num_vocabs:
                    vocab[list_idx].extend(tokens)
                else:
                    self.logger.warning(f"Calculated invalid list index {list_idx} for token type {token_type} (idx={idx}). Skipping.")

        # --- Add Type/Value Tokens ---
        # SLOT 0: Primary event types
        pitch_tokens = [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        add_to_vocab("Pitch", pitch_tokens)
        pitch_drum_tokens = [f"PitchDrum_{i}" for i in range(*self.config.drums_pitch_range)]
        add_to_vocab("PitchDrum", pitch_drum_tokens)

        bar_token = ["Bar_None"] # MuMIDI uses Bar_None
        add_to_vocab("Bar", bar_token)

        # Determine max number of positions needed
        max_num_beats = 4 # Assume 4/4
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        position_tokens = [f"Position_{i}" for i in range(num_positions)]
        add_to_vocab("Position", position_tokens)

        # Add TrackName OR Program tokens
        if self.config.additional_params.get('use_track_names', True):
            track_name_tokens = [f"TrackName_{family}" for family in TARGET_INSTRUMENT_FAMILIES]
            track_name_tokens.append("TrackName_unknown")
            add_to_vocab("TrackName", track_name_tokens)
        else:
            program_tokens = [f"Program_{program}" for program in self.config.programs]
            add_to_vocab("Program", program_tokens)

        # Add CCType and CCValue if enabled
        if self.config.additional_params.get('use_filtered_ccs', True):
            cc_type_tokens = [f"CCType_{i}" for i in range(128)]
            add_to_vocab("CCType", cc_type_tokens)
            num_bins = self.config.additional_params.get('num_cc_bins', DEFAULT_CC_BINS)
            cc_value_tokens = [f"CCValue_{i}" for i in range(num_bins)]
            add_to_vocab("CCValue", cc_value_tokens)

        # SLOT 1: BarPosEnc
        bar_pos_enc_tokens = [
            f"BarPosEnc_{i}"
            for i in range(self.config.additional_params.get("max_bar_embedding", 256))
        ]
        add_to_vocab("BarPosEnc", bar_pos_enc_tokens)

        # SLOT 2: PositionPosEnc
        pos_pos_enc_tokens = ["PositionPosEnc_None"]
        pos_pos_enc_tokens += [f"PositionPosEnc_{i}" for i in range(num_positions)]
        add_to_vocab("PositionPosEnc", pos_pos_enc_tokens)

        # Optional types (Tempo, Velocity, Duration, Chord, Rest)
        if self.config.use_tempos:
            tempo_tokens = [f"Tempo_{i}" for i in self.tempos]
            add_to_vocab("Tempo", tempo_tokens)

        if self.config.use_velocities:
            velocity_tokens = [f"Velocity_{i}" for i in self.velocities]
            add_to_vocab("Velocity", velocity_tokens)

        if self.config.using_note_duration_tokens:
            duration_tokens = [f'Duration_{".".join(map(str, duration))}' for duration in self.durations]
            add_to_vocab("Duration", duration_tokens)

        if self.config.use_chords:
            chord_tokens = self._create_chords_tokens()
            add_to_vocab("Chord", chord_tokens)

        if self.config.use_rests:
            rest_tokens = [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]
            add_to_vocab("Rest", rest_tokens)


        # --- Explicitly add special tokens to ALL sub-vocabularies ---
        special_tokens_to_add = self.config.special_tokens
        for i in range(num_vocabs):
             existing_tokens = set(vocab[i])
             for sp_tok in special_tokens_to_add:
                 if sp_tok not in existing_tokens:
                     vocab[i].append(sp_tok)


        # Validate vocabulary sizes (optional sanity check)
        for i, v in enumerate(vocab):
             if not v:
                 missing_type = "Unknown"
                 for token_type, idx in self.vocab_types_idx.items():
                     list_idx = idx if idx >= 0 else num_vocabs + idx
                     if list_idx == i:
                         missing_type = token_type
                         break
                 self.logger.warning(f"Vocabulary subset at index {i} (expected type: {missing_type}) is empty.")

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = {
            "Bar": {"Bar", "Position"},
            "Position": {"Program"},
            "Program": {"Pitch", "PitchDrum"},
            "Pitch": {"Pitch", "Program", "Bar", "Position"},
            "PitchDrum": {"PitchDrum", "Program", "Bar", "Position"},
        }

        if self.config.use_chords:
            dic["Program"] |= {"Chord"}
            dic["Chord"] = {"Pitch"}

        return dic

    def _tokens_errors(self, tokens: list[list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        This method is intended to be overridden by tokenizer classes. The
        implementation in the ``MusicTokenizer`` class will check token types,
        duplicated notes and time errors. It works for ``REMI``, ``TSD`` and
        ``Structured``.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err = 0
        previous_type = tokens[0][0].split("_")[0]
        current_pitches = []
        current_bar = int(tokens[0][1].split("_")[1])
        current_pos = tokens[0][2].split("_")[1]
        current_pos = int(current_pos) if current_pos != "None" else -1

        for token in tokens[1:]:
            bar_value = int(token[1].split("_")[1])
            pos_value = token[2].split("_")[1]
            pos_value = int(pos_value) if pos_value != "None" else -1
            token_type, token_value = token[0].split("_")

            if any(tok.split("_")[0] in ["PAD", "MASK"] for i, tok in enumerate(token)):
                err += 1
                continue

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Bar":
                    current_bar += 1
                    current_pos = -1
                    current_pitches = []
                elif self.config.remove_duplicated_notes and token_type == "Pitch":
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == "Position":
                    if int(token_value) <= current_pos or int(token_value) != pos_value:
                        err += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = []
                elif token_type == "Program":
                    current_pitches = []

                if pos_value < current_pos or bar_value < current_bar:
                    err += 1
            # Bad token type
            else:
                err += 1

            previous_type = token_type
        return err
