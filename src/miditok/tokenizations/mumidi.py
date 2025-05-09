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
        self.logger.setLevel(logging.DEBUG) # <<< CHANGE: Set to DEBUG for these tests

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
            # "Tempo": 3, # <<< REMOVED from dedicated slot
        }
        # Add TrackName if used (or Program as fallback) to slot 0
        if self.config.additional_params.get('use_track_names', True):
             self.vocab_types_idx["TrackName"] = 0
        else:
             self.vocab_types_idx["Program"] = 0

        # Add CC types if used to slot 0 # <<< CHANGE: Will map to dedicated slots
        # if self.config.additional_params.get('use_filtered_ccs', True):
        #      self.vocab_types_idx["CCType"] = 0
        #      self.vocab_types_idx["CCValue"] = 0

        # Add optional token types, adjusting indices based on what's present
        current_neg_idx = -1
        if self.config.using_note_duration_tokens:
             self.vocab_types_idx["Duration"] = current_neg_idx
             current_neg_idx -= 1
        if self.config.use_velocities:
             self.vocab_types_idx["Velocity"] = current_neg_idx
             current_neg_idx -= 1 # Decrement after Velocity
        # <<< NEW: Assign dedicated negative slots for CCs if used >>>
        if self.config.additional_params.get('use_filtered_ccs', True):
             self.vocab_types_idx["CCType"] = current_neg_idx
             current_neg_idx -= 1
             self.vocab_types_idx["CCValue"] = current_neg_idx
             # current_neg_idx -= 1 # Decrement only if more types follow

        # Chord and Rest will also go to slot 0 if enabled
        if self.config.use_chords:
             self.vocab_types_idx["Chord"] = 0
        if self.config.use_rests:
             self.vocab_types_idx["Rest"] = 0
        # If using tempos, and they are to be event-like, map "Tempo" type to vocab slot 0
        if self.config.use_tempos:
            self.vocab_types_idx["Tempo"] = 0

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
             pass # Skipping full chord implementation for now


        # --- MODIFIED: Generate single CC events --- #
        if self.use_filtered_ccs and track.controls:
             cc_events = self._create_control_change_events_bundled(track.controls, track_family)
             # Assign the track_id to the desc field for sorting/grouping CC events
             for cc_event in cc_events:
                 cc_event.desc = str(track_id)  # Store as string for consistency
             events.extend(cc_events)
        # --- END MODIFICATION --- #

        return events


    def _create_control_change_events_filtered(self, controls: List[ControlChange], track_family: str) -> List[Event]:
        """
        DEPRECATED: Generates separate CCType and CCValue events.
        Use _create_control_change_events_bundled instead.
        """
        # Keep the old function signature but log warning and call new one?
        # Or just remove/rename this one? Let's keep signature for now but make it call bundled.
        self.logger.warning("_create_control_change_events_filtered is deprecated, use _create_control_change_events_bundled.")
        return self._create_control_change_events_bundled(controls, track_family)

    def _create_control_change_events_bundled(self, controls: List[ControlChange], track_family: str) -> List[Event]:
        """
        Generates single 'ControlChange' Event objects based on filtering rules,
        bundling CC number and binned value.

        :param controls: List of ControlChange objects from the track.
        :param track_family: The instrument family of the current track.
        :return: List of 'ControlChange' Events.
        """
        events = []
        if not controls:
            return events

        allowed_ccs: Set[int]
        program, _ = get_program_from_instrument_family(track_family)
        inst_type = get_instrument_type_from_program(program)

        if self.cc_filter_mode == 'all':
            allowed_ccs = set(range(128))
        elif self.cc_filter_mode == 'explicit_list':
            allowed_ccs = self.explicit_cc_list
        elif self.cc_filter_mode == 'instrument_aware':
             allowed_ccs = self.cc_instrument_map.get(track_family, self.cc_instrument_map.get(inst_type, self.cc_instrument_map["unknown"]))
        else: # 'none' or invalid mode
            return events

        num_bins = self.config.additional_params.get('num_cc_bins', DEFAULT_CC_BINS)

        for control in controls:
            if control.number in allowed_ccs:
                value = min(max(control.value, 0), 127)
                binned_value = int((value / 127.0) * (num_bins - 1))
                # Create ONE event bundling number and binned value
                event_value = (control.number, binned_value)
                events.append(Event(type_="ControlChange", value=event_value, time=control.time))

        return events


    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> list[list[str | None]]:
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
        _using_track_names = self.config.additional_params.get('use_track_names', True)

        # Add actual Tempo events to all_track_events
        if self.config.use_tempos and score.tempos:
            for tempo_obj in score.tempos:
                raw_tempo = round(tempo_obj.tempo, 2) # As per original quantization logic
                quantized_tempo = min(self.tempos, key=lambda x: abs(x - raw_tempo))
                all_track_events.append(Event(type_="Tempo", value=str(quantized_tempo), time=tempo_obj.time, desc="TempoEvent"))

        for track in score.tracks:
             track_id: str | int
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
            "Tempo": 2, # Use event-based tempo
            "ControlChange": 3, # ADDED: Priority for the bundled CC event
        }
        # --- Sort All Events ---
        # Sort by time, then priority, then value (for stable sort within same time/priority)
        all_track_events.sort(key=lambda x: (x.time, event_priority.get(x.type_, 10), x.value))

        self.logger.debug(f"MUMIDI DEBUG: all_track_events before main loop ({len(all_track_events)} items): {all_track_events[:20]}") # ADDED: Print first 20 events

        # --- Build Token Sequence ---
        tokens: list[list[str | None]] = []
        ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
        ticks_per_bar = score.ticks_per_quarter * 4 # Assume 4/4

        # Pre-calculate vocabulary sizes and indices for efficiency
        num_pos_idx = 0
        num_neg_idx = 0
        if self.vocab_types_idx:
             max_pos_idx = max((idx for idx in self.vocab_types_idx.values() if idx >= 0), default=0)
             min_neg_idx = min((idx for idx in self.vocab_types_idx.values() if idx < 0), default=-1)
             num_token_types = max_pos_idx + 1 + abs(min_neg_idx)
        else: # Should not happen if tokenizer is initialized correctly
            self.logger.error("vocab_types_idx is missing or empty! Cannot determine num_token_types.")
            return [] # Return empty sequence

        def get_list_idx(token_type: str) -> int | None:
            public_idx = self.vocab_types_idx.get(token_type)
            if public_idx is None: return None
            list_idx = public_idx if public_idx >= 0 else num_token_types + public_idx
            return list_idx if 0 <= list_idx < num_token_types else None

        bar_pos_enc_idx = get_list_idx("BarPosEnc")
        pos_pos_enc_idx = get_list_idx("PositionPosEnc")
        vel_list_idx = get_list_idx("Velocity")
        dur_list_idx = get_list_idx("Duration")
        cc_type_list_idx = get_list_idx("CCType")
        cc_val_list_idx = get_list_idx("CCValue")
        track_name_list_idx = get_list_idx("TrackName")
        program_list_idx = get_list_idx("Program")
        tempo_list_idx = get_list_idx("Tempo") # Mapped to 0
        bar_list_idx = get_list_idx("Bar")       # Mapped to 0
        pos_list_idx = get_list_idx("Position")   # Mapped to 0
        pitch_list_idx = get_list_idx("Pitch")     # Mapped to 0
        pitch_drum_list_idx = get_list_idx("PitchDrum") # Mapped to 0

        # --- Get Pad Token directly from config special tokens --- #
        # Find the token string starting with 'PAD' in special_tokens
        pad_token_str = self.config.special_tokens[0] # Get PAD token string (e.g., "PAD_None")
        if hasattr(self.config, 'special_tokens') and isinstance(self.config.special_tokens, list):
            found_pad = [tok for tok in self.config.special_tokens if tok.startswith("PAD")]
            if found_pad:
                pad_token_str = found_pad[0]
            else:
                self.logger.warning("PAD token string not found in config.special_tokens, using default 'PAD_None'")
        else:
            self.logger.warning("config.special_tokens not found or not a list, using default 'PAD_None' for pad token string")
        # --- Use pad_token_str where pad_token was used below --- #

        _using_track_names = self.config.additional_params.get('use_track_names', True)

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track_id: str | int = "unknown" if _using_track_names else -2
        if not _using_track_names: current_track_id = -2

        velocity_none_str = f"Velocity_{min(self.velocities, key=lambda x: abs(x - DEFAULT_VELOCITY))}" if self.config.use_velocities and hasattr(self, 'velocities') and self.velocities is not None and self.velocities.size > 0 else pad_token_str
        default_dur_val = self.durations[0] if self.config.using_note_duration_tokens and self.durations else (0,0)
        duration_none_str = f'Duration_{".".join(map(str, default_dur_val))}' if self.config.using_note_duration_tokens and self.durations else pad_token

        # --- Main loop through sorted events ---
        for event in all_track_events:
            event_time = event.time
            event_type = event.type_
            event_value = event.value
            
            event_track_id = None; note_velocity = None; note_duration_str = None
            if event_type in ["Pitch", "PitchDrum"]:
                try:
                    if isinstance(event.desc, str) and "|" in event.desc:
                        parts = event.desc.split("|")
                        note_velocity = int(parts[0]) if parts[0] and parts[0] != 'None' else None
                        note_duration_str = parts[1] if parts[1] else None
                        event_track_id = parts[2]
                        if not _using_track_names and isinstance(event_track_id, str) and event_track_id.lstrip('-').isdigit(): event_track_id = int(event_track_id)
                    else: raise TypeError(f"event.desc format error: {event.desc}")
                except Exception as e: self.logger.warning(f"Desc unpack error for {event}: {e}"); event_track_id = "unknown" if _using_track_names else -1
            elif event_type == "ControlChange" or event_type == "TrackName" or (_using_track_names is False and event_type == "Program") or event_type == "Tempo":
                 event_track_id = event.desc # Tempo desc was "TempoEvent", TrackName desc is family, CC desc is track_id from _track_to_tokens
                 if event_type == "Tempo": event_track_id = current_track_id # Tempo events are global, associate with current track context for consistency in loop
                 elif isinstance(event_track_id, str) and not _using_track_names and event_track_id.lstrip('-').isdigit(): event_track_id = int(event_track_id)

            if event_time != current_tick:
                new_bar = event_time // ticks_per_bar
                if new_bar > current_bar:
                    for bar_idx in range(current_bar + 1, new_bar + 1):
                        bar_token_list = [pad_token_str] * num_token_types
                        if bar_list_idx is not None: bar_token_list[bar_list_idx] = "Bar_None"
                        if bar_pos_enc_idx is not None: bar_token_list[bar_pos_enc_idx] = f"BarPosEnc_{bar_idx}"
                        if pos_pos_enc_idx is not None: bar_token_list[pos_pos_enc_idx] = "PositionPosEnc_None"
                        if vel_list_idx is not None: bar_token_list[vel_list_idx] = velocity_none_str
                        if dur_list_idx is not None: bar_token_list[dur_list_idx] = duration_none_str
                        if cc_type_list_idx is not None: bar_token_list[cc_type_list_idx] = pad_token_str
                        if cc_val_list_idx is not None: bar_token_list[cc_val_list_idx] = pad_token_str
                        self.logger.debug(f"MUMIDI DEBUG: Appending BAR token list: {bar_token_list}")
                        tokens.append(bar_token_list)
                    current_bar = new_bar; current_pos = -1
                pos_index = int((event_time % ticks_per_bar) / ticks_per_sample)
                if pos_index != current_pos:
                    current_pos = pos_index; current_tick = event_time
                    pos_token_list = [pad_token_str] * num_token_types
                    if pos_list_idx is not None: pos_token_list[pos_list_idx] = f"Position_{current_pos}"
                    if bar_pos_enc_idx is not None: pos_token_list[bar_pos_enc_idx] = f"BarPosEnc_{current_bar}"
                    if pos_pos_enc_idx is not None: pos_token_list[pos_pos_enc_idx] = f"PositionPosEnc_{current_pos}"
                    if vel_list_idx is not None: pos_token_list[vel_list_idx] = velocity_none_str
                    if dur_list_idx is not None: pos_token_list[dur_list_idx] = duration_none_str
                    if cc_type_list_idx is not None: pos_token_list[cc_type_list_idx] = pad_token_str
                    if cc_val_list_idx is not None: pos_token_list[cc_val_list_idx] = pad_token_str
                    self.logger.debug(f"MUMIDI DEBUG: Appending POSITION token list: {pos_token_list}")
                    tokens.append(pos_token_list)
                    current_track_id = "unknown" if _using_track_names else -2 
            else: current_tick = event_time

            if event_track_id is not None and event_track_id != current_track_id and event_type not in ["Tempo"]:
                 current_track_id = event_track_id
                 id_token_type = "TrackName" if _using_track_names else "Program"
                 if event_type != id_token_type: 
                     track_token_list = [pad_token_str] * num_token_types
                     id_list_idx = track_name_list_idx if _using_track_names else program_list_idx
                     if id_list_idx is not None: track_token_list[id_list_idx] = f"{id_token_type}_{current_track_id}"
                     if bar_pos_enc_idx is not None: track_token_list[bar_pos_enc_idx] = f"BarPosEnc_{current_bar}"
                     if pos_pos_enc_idx is not None: track_token_list[pos_pos_enc_idx] = f"PositionPosEnc_{current_pos}"
                     if vel_list_idx is not None: track_token_list[vel_list_idx] = velocity_none_str
                     if dur_list_idx is not None: track_token_list[dur_list_idx] = duration_none_str 
                     if cc_type_list_idx is not None: track_token_list[cc_type_list_idx] = pad_token_str
                     if cc_val_list_idx is not None: track_token_list[cc_val_list_idx] = pad_token_str
                     self.logger.debug(f"MUMIDI DEBUG: Appending TRACK token list: {track_token_list}") 
                     tokens.append(track_token_list)
            
            event_token_list = [pad_token_str] * num_token_types
            if bar_pos_enc_idx is not None: event_token_list[bar_pos_enc_idx] = f"BarPosEnc_{current_bar}"
            if pos_pos_enc_idx is not None: event_token_list[pos_pos_enc_idx] = f"PositionPosEnc_{current_pos}"

            if event_type in ["Pitch", "PitchDrum"]:
                idx = pitch_list_idx if event_type == "Pitch" else pitch_drum_list_idx
                if idx is not None: event_token_list[idx] = f"{event_type}_{event_value}"
                if vel_list_idx is not None: event_token_list[vel_list_idx] = f"Velocity_{min(self.velocities, key=lambda x: abs(x - note_velocity))}" if note_velocity is not None else velocity_none_str
                if dur_list_idx is not None: event_token_list[dur_list_idx] = f"Duration_{note_duration_str}" if note_duration_str is not None else duration_none_str
                if cc_type_list_idx is not None: event_token_list[cc_type_list_idx] = pad_token_str
                if cc_val_list_idx is not None: event_token_list[cc_val_list_idx] = pad_token_str
                self.logger.debug(f"MUMIDI DEBUG: Appending PITCH/DRUM event token list: {event_token_list}") 
                tokens.append(event_token_list)
            elif event_type == "ControlChange":
                cc_num, binned_val = event_value
                if cc_type_list_idx is not None: event_token_list[cc_type_list_idx] = f"CCType_{cc_num}"
                if cc_val_list_idx is not None: event_token_list[cc_val_list_idx] = f"CCValue_{binned_val}"
                if vel_list_idx is not None: event_token_list[vel_list_idx] = velocity_none_str
                if dur_list_idx is not None: event_token_list[dur_list_idx] = duration_none_str
                self.logger.debug(f"MUMIDI DEBUG: Appending CC event token list: {event_token_list}") 
                tokens.append(event_token_list)
            elif event_type == "Tempo":
                if tempo_list_idx is not None: event_token_list[tempo_list_idx] = f"Tempo_{event_value}"
                if vel_list_idx is not None: event_token_list[vel_list_idx] = velocity_none_str
                if dur_list_idx is not None: event_token_list[dur_list_idx] = duration_none_str
                if cc_type_list_idx is not None: event_token_list[cc_type_list_idx] = pad_token_str
                if cc_val_list_idx is not None: event_token_list[cc_val_list_idx] = pad_token_str
                self.logger.debug(f"MUMIDI DEBUG: Appending TEMPO event token list: {event_token_list}") 
                tokens.append(event_token_list)
            elif event_type == "TrackName" or event_type == "Program": 
                idx = track_name_list_idx if event_type == "TrackName" else program_list_idx
                if idx is not None: event_token_list[idx] = f"{event_type}_{event_value}"
                if vel_list_idx is not None: event_token_list[vel_list_idx] = velocity_none_str
                if dur_list_idx is not None: event_token_list[dur_list_idx] = duration_none_str
                if cc_type_list_idx is not None: event_token_list[cc_type_list_idx] = pad_token_str
                if cc_val_list_idx is not None: event_token_list[cc_val_list_idx] = pad_token_str
                self.logger.debug(f"MUMIDI DEBUG: Appending TRACKNAME/PROGRAM event token list: {event_token_list}") 
                tokens.append(event_token_list)
            elif event_type not in ["Bar", "Position"]: 
                event_token_list[0] = f"{event_type}_{event_value}" 
                if vel_list_idx is not None: event_token_list[vel_list_idx] = velocity_none_str
                if dur_list_idx is not None: event_token_list[dur_list_idx] = duration_none_str
                if cc_type_list_idx is not None: event_token_list[cc_type_list_idx] = pad_token_str
                if cc_val_list_idx is not None: event_token_list[cc_val_list_idx] = pad_token_str
                self.logger.debug(f"MUMIDI DEBUG: Appending OTHER event token list ({event_type}): {event_token_list}") 
                tokens.append(event_token_list)

        self.logger.debug(f"MUMIDI DEBUG: Final raw token list: {tokens}") # Keep this debug log
        return tokens # <<< CHANGE 2: Return the raw list

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

        # Add CCType and CCValue if enabled # <<< CHANGE: Add to their dedicated slots
        if self.config.additional_params.get('use_filtered_ccs', True):
            cc_type_tokens = [f"CCType_{i}" for i in range(128)]
            add_to_vocab("CCType", cc_type_tokens) # Will use index from vocab_types_idx
            num_bins = self.config.additional_params.get('num_cc_bins', DEFAULT_CC_BINS)
            cc_value_tokens = [f"CCValue_{i}" for i in range(num_bins)]
            add_to_vocab("CCValue", cc_value_tokens) # Will use index from vocab_types_idx

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

        # Optional types (Velocity, Duration, Chord, Rest)
        # Tempo tokens are now added to slot 0
        if self.config.use_tempos:
            tempo_tokens = [f"Tempo_{i}" for i in self.tempos]
            add_to_vocab("Tempo", tempo_tokens) # This will add to slot 0 if "Tempo" is mapped to 0 in vocab_types_idx

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

    def _tokens_to_score(
        self,
        tokens: TokSequence,
        _: None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param _: in place of programs of the parent method, unused here.
            (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        score = Score(self.time_division)

        # Tempos
        if self.config.use_tempos and len(tokens) > 0:
            first_tempo = float(tokens.tokens[0][3].split("_")[1])
        else:
            first_tempo = self.default_tempo
        score.tempos.append(Tempo(0, first_tempo))

        tracks = {}
        current_tick = 0
        current_bar = -1
        current_track = 0  # default set to piano
        ticks_per_beat = score.ticks_per_quarter
        for time_step in tokens.tokens:
            tok_type, tok_val = time_step[0].split("_")
            if tok_type == "Bar":
                current_bar += 1
                current_tick = current_bar * ticks_per_beat * 4
            elif tok_type == "Position":
                if current_bar == -1:
                    current_bar = (
                        0  # as this Position token occurs before any Bar token
                    )
                current_tick = current_bar * ticks_per_beat * 4 + int(tok_val)
            elif tok_type == "Program":
                current_track = tok_val
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif tok_type in {"Pitch", "PitchDrum"}:
                vel = (
                    time_step[self.vocab_types_idx["Velocity"]].split("_")[1]
                    if self.config.use_velocities
                    else DEFAULT_VELOCITY
                )
                duration = (
                    time_step[-1].split("_")[1]
                    if self.config.using_note_duration_tokens
                    else int(self.config.default_note_duration * ticks_per_beat)
                )
                if any(val == "None" for val in (vel, duration)):
                    continue
                pitch = int(tok_val)
                vel = int(vel)
                if isinstance(duration, str):
                    duration = self._tpb_tokens_to_ticks[ticks_per_beat][duration]

                tracks[current_track].append(Note(current_tick, duration, pitch, vel))

            # Decode tempo if required
            if self.config.use_tempos:
                tempo_val = float(time_step[3].split("_")[1])
                if tempo_val != score.tempos[-1].tempo:
                    score.tempos.append(Tempo(current_tick, tempo_val))

        # Appends created notes to Score object
        for program, notes in tracks.items():
            if int(program) == -1:
                score.tracks.append(Track(name="Drums", program=0, is_drum=True))
            else:
                score.tracks.append(
                    Track(
                        name=MIDI_INSTRUMENTS[int(program)]["name"],
                        program=int(program),
                        is_drum=False,
                    )
                )
            score.tracks[-1].notes = notes

        return score

    def encode(self, score: Score | Path | str | bytes, *args, **kwargs) -> TokSequence:
        """
        Encodes a symusic.Score object into a MuMIDI TokSequence object.

        Overrides the base class encode method to handle MuMIDI's specific
        list[list[str]] -> list[list[int]] conversion.
        """
        # 1. Basic loading and preprocessing (borrowed from base class logic)
        if not isinstance(score, Score):
            # Attempt to load if path/str/bytes - requires symusic >= 0.3.0
            try:
                score = Score(score) 
            except Exception as e:
                raise ValueError(f"Could not load input score. Input type: {type(score)}. Error: {e}") from e
        
        # Preprocess the score (modifies in place)
        # Use self.preprocess_score which handles TPQ, quantization etc.
        preprocessed_score = self.preprocess_score(score)
        
        # Attribute controls are not used/compatible with MuMIDI's current logic
        attribute_controls_indexes = None 

        # 2. Call the MuMIDI-specific method to get the list[list[str]]
        # Assumes _score_to_tokens now returns list[list[str | None]]
        tokens_list_of_lists = self._score_to_tokens(preprocessed_score, attribute_controls_indexes) 

        # 3. --- MuMIDI-specific ID conversion ---
        ids_list_of_lists = []
        if not hasattr(self, 'vocab') or not isinstance(self.vocab, list) or not self.vocab:
             self.logger.error("MuMIDI vocabulary not properly initialized. Cannot convert tokens to IDs.")
             # Return an empty sequence or raise error? Let's return empty for now.
             return TokSequence(ids=[], tokens=[])

        num_token_types = len(self.vocab) # Get number of vocabularies/slots
        pad_token_str = self.config.special_tokens[0] # Get PAD token string (e.g., "PAD_None")

        for timestep_tokens in tokens_list_of_lists:
            timestep_ids = []
            if isinstance(timestep_tokens, list) and len(timestep_tokens) == num_token_types: # Sanity check
                for vocab_idx, token_str in enumerate(timestep_tokens):
                    if token_str is None: # Handle potential None values if they slip through
                       token_str = pad_token_str 
                    
                    try:
                        # Use the overridden __getitem__ which handles tuples (vocab_idx, token_str)
                        # vocab_idx corresponds to the index in the inner list
                        token_id = self[(vocab_idx, token_str)] 
                    except KeyError:
                        # Handle unknown tokens - map to PAD ID for that specific vocabulary slot
                        try:
                            pad_id = self[(vocab_idx, pad_token_str)]
                        except KeyError:
                            # Fallback if PAD token itself isn't in that specific vocab slot (shouldn't happen ideally)
                            self.logger.error(f"PAD token '{pad_token_str}' not found in vocab slot {vocab_idx}. Using 0 as fallback ID.")
                            pad_id = 0 
                        token_id = pad_id
                        # Only warn once per unknown token type per encoding run? Could get noisy.
                        # self.logger.warning(f"Unknown token '{token_str}' in vocab slot {vocab_idx}. Using PAD ID {pad_id}.")
                        
                    timestep_ids.append(token_id)
            else:
                self.logger.warning(f"Skipping timestep with unexpected format/length. Expected list of {num_token_types}, got: {timestep_tokens}")
                # Append a list of PAD IDs? Or skip entirely? Let's skip for now.
                continue # Skip this timestep
                
            ids_list_of_lists.append(timestep_ids)
        # --- End ID conversion ---

        # 4. Create the final TokSequence
        # Store list[list[str]] in .tokens and list[list[int]] in .ids
        tok_sequence = TokSequence(tokens=tokens_list_of_lists, ids=ids_list_of_lists)

        # 5. Add BOS/EOS - MuMIDI typically doesn't use them, but if needed:
        # Since .tokens and .ids are list[list], prepending/appending BOS/EOS
        # would require creating appropriate pooled token lists for BOS/EOS.
        # Example: bos_token_list = [self.bos_token] * num_token_types -> convert to IDs -> prepend
        # Skipping BOS/EOS addition for MuMIDI for now, as it might not be standard.

        return tok_sequence

