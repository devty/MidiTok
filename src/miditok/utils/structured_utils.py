"""
Utility functions for structured MIDI tokenization, including instrument mappings
and chord analysis, adapted for integration into MidiTok.
"""

import re
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from collections import defaultdict
import traceback

# MIDI parsing/handling libraries
import symusic
import mido

# Optional: music21 for chord analysis
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    music21 = None # Define as None if not available

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure this module's logger is at DEBUG level
logging.getLogger('mido').setLevel(logging.ERROR)
if MUSIC21_AVAILABLE:
    # Suppress music21 verbose output if desired
    # env = music21.environment.Environment()
    # env['warnings'] = 0
    pass

# --- Instrument Mapping Constants (from instrument_mappings.py) ---

INSTRUMENT_KEYWORDS = {
    "strings": ["violin", "viola", "cello", "bass", "strings", "str", "vln", "vla", "vc", "cb"],
    "brass": ["trumpet", "trombone", "horn", "tuba", "brass", "tpt", "tbn", "hn"],
    "woodwinds": ["flute", "oboe", "clarinet", "bassoon", "woodwind", "fl", "ob", "cl", "bsn"],
    "percussion": ["percussion", "timpani", "drums", "cymbals", "perc", "timp"],
    "piano": ["piano", "pno", "grand", "upright"],
    "choir": ["choir", "chorus", "vocals", "soprano", "alto", "tenor", "bass"],
    "synth": ["synth", "pad", "lead", "electronic"],
    "guitar": ["guitar", "gtr", "acoustic", "electric"]
}

PROGRAM_RANGES = {
    "piano": (0, 7),
    "chromatic percussion": (8, 15), # Renamed from percussion for clarity based on GM
    "organ": (16, 23),
    "guitar": (24, 31),
    "bass": (32, 39),
    "strings": (40, 47),
    "ensemble": (48, 55),
    "brass": (56, 63),
    "reed": (64, 71), # Renamed from woodwinds based on GM groups
    "pipe": (72, 79), # Renamed from woodwinds based on GM groups
    "synth lead": (80, 87),
    "synth pad": (88, 95),
    "synth effects": (96, 103),
    "ethnic": (104, 111),
    "percussive": (112, 119),
    "sound effects": (120, 127),
    # Add drum kit as a special case if needed, though program numbers overlap percussion
    "drum kit": (-1, -1) # Use -1 as a pseudo-program for drums
}


INSTRUMENT_CC_MAPPINGS = {
    # Using GM family names (mostly)
    "piano": {"essential": [64], "extended": [1, 7, 10, 11, 64, 66, 67, 91, 93]}, # Added Mod, Vol, Pan, Reverb, Chorus
    "chromatic percussion": {"essential": [11, 64], "extended": [1, 7, 10, 11, 64, 91, 93]}, # Vibraphone might use sustain
    "organ": {"essential": [11], "extended": [1, 7, 10, 11, 66, 91, 93]}, # Less sustain, maybe sostenuto, Leslie speed? (often not CC 1)
    "guitar": {"essential": [11, 64], "extended": [1, 7, 10, 11, 64, 65, 91, 93]}, # Added portamento, etc.
    "bass": {"essential": [11], "extended": [1, 7, 10, 11, 65, 91, 93]},
    "strings": {"essential": [1, 11, 64], "extended": [1, 7, 10, 11, 64, 65, 68, 91, 93]}, # Legato, etc.
    "ensemble": {"essential": [1, 11, 64], "extended": [1, 7, 10, 11, 64, 91, 93]}, # Generic ensemble
    "brass": {"essential": [1, 11], "extended": [1, 7, 10, 11, 65, 68, 91, 93]}, # Mutes often sys-ex, legato CC
    "reed": {"essential": [1, 11], "extended": [1, 7, 10, 11, 65, 91, 93]},
    "pipe": {"essential": [1, 11], "extended": [1, 7, 10, 11, 91, 93]}, # Flutes less common CCs
    "synth lead": {"essential": [1, 11], "extended": [1, 5, 7, 10, 11, 65, 71, 74, 91, 93]}, # Added Portamento, Filter Cutoff, Resonance
    "synth pad": {"essential": [1, 11], "extended": [1, 5, 7, 10, 11, 71, 74, 91, 93]},
    "synth effects": {"essential": [11], "extended": [1, 7, 10, 11, 71, 74, 91, 93]},
    "ethnic": {"essential": [1, 11, 64], "extended": [1, 7, 10, 11, 64, 91, 93]}, # Highly variable
    "percussive": {"essential": [11], "extended": [1, 7, 10, 11, 91, 93]}, # e.g., steel drums, etc.
    "sound effects": {"essential": [11], "extended": [1, 7, 10, 11, 91, 93]},
    "drum kit": {"essential": [], "extended": []}, # Drums typically don't use these CCs
    "unknown": {"essential": [1, 7, 11, 64], "extended": list(range(128))} # Default includes common, extended allows all
}

# --- Instrument Mapping Functions (from instrument_mappings.py) ---

def normalize_track_name(name: str) -> str:
    """
    Normalize a track name by removing special characters and converting to lowercase.
    """
    if name is None: return "unknown"
    # Convert to lowercase and remove special characters except underscore/hyphen
    normalized = re.sub(r'[^\w\s_-]', '', name.lower())
    # Replace multiple whitespace chars or underscores/hyphens with single underscore
    normalized = re.sub(r'[\s_-]+', '_', normalized).strip('_')
    if not normalized: return "unknown" # Handle empty names after normalization
    return normalized

def get_instrument_type_from_track_name(track_name: str) -> str:
    """
    Determine instrument type (GM family name) from track name using keyword matching.
    """
    if track_name is None: return "unknown"
    name_lower = track_name.lower()

    # Check each instrument family's keywords
    for family, keywords in INSTRUMENT_KEYWORDS.items():
        # Use word boundaries for more precise matching if needed (e.g., \bkeyword\b)
        if any(keyword in name_lower for keyword in keywords):
            # Map specific keyword match to broader GM family name if needed
            if family == "woodwinds": # Example: map woodwinds keyword match to reed/pipe?
                 # This is ambiguous, sticking to program ranges is safer for GM families
                 # For track name based, maybe return a more general "woodwind" type?
                 return "reed" # Default woodwind match to reed
            return family # Return the matched family name directly

    return "unknown"

def get_instrument_type_from_program(program: int) -> str:
    """
    Determine instrument type (GM family name) from MIDI program number.
    Returns 'drum kit' if program is -1 (common convention for drum tracks).
    """
    if program == -1: # Explicit check for drum track convention
        return "drum kit"
    for family, (start, end) in PROGRAM_RANGES.items():
        # Skip drum kit range here as we handled -1 explicitly
        if family == "drum kit": continue
        if start <= program <= end:
            return family

    return "unknown"

def get_relevant_ccs_for_instrument(instrument_type: str, extended: bool = False) -> List[int]:
    """
    Get list of relevant CC numbers for an instrument type (GM family name).
    """
    if instrument_type is None: instrument_type = "unknown"
    family_lower = instrument_type.lower()

    # Map potential aliases if needed (e.g., if track name gave "woodwind")
    if family_lower == "woodwind": family_lower = "reed" # Example mapping

    if family_lower in INSTRUMENT_CC_MAPPINGS:
        mapping = INSTRUMENT_CC_MAPPINGS[family_lower]
        return mapping["extended"] if extended else mapping["essential"]

    # Default CCs if instrument type unknown or not in map
    logger.debug(f"Instrument type '{instrument_type}' not found in CC mappings, returning default.")
    default_ccs = [1, 7, 10, 11, 64, 91, 93] # Mod, Vol, Pan, Expr, Sus, Reverb, Chorus
    return list(range(128)) if extended else default_ccs

# Define the target instrument families for TrackName tokens
TARGET_INSTRUMENT_FAMILIES = {
    "piano", "guitar", "bass", "strings", "brass", "reed", "pipe", # Keep GM groups where sensible
    "synth", # Combine lead/pad/effects
    "drums", # Use drums instead of drum kit for token
    "percussion", # Chromatic + Percussive
    "choir",
    "organ",
    "ensemble", # Generic ensemble
    "ethnic",
    "sound_effects", # Use underscore for consistency
    "other" # Fallback category
}

def map_track_name_to_family(track_name: Optional[str]) -> str:
    """
    Normalizes a track name and maps it to a predefined instrument family.
    Uses INSTRUMENT_KEYWORDS for matching.

    Args:
        track_name: The raw track name from the MIDI file.

    Returns:
        A string representing the mapped instrument family (e.g., "strings", "piano", "drums", "other").
    """
    if not track_name: return "other"
    name_norm = normalize_track_name(track_name)
    if name_norm == "unknown": return "other"

    name_lower = name_norm.lower() # Use normalized name for matching

    # Direct keyword matching (similar to get_instrument_type_from_track_name)
    for family, keywords in INSTRUMENT_KEYWORDS.items():
        if any(keyword in name_lower for keyword in keywords):
            # Map the found keyword family to our target families
            if family == "strings": return "strings"
            if family == "brass": return "brass"
            # Woodwinds need mapping to reed/pipe? Or a general woodwind?
            # Let's map to reed for now, pipe is less common track name keyword target
            if family == "woodwinds": return "reed"
            if family == "percussion": return "percussion" # Keep distinct from drums for now
            if family == "piano": return "piano"
            if family == "choir": return "choir"
            if family == "synth": return "synth"
            if family == "guitar": return "guitar"
            # How to handle "bass"? Bass keywords could be guitar range or actual bass
            # Let's assume bass keywords mean bass guitar/instrument for now
            if family == "bass": return "bass"
            # Explicitly handle drums
            if "drum" in name_lower: return "drums"

    # Check specific program range families if keyword families didn't map
    # e.g. if keyword dict didn't have bass, organ etc.
    # This is less reliable from name alone, but a fallback
    if "organ" in name_lower: return "organ"
    if "ensemble" in name_lower: return "ensemble"
    if any(eth in name_lower for eth in ["ethnic", "sitar", "koto", "shakuhachi"]): return "ethnic"
    if any(fx in name_lower for fx in ["effect", "fx", "sound"]): return "sound_effects"

    # Final fallback
    return "other"

# --- Chord Analysis Constants and Functions (from music_theory.py) ---

# Updated list of qualities expected by the tokenizer
TOKENIZER_EXPECTED_QUALITIES = {'maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'sus4', 'sus2', 'other'}

def standardize_chord_quality(chord: music21.chord.Chord) -> str:
    """
    Standardizes music21 chord quality to a common format matching tokenizer vocabulary.
    Tries to identify common triads, sevenths, and suspended chords.
    Includes error handling for non-standard chord objects.
    """
    if not MUSIC21_AVAILABLE or not isinstance(chord, music21.chord.Chord):
        return "other"

    quality = "other" # Default
    try:
        # Use music21's quality detection where possible
        m21_quality = chord.quality
        # Map music21 qualities to our standard set
        if m21_quality == 'major': return "maj"
        if m21_quality == 'minor': return "min"
        if m21_quality == 'diminished': return "dim"
        if m21_quality == 'augmented': return "aug"
        if m21_quality == 'dominant-seventh': return "dom7"
        if m21_quality == 'major-seventh': return "maj7"
        if m21_quality == 'minor-seventh': return "min7"
        # Less common but possible music21 qualities
        if m21_quality == 'half-diminished': # Often maps to min7b5
             return "other" # Or add m7b5 to TOKENIZER_EXPECTED_QUALITIES
        if m21_quality == 'diminished-seventh':
             return "dim" # Often simplified, or add dim7 to expected

        # If quality is unknown or 'other', check intervals manually for sus etc.
        if m21_quality in ['other', 'unknown']:
            if chord.isSuspendedTriad(): # Check for sus4/sus2 specifically
                # Check intervals to differentiate sus2/sus4
                interval_classes = sorted([i.midi % 12 for i in chord.notes])
                root_pc = chord.root().pitchClass
                relative_intervals = sorted([(pc - root_pc + 12) % 12 for pc in interval_classes])
                if relative_intervals == [0, 5, 7]: return "sus4"
                if relative_intervals == [0, 2, 7]: return "sus2"

        # Fallback based on common names if quality mapping failed
        try:
            common_name = chord.commonName.lower()
            # Check simple names first
            if 'maj' in common_name and 'min' not in common_name: return "maj"
            if 'min' in common_name and 'maj' not in common_name: return "min"
            if 'dim' in common_name: return "dim"
            if 'aug' in common_name: return "aug"
            # Check seventh chords (be careful with order)
            if 'dominant-seventh' in common_name: return "dom7"
            if 'major-seventh' in common_name: return "maj7"
            if 'minor-seventh' in common_name: return "min7"
            if 'sus' in common_name:
                 # Need interval check again if just 'sus' found
                 interval_classes = sorted([i.midi % 12 for i in chord.notes])
                 root_pc = chord.root().pitchClass
                 relative_intervals = sorted([(pc - root_pc + 12) % 12 for pc in interval_classes])
                 if relative_intervals == [0, 5, 7]: return "sus4"
                 if relative_intervals == [0, 2, 7]: return "sus2"

        except AttributeError: pass # commonName might fail

        # If nothing matched, return the default 'other'
        return quality

    except Exception as e:
        logger.warning(f"Could not standardize chord quality for {chord} due to unexpected error: {e}")
        return "other"


def extract_chord_events(score: symusic.Score) -> List[Dict[str, Any]]:
    """
    Extract chord events (root_pc, quality, time) from a symusic.Score.
    Uses music21 if available (via temporary file), otherwise falls back to basic mido analysis.

    Args:
        score: The symusic.Score object to analyze.

    Returns:
        A list of dictionaries, each representing a chord event:
        {'root_pc': int (0-11), 'quality': str, 'time': int (ticks)}
    """
    print("\n=== Starting extract_chord_events ===")
    print(f"Score time division: {score.ticks_per_quarter}")
    print(f"Score tracks: {len(score.tracks)}")
    print(f"Music21 available: {MUSIC21_AVAILABLE}")
    
    chord_events = []
    ticks_per_quarter = score.ticks_per_quarter

    # --- Method 1: music21 (preferred) ---
    if MUSIC21_AVAILABLE:
        # music21 needs a file path, so dump the symusic score to a temporary MIDI file
        temp_midi_file = None
        try:
            print("\nTrying music21 analysis...")
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
                temp_midi_file = tmp_midi.name
                print(f"Created temp file: {temp_midi_file}")
                score.dump_midi(temp_midi_file) # Use symusic's dump method

            # Parse the temporary file with music21
            print("Parsing with music21...")
            m21_score = music21.converter.parse(temp_midi_file)
            # Use chordify to find chords
            print("Chordifying score...")
            chordified_score = m21_score.chordify()

            print("\nExtracting chords...")
            chord_count = 0
            quality_counts = defaultdict(int)
            root_counts = defaultdict(int)
            
            for element in chordified_score.recurse().getElementsByClass('Chord'):
                if not element.isRest: # Ignore rests that might be chordified
                    try:
                        # Standardize quality FIRST
                        quality = standardize_chord_quality(element)
                        quality_counts[quality] += 1
                        
                        # Ensure quality is one of the expected ones
                        if quality not in TOKENIZER_EXPECTED_QUALITIES: quality = 'other'

                        root = element.root()
                        if root is None: 
                            print(f"No root found for chord: {element}")
                            continue # Skip if no root found

                        root_pc = root.pitchClass # 0-11
                        root_counts[root_pc] += 1
                        # Convert music21 offset (quarter notes) to ticks
                        time_ticks = int(round(element.offset * ticks_per_quarter))

                        # Avoid duplicate chords at the exact same time
                        is_duplicate = False
                        if chord_events and chord_events[-1]['time'] == time_ticks:
                             is_duplicate = True

                        if not is_duplicate:
                             chord_events.append({
                                 "root_pc": root_pc,
                                 "quality": quality,
                                 "time": time_ticks # Use 'time' key consistent with MidiTok events
                             })
                             chord_count += 1
                             if chord_count % 10 == 0:
                                 print(f"Processed {chord_count} chords...")
                    except Exception as inner_e:
                        print(f"Failed extracting/processing music21 chord element {element}: {inner_e}")
                        continue # Skip this chord

            print("\nChord analysis summary:")
            print(f"Total chords found: {chord_count}")
            print(f"Quality distribution: {dict(quality_counts)}")
            print(f"Root distribution: {dict(root_counts)}")
            
            # Sort by time as chordify might not preserve perfect order
            chord_events.sort(key=lambda x: x['time'])
            # Clean up temp file
            if temp_midi_file: os.unlink(temp_midi_file)
            return chord_events

        except Exception as e:
            print(f"Music21 chord analysis failed: {e}")
            print("Stack trace:")
            traceback.print_exc()
            # Clean up temp file in case of error
            if temp_midi_file and os.path.exists(temp_midi_file): os.unlink(temp_midi_file)
            # Fall through to mido fallback

    # --- Method 2: mido Fallback (if music21 failed or unavailable) ---
    print("\nUsing mido fallback for chord extraction (no root info, less reliable).")
    temp_midi_file_mido = None
    try:
        # Need to dump to file for mido as well
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            temp_midi_file_mido = tmp_midi.name
            print(f"Created temp file for mido: {temp_midi_file_mido}")
            score.dump_midi(temp_midi_file_mido)

        midi = mido.MidiFile(temp_midi_file_mido)
        current_time_ticks = 0
        # {track_idx: {channel: {note: time_on}}} - Track separation is important
        active_notes_by_track_channel = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Heuristic parameters
        SIMULTANEITY_THRESHOLD_TICKS = int(ticks_per_quarter / 16) # Shorter threshold (e.g., 64th note)
        MIN_CHORD_NOTES = 3

        potential_chords = [] # Store groups of potentially simultaneous notes with their time

        print("\nAnalyzing MIDI tracks...")
        for track_idx, track in enumerate(midi.tracks):
             print(f"\nProcessing track {track_idx}")
             current_time_ticks = 0 # Reset time for each track
             notes_in_track = 0
             for msg in track:
                 current_time_ticks += msg.time
                 channel = getattr(msg, 'channel', 0) # Default channel 0

                 if msg.type == 'note_on' and msg.velocity > 0:
                     active_notes_by_track_channel[track_idx][channel][msg.note] = current_time_ticks
                     notes_in_track += 1
                 elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                     if msg.note in active_notes_by_track_channel[track_idx][channel]:
                         # Mark note off time? Or just remove? Removing is simpler.
                         del active_notes_by_track_channel[track_idx][channel][msg.note]

                 # Snapshot active notes *within the current track* at this time step
                 # Collect notes that started recently across channels within this track
                 current_simultaneous_notes_in_track = []
                 start_time_for_chord = float('inf')
                 active_notes_details = [] # Store (note, start_time)

                 for ch, notes_dict in active_notes_by_track_channel[track_idx].items():
                     for note, time_on in notes_dict.items():
                         active_notes_details.append((note, time_on))

                 # Sort by start time to find the earliest start time of potentially simultaneous notes
                 active_notes_details.sort(key=lambda x: x[1])

                 # Check for notes starting within the threshold of the *earliest* note in the potential chord
                 if active_notes_details:
                      potential_chord_start_time = active_notes_details[0][1]
                      for note, time_on in active_notes_details:
                           if time_on - potential_chord_start_time <= SIMULTANEITY_THRESHOLD_TICKS:
                                current_simultaneous_notes_in_track.append(note)
                                start_time_for_chord = min(start_time_for_chord, time_on)

                 if len(current_simultaneous_notes_in_track) >= MIN_CHORD_NOTES:
                     # Try to detect quality (root cannot be determined reliably here)
                     notes_for_quality = sorted(list(set(current_simultaneous_notes_in_track)))
                     quality = detect_chord_quality_basic(notes_for_quality) # Use simplified quality detector

                     if quality and quality != 'other': # Only add if a basic quality is detected
                          # Ensure quality is one of the expected ones
                          if quality not in TOKENIZER_EXPECTED_QUALITIES: quality = 'other'

                          # Add potential chord, avoid duplicates at nearly the same time
                          is_duplicate = False
                          # Check against the last few chords added
                          for existing_chord in reversed(potential_chords[-5:]): # Check last 5
                              # Check if same quality and very close time
                              if existing_chord['quality'] == quality and \
                                 abs(existing_chord['time'] - start_time_for_chord) <= SIMULTANEITY_THRESHOLD_TICKS:
                                  is_duplicate = True
                                  break
                              # Optimization: If we've gone back further in time, stop checking
                              if existing_chord['time'] < start_time_for_chord - SIMULTANEITY_THRESHOLD_TICKS:
                                   break

                          if not is_duplicate:
                              potential_chords.append({
                                  "root_pc": -1, # Indicate root is unknown
                                  "quality": quality,
                                  "time": start_time_for_chord
                              })
             
             print(f"Found {notes_in_track} notes in track {track_idx}")

        # Sort all found potential chords by time and return
        potential_chords.sort(key=lambda x: x['time'])
        print(f"\nFound {len(potential_chords)} potential chords using mido fallback")
        # Clean up temp file
        if temp_midi_file_mido: os.unlink(temp_midi_file_mido)
        return potential_chords

    except Exception as e:
        print(f"Mido fallback chord analysis failed: {e}")
        print("Stack trace:")
        traceback.print_exc()
        # Clean up temp file in case of error
        if temp_midi_file_mido and os.path.exists(temp_midi_file_mido): os.unlink(temp_midi_file_mido)
        return [] # Return empty list on failure

def detect_chord_quality_basic(notes: List[int]) -> str:
    """
    Simplified quality detection for mido fallback (no root context).
    Tries to match basic triads/7ths based on pitch classes present. Less reliable.
    Returns one of TOKENIZER_EXPECTED_QUALITIES or 'other'.
    """
    if len(notes) < 3: return "other"

    pitch_classes = sorted(list(set(note % 12 for note in notes)))
    num_pcs = len(pitch_classes)

    # Check all pitch classes present as potential roots
    for i in range(num_pcs):
        root_pc = pitch_classes[i]
        # Calculate intervals relative to this potential root
        intervals = sorted([(pc - root_pc + 12) % 12 for pc in pitch_classes])

        # Check against known interval patterns
        # Prioritize more complex chords first if intervals match subset
        if intervals == [0, 4, 7, 11]: return "maj7"
        if intervals == [0, 3, 7, 10]: return "min7"
        if intervals == [0, 4, 7, 10]: return "dom7"
        # if intervals == [0, 3, 6, 9]: return "dim7" # Map to dim?
        if intervals == [0, 3, 6, 10]: return "other" # m7b5 -> other
        # if intervals == [0, 3, 7, 11]: return "minmaj7" # -> other

        if intervals == [0, 4, 7]: return "maj"
        if intervals == [0, 3, 7]: return "min"
        if intervals == [0, 3, 6]: return "dim"
        if intervals == [0, 4, 8]: return "aug"

        # Sus chords check (need root context, but check patterns)
        if intervals == [0, 5, 7]: return "sus4"
        if intervals == [0, 2, 7]: return "sus2"

    # Check common inversions or subsets (more complex logic needed here for reliability)

    return "other" # Could not identify a common chord relative to any note as root
