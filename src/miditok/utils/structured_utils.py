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
logger.setLevel(logging.WARNING)  # Ensure this module's logger is at DEBUG level
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
TOKENIZER_EXPECTED_QUALITIES = {
    # Basic Triads & 7ths
    'maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7',
    # Suspended
    'sus4', 'sus2',
    # Extended & Altered
    'm7b5', 'dim7', 'maj6', 'min6', 'dom9', 'maj9', 'min9',
    'dom11', 'maj11', 'min11', 'dom13', 'maj13', 'min13',
    'power', 'aug7', 'augMaj7',
    # Fallback
    'other'
}

def standardize_chord_quality(chord: music21.chord.Chord) -> str:
    """
    Standardizes music21 chord quality to a common format matching TOKENIZER_EXPECTED_QUALITIES.
    Tries common triads, sevenths, sixths, ninths, 11ths, 13ths, suspended, augmented, power, etc.
    Includes error handling for non-standard chord objects.
    """
    if not MUSIC21_AVAILABLE or not isinstance(chord, music21.chord.Chord):
        return "other"

    try:
        # --- Direct Quality Mapping ---
        m21_quality = chord.quality
        if m21_quality == 'major': return "maj"
        if m21_quality == 'minor': return "min"
        if m21_quality == 'diminished': return "dim"
        if m21_quality == 'augmented': return "aug"
        if m21_quality == 'dominant-seventh': return "dom7"
        if m21_quality == 'major-seventh': return "maj7"
        if m21_quality == 'minor-seventh': return "min7"
        if m21_quality == 'half-diminished': return "m7b5"
        if m21_quality == 'diminished-seventh': return "dim7"
        if m21_quality == 'major-sixth': return "maj6"
        if m21_quality == 'minor-sixth': return "min6"
        if m21_quality == 'dominant-ninth': return "dom9"
        if m21_quality == 'major-ninth': return "maj9"
        if m21_quality == 'minor-ninth': return "min9"
        # Music21 quality doesn't typically distinguish 11ths/13ths directly? Rely on names/intervals.
        if m21_quality == 'augmented-seventh': return "aug7"
        if m21_quality == 'augmented-major-seventh': return "augMaj7"

        # --- Interval/Name-Based Mapping (if direct quality failed or insufficient) ---
        # Check suspended explicitly
        try:
            if chord.isSuspendedTriad():
                # Check intervals to differentiate sus2/sus4 (relative to root)
                root_pc = chord.root().pitchClass
                relative_intervals = sorted([(note.pitch.pitchClass - root_pc + 12) % 12 for note in chord.notes])
                if 5 in relative_intervals and 7 in relative_intervals: return "sus4" # Check for P4, P5
                if 2 in relative_intervals and 7 in relative_intervals: return "sus2" # Check for M2, P5
        except AttributeError:
             logger.debug(f"AttributeError checking isSuspendedTriad for {chord}. Skipping suspended check.")
             pass # Continue if the attribute doesn't exist

        # Check power chord (root and fifth only)
        #if chord.isPowerChord(): return "power"

        # Fallback to common names for more complex chords (11ths, 13ths)
        try:
            common_name = chord.commonName.lower()
            # Use specific common names if available
            if 'dominant-11th' in common_name: return "dom11"
            if 'major-11th' in common_name: return "maj11"
            if 'minor-11th' in common_name: return "min11"
            if 'dominant-13th' in common_name: return "dom13"
            if 'major-13th' in common_name: return "maj13"
            if 'minor-13th' in common_name: return "min13"
            # Re-check simpler ones if direct quality missed them somehow
            if 'major-sixth' in common_name: return "maj6"
            if 'minor-sixth' in common_name: return "min6"
            if 'augmented-seventh' in common_name: return "aug7"
            if 'augmented-major-seventh' in common_name: return "augMaj7"
            # Handle potential variations like 'half-diminished seventh' -> m7b5
            if 'half-diminished' in common_name: return "m7b5"

        except AttributeError: pass # commonName might fail

        # --- Final Interval Checks (Less reliable, use as last resort) ---
        # Example: check for augmented triad if quality='other' but intervals match
        if m21_quality == 'other':
            root_pc = chord.root().pitchClass
            relative_intervals = sorted([(note.pitch.pitchClass - root_pc + 12) % 12 for note in chord.notes])
            # Check interval content for basic types missed earlier
            has_major_third = 4 in relative_intervals
            has_minor_third = 3 in relative_intervals
            has_perfect_fifth = 7 in relative_intervals
            has_dim_fifth = 6 in relative_intervals
            has_aug_fifth = 8 in relative_intervals

            if has_major_third and has_aug_fifth: return "aug"
            if has_minor_third and has_dim_fifth: return "dim"
            # Could add more checks here but they get complex and overlap with above checks

        # If nothing matched, return the default 'other'
        logger.debug(f"Chord {chord} with quality '{m21_quality}' and common name '{getattr(chord, 'commonName', 'N/A')}' mapped to 'other'")
        return "other"

    except Exception as e:
        logger.warning(f"Could not standardize chord quality for {chord} due to unexpected error: {e}")
        return "other"

def get_neo_riemannian_transformation(chord1: music21.chord.Chord, chord2: music21.chord.Chord) -> Optional[str]:
    """
    Calculates the Neo-Riemannian transformation (P, L, R) between two major/minor triads.

    Args:
        chord1: The first music21 Chord object (must be major or minor triad).
        chord2: The second music21 Chord object (must be major or minor triad).

    Returns:
        The transformation type ("P", "L", "R") or None if not applicable or chords are not major/minor triads.
    """
    if not MUSIC21_AVAILABLE or chord1 is None or chord2 is None:
        return None

    # Check if both are major or minor triads
    quality1 = chord1.quality
    quality2 = chord2.quality
    is_triad1 = quality1 in ["major", "minor"] and len(chord1.pitches) == 3
    is_triad2 = quality2 in ["major", "minor"] and len(chord2.pitches) == 3

    if not (is_triad1 and is_triad2):
        return None

    root1_pc = chord1.root().pitchClass
    root2_pc = chord2.root().pitchClass
    interval_semitones = abs(root1_pc - root2_pc)
    # Handle wrap-around distance
    if interval_semitones > 6: interval_semitones = 12 - interval_semitones

    # P (Parallel): Same root, quality change
    if root1_pc == root2_pc and quality1 != quality2:
        return "P"

    # L (Leading-Tone): Root moves by semitone (1), quality change
    if interval_semitones == 1 and quality1 != quality2:
        return "L"

    # R (Relative): Root moves by minor third (3) for maj->min, or major third (4) for min->maj. Quality must match.
    if quality1 == "major" and quality2 == "minor" and interval_semitones == 3: # Maj -> Min (e.g. Cmaj -> Amin)
        # Check relative relationship (root distance)
        if (root1_pc - root2_pc + 12) % 12 == 3:
             return "R"
    if quality1 == "minor" and quality2 == "major" and interval_semitones == 4: # Min -> Maj (e.g. Amin -> Cmaj)
        # Check relative relationship (root distance)
        if (root2_pc - root1_pc + 12) % 12 == 4:
             return "R"

    # Other transformations (S, N, H) are more complex and not implemented here.
    return None

def extract_chord_events(
    score: symusic.Score,
    analyze_roman_numerals: bool = True,  # New parameter
    analyze_neo_riemannian: bool = True   # New parameter
) -> List[Dict[str, Any]]:
    """
    Extract chord events (root_pc, quality, time, roman_numeral, neo_riemannian)
    from a symusic.Score. Requires music21 for analysis.
    Saves score to a temporary MIDI file for processing.

    Args:
        score: The symusic.Score object to analyze.
        analyze_roman_numerals: Whether to analyze Roman numerals.
        analyze_neo_riemannian: Whether to analyze Neo-Riemannian transformations.

    Returns:
        A list of dictionaries, each representing a chord event:
        {'root_pc': int (0-11), 'quality': str, 'time': int (ticks),
         'roman_numeral': Optional[str], 'neo_riemannian': Optional[str]}
        Returns empty list if music21 is unavailable or analysis fails.
    """
    # <<< Remove Debug Prints >>>
    #print(f"\n=== ENTERING extract_chord_events ===")
    #print(f"  analyze_roman_numerals = {analyze_roman_numerals}")
    #print(f"  analyze_neo_riemannian = {analyze_neo_riemannian}")
    # <<< End Remove >>>

    #print("\n=== Starting extract_chord_events (music21 required) ===")
    #print(f"Score time division: {score.ticks_per_quarter}")
    #print(f"Score tracks: {len(score.tracks)}")
    #print(f"Music21 available: {MUSIC21_AVAILABLE}")

    if not MUSIC21_AVAILABLE:
        logger.error("music21 is required for chord analysis but not found. Skipping chord extraction.")
        return []

    chord_events = []
    ticks_per_quarter = score.ticks_per_quarter
    temp_midi_file = None

    try:
        #print("\nRunning music21 analysis...")
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            temp_midi_file = tmp_midi.name
            #print(f"Created temp file: {temp_midi_file}")
            score.dump_midi(temp_midi_file)

        #print("Parsing with music21...")
        m21_score = music21.converter.parse(temp_midi_file)
        #print("Chordifying score...")
        chordified_score = m21_score.chordify()

        key = None
        if analyze_roman_numerals:
            #print("\nAnalyzing key (for Roman Numerals)...")
            try:
                key = m21_score.analyze('key')
                #print(f"Key analysis result: {key}")
            except Exception as key_err:
                print(f"Key analysis failed: {key_err}")
                key = None
        #else:
             #print("\nSkipping key analysis (Roman Numerals disabled).")

        #print("\nExtracting chords...")
        chord_count = 0
        rn_count = 0
        nr_count = 0
        quality_counts = defaultdict(int)
        root_counts = defaultdict(int)
        previous_chord = None

        all_chord_elements = list(chordified_score.recurse().getElementsByClass('Chord'))
        #print(f"Found {len(all_chord_elements)} potential chord elements.")

        for element in all_chord_elements:
            if not element.isRest:
                current_chord_data = {}
                try:
                    quality = standardize_chord_quality(element)
                    quality_counts[quality] += 1

                    if quality not in TOKENIZER_EXPECTED_QUALITIES:
                        logger.warning(f"Standardized quality '{quality}' not in expected set for {element}. Using 'other'.")
                        quality = 'other'

                    root = element.root()
                    if root is None:
                        logger.debug(f"Skipping chord element (no root): {element}")
                        previous_chord = None
                        continue

                    root_pc = root.pitchClass
                    root_counts[root_pc] += 1
                    time_ticks = int(round(element.offset * ticks_per_quarter))

                    current_chord_data = {
                        "root_pc": root_pc,
                        "quality": quality,
                        "time": time_ticks,
                        "roman_numeral": None,
                        "neo_riemannian": None
                    }

                    if analyze_roman_numerals and key:
                        try:
                            #print(f"  Attempting Roman Numeral analysis for: {element} in key {key}")
                            rn_obj = music21.roman.romanNumeralFromChord(element, key, ignoreDiminishedMinorQuality=True)
                            rn_figure = rn_obj.figure
                            rn_figure_cleaned = rn_figure.replace(' ', '')
                            current_chord_data['roman_numeral'] = rn_figure_cleaned
                            rn_count += 1
                        except music21.roman.RomanNumeralException as rn_err:
                            logger.debug(f"Could not determine Roman Numeral for {element} in key {key}: {rn_err}")
                        except Exception as rn_err_general:
                             logger.warning(f"Unexpected error during Roman Numeral analysis for {element}: {rn_err_general}")

                    if analyze_neo_riemannian and previous_chord:
                         try:
                            #print(f"  Attempting Neo-Riemannian analysis between: {previous_chord} and {element}")
                            nr_label = get_neo_riemannian_transformation(previous_chord, element)
                            if nr_label:
                                current_chord_data['neo_riemannian'] = nr_label
                                nr_count += 1
                         except Exception as nr_err:
                              logger.warning(f"Error during Neo-Riemannian analysis between {previous_chord} and {element}: {nr_err}")

                    is_duplicate = False
                    if chord_events and chord_events[-1]['time'] == time_ticks:
                         if (chord_events[-1]['root_pc'] == root_pc and
                             chord_events[-1]['quality'] == quality and
                             chord_events[-1]['roman_numeral'] == current_chord_data['roman_numeral'] and
                             chord_events[-1]['neo_riemannian'] == current_chord_data['neo_riemannian']):
                            is_duplicate = True

                    if not is_duplicate:
                        chord_events.append(current_chord_data)
                        chord_count += 1
                        #if chord_count % 50 == 0: # Log less frequently
                        #    print(f"Processed {chord_count} chords ({rn_count} RNs, {nr_count} NRs)...")
                        previous_chord = element
                    else:
                         previous_chord = element

                except Exception as inner_e:
                    print(f"Failed extracting/processing music21 chord element {element}: {inner_e}")
                    traceback.print_exc(limit=1)
                    previous_chord = None
                    continue
            else:
                 previous_chord = None

        #print("\nChord analysis summary:")
        #print(f"Total unique chord events generated: {chord_count}")
        #print(f"Roman numerals identified: {rn_count}")
        #print(f"Neo-Riemannian transformations identified: {nr_count}")
        #print(f"Quality distribution: {dict(quality_counts)}")
        #print(f"Root distribution: {dict(root_counts)}")

        chord_events.sort(key=lambda x: x['time'])
        if temp_midi_file: os.unlink(temp_midi_file)
        #print("\n=== Finished extract_chord_events ===")
        return chord_events

    except Exception as e:
        print(f"Music21 chord analysis failed: {e}")
        print("Stack trace:")
        traceback.print_exc()
        if temp_midi_file and os.path.exists(temp_midi_file): os.unlink(temp_midi_file)
        return []
