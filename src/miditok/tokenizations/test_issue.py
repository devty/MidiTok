import logging
logging.basicConfig(level=logging.DEBUG) # Enable debug logging for miditok/symusic

from orchestratelyllm.lightning_datamodule import load_tokenizer # Assuming this utility exists
# Or directly: from miditok.tokenizations import MuMIDI
# from miditok.classes import TokenizerConfig
import symusic

tokenizer_path = "./final_tokenizer_for_checkpoint/" # Or the direct .json file
midi_file_path = "midi_cache_manual/reconstructed_hf_midis/gigamidi/gigamidi_972561.mid" # Problematic file

try:
    print(f"Loading tokenizer from: {tokenizer_path}")
    # Option 1: Using your loading utility
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Option 2: If load_tokenizer doesn't exist or for more direct control
    # tokenizer_config = TokenizerConfig.from_json(tokenizer_path / "tokenizer.json") # If it's a dir
    # tokenizer = MuMIDI(tokenizer_config=tokenizer_config)
    # Or tokenizer = MuMIDI(params=tokenizer_path / "tokenizer.json")

    print(f"Successfully loaded tokenizer: {type(tokenizer)}")
    print(f"Loading MIDI file: {midi_file_path}")
    score = symusic.Score(midi_file_path)
    print(f"Successfully loaded score. TPQ: {score.ticks_per_quarter}, Time Signatures: {score.time_signatures}")

    print("Attempting to tokenize...")
    # This is the critical call where the error likely occurs
    tok_sequence = tokenizer(score) # or tokenizer.encode(score)

    print("Tokenization successful (at least no crash).")
    if hasattr(tok_sequence, 'ids'):
        print(f"Token IDs: {tok_sequence.ids}")
    elif isinstance(tok_sequence, list):
         print(f"Token IDs (list): {tok_sequence}")


except Exception as e:
    print(f"Error during isolated tokenization of {midi_file_path}:")
    import traceback
    traceback.print_exc()
