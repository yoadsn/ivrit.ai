import os

import numpy as np

from vad.definitions import VAD_SPEECH_PROBS_FILENAME, VAD_SPEECH_PROBS_NP_CACHE_FILENAME

# Sessions whose mean speech probability is below this value are considered
# to contain no meaningful speech (blank / silent audio).
EMPTY_AUDIO_SPEECH_PROB_THRESHOLD = 0.2


def get_frame_vad_probs_filename(root_dir: str, source: str, episode: str) -> str:
    return os.path.join(root_dir, source, episode, VAD_SPEECH_PROBS_FILENAME)


def load_frame_vad_probs(filename: str) -> np.ndarray:
    np_cache_file_name = filename.replace(VAD_SPEECH_PROBS_FILENAME, VAD_SPEECH_PROBS_NP_CACHE_FILENAME)
    if os.path.exists(np_cache_file_name):
        return np.load(np_cache_file_name)

    with open(filename, "r") as f:
        speech_probs_per_frame = np.array([float(v) for v in f.readlines() if v])
        np.save(np_cache_file_name, speech_probs_per_frame)
        return speech_probs_per_frame


def is_empty_audio(vad_probs_filename: str) -> bool:
    """Return True when the VAD output indicates the audio is effectively silent.

    A session is considered empty when the mean speech probability across all
    frames is below ``EMPTY_AUDIO_SPEECH_PROB_THRESHOLD``.
    """
    probs = load_frame_vad_probs(vad_probs_filename)
    if len(probs) == 0:
        return False
    return float(np.mean(probs)) < EMPTY_AUDIO_SPEECH_PROB_THRESHOLD
