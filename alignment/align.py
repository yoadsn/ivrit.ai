import logging
from pathlib import Path
from typing import Union

import stable_whisper
from faster_whisper import WhisperModel
from stable_whisper.whisper_compatibility import SAMPLE_RATE
from tqdm import tqdm

logger = logging.getLogger(__name__)

from alignment.seekable_audio_loader import SeekableAudioLoader
from alignment.utils import (
    create_transcript_from_segments,
    find_probable_segment_before_time,
    get_breakable_align_model,
    get_confusion_zone,
    get_text_from_segments,
)
from utils.vtt import vtt_to_whisper_result


def _sanitize_align_result(result: stable_whisper.WhisperResult, text_fed: str) -> None:
    """Clean up a model.align() result so its concatenated word text matches text_fed.

    model.align() from stable_whisper has a known bug: it can duplicate text in
    zero-duration words (words where start == end) that appear when alignment
    partially fails.  The duplication can occur:

    - At the boundary between properly-aligned and zero-duration words.
    - Deep inside a run of zero-duration words.
    - Across the entire result when all words are zero-duration.

    This function also strips the spurious leading space that the Whisper tokenizer
    prepends to the first token of every sequence.

    Operates in-place on *result*.

    Strategy (ordered from cheapest to most expensive):

    1. **Leading space** — if text_fed does not start with a space but the result
       does, strip the leading space from the first word.  O(1).

    2. **Quick exit** — if lengths already match after step 1, return.  O(n) for
       one concatenation pass.

    3. **Boundary overlap** — look for a suffix/prefix overlap at the last
       non-zero/zero-duration word boundary.  Handles the most common duplication
       pattern.  O(extra) string comparisons.

    4. **Exhaustive scan** — sliding window over all zero-duration word runs
       looking for a contiguous group of exactly ``extra`` chars whose removal
       yields text_fed.  O(zero_dur_words * avg_dup_span) with early exit.
       Sub-millisecond even for 2000+ zero-duration words.
    """
    if not result.segments:
        return

    # ------------------------------------------------------------------
    # Step 1: Strip spurious leading space
    # ------------------------------------------------------------------
    if not text_fed.startswith(' '):
        first_seg = result.segments[0]
        if first_seg.words:
            if first_seg.words[0].word.startswith(' '):
                first_seg.words[0].word = first_seg.words[0].word[1:]
        elif first_seg._default_text.startswith(' '):
            first_seg._default_text = first_seg._default_text[1:]

    # ------------------------------------------------------------------
    # Step 2: Quick length check
    # ------------------------------------------------------------------
    all_words = result.all_words()
    if not all_words:
        return

    result_text = ''.join(w.word for w in all_words)
    extra = len(result_text) - len(text_fed)

    if extra == 0:
        return  # Nothing to fix
    if extra < 0:
        # Result is shorter than expected — not a duplication issue we can fix.
        logger.warning(
            f"_sanitize_align_result: result is {-extra} chars shorter than text_fed "
            f"({len(result_text)} vs {len(text_fed)}); cannot fix."
        )
        return

    # ------------------------------------------------------------------
    # Step 3: Boundary overlap (fast path for the most common pattern)
    # ------------------------------------------------------------------
    # Find boundary: last word with non-zero duration.
    boundary_idx = None
    for i in range(len(all_words) - 1, -1, -1):
        if all_words[i].end > all_words[i].start:
            boundary_idx = i
            break

    if boundary_idx is not None and boundary_idx < len(all_words) - 1:
        text_before = ''.join(w.word for w in all_words[: boundary_idx + 1])
        text_after = ''.join(w.word for w in all_words[boundary_idx + 1 :])

        # Check suffix of text_before == prefix of text_after
        overlap_len = 0
        max_check = min(len(text_before), len(text_after), extra + 5)
        for length in range(1, max_check + 1):
            if text_before.endswith(text_after[:length]):
                overlap_len = length

        if overlap_len == extra:
            if _remove_zero_dur_words_at(result, all_words, boundary_idx + 1, overlap_len):
                logger.warning(
                    f"_sanitize_align_result: removed {overlap_len} duplicate chars "
                    f"at boundary (word {boundary_idx})."
                )
                return

    # ------------------------------------------------------------------
    # Step 4: Exhaustive scan over zero-duration word runs
    # ------------------------------------------------------------------
    # Build word char offsets
    word_starts = []
    pos = 0
    for w in all_words:
        word_starts.append(pos)
        pos += len(w.word)

    # Collect contiguous runs of zero-duration words
    zero_dur_indices = [i for i, w in enumerate(all_words) if w.start == w.end]
    if not zero_dur_indices:
        logger.warning(
            f"_sanitize_align_result: {extra} extra chars but no zero-duration words "
            f"to remove; cannot fix."
        )
        return

    runs = []
    run_start = zero_dur_indices[0]
    prev = zero_dur_indices[0]
    for idx in zero_dur_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((run_start, prev))
            run_start = idx
            prev = idx
    runs.append((run_start, prev))

    for rs, re in runs:
        i = rs
        while i <= re:
            chars = 0
            j = i
            while j <= re:
                chars += len(all_words[j].word)
                if chars == extra:
                    candidate = (
                        result_text[:word_starts[i]]
                        + result_text[word_starts[j] + len(all_words[j].word) :]
                    )
                    if candidate == text_fed:
                        n_words = j - i + 1
                        target = set(id(all_words[k]) for k in range(i, j + 1))
                        for seg in result.segments:
                            if seg.words:
                                seg.words = [w for w in seg.words if id(w) not in target]
                        result.segments = [s for s in result.segments if s.words]
                        logger.warning(
                            f"_sanitize_align_result: removed {n_words} duplicate words "
                            f"({extra} chars at word {i}, char pos {word_starts[i]}) "
                            f"via exhaustive scan."
                        )
                        return
                    break  # This sub-run text didn't match; advance start
                elif chars > extra:
                    break
                j += 1
            i += 1

    logger.warning(
        f"_sanitize_align_result: could not fix {extra} extra chars "
        f"({len(all_words)} words, {len(zero_dur_indices)} zero-dur). "
        f"Result text will not match text_fed."
    )


def _remove_zero_dur_words_at(
    result: stable_whisper.WhisperResult,
    all_words: list,
    start_idx: int,
    chars_to_remove: int,
) -> bool:
    """Remove contiguous zero-duration words starting at start_idx totalling chars_to_remove.

    Returns True if exactly chars_to_remove characters were removed, False otherwise.
    """
    chars = 0
    end_idx = start_idx
    while end_idx < len(all_words):
        w = all_words[end_idx]
        if w.start != w.end:
            break  # Hit a real-duration word
        chars += len(w.word)
        if chars == chars_to_remove:
            # Remove words start_idx..end_idx (inclusive)
            target = set(id(all_words[k]) for k in range(start_idx, end_idx + 1))
            for seg in result.segments:
                if seg.words:
                    seg.words = [w for w in seg.words if id(w) not in target]
            result.segments = [s for s in result.segments if s.words]
            return True
        elif chars > chars_to_remove:
            break  # Overshot — can't remove on word boundaries
        end_idx += 1
    return False


def _remove_cross_call_text_overlap(
    committed_pieces: list[stable_whisper.result.Segment],
    new_segments: list[stable_whisper.result.Segment],
) -> list[stable_whisper.result.Segment]:
    """Remove text overlap between committed aligned pieces and new skip-pass segments.

    Safety net for cross-call duplication: if the main alignment pass committed a
    zero-duration tail and the skip pass starts with overlapping text, remove the
    duplicate words from the head of new_segments.

    Returns the (possibly trimmed) new_segments list.
    """
    if not committed_pieces or not new_segments:
        return new_segments

    # Collect zero-duration words from the tail of committed_pieces
    tail_zero_words = []
    for seg in reversed(committed_pieces):
        if not seg.words:
            continue
        for w in reversed(seg.words):
            if w.start == w.end:
                tail_zero_words.append(w)
            else:
                break
        # If we found zero-dur words and this segment started with a real word, stop
        if tail_zero_words and seg.words and seg.words[0].start != seg.words[0].end:
            break
        if not tail_zero_words:
            return new_segments
    tail_zero_words.reverse()

    if not tail_zero_words:
        return new_segments

    tail_text = ''.join(w.word for w in tail_zero_words)

    # Collect zero-duration words from the head of new_segments
    head_zero_words = []
    for seg in new_segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start == w.end:
                head_zero_words.append(w)
            else:
                break
        if head_zero_words and seg.words and seg.words[-1].start != seg.words[-1].end:
            continue
        break

    if not head_zero_words:
        return new_segments

    head_text = ''.join(w.word for w in head_zero_words)

    # Find overlap: suffix of tail_text == prefix of head_text
    overlap_len = 0
    max_check = min(len(tail_text), len(head_text))
    for length in range(1, max_check + 1):
        if head_text[:length] == tail_text[-length:]:
            overlap_len = length

    if overlap_len == 0:
        return new_segments

    # Remove exactly overlap_len chars from the head of new_segments
    chars_removed = 0
    words_removed = 0
    for w in head_zero_words:
        if chars_removed + len(w.word) <= overlap_len:
            chars_removed += len(w.word)
            words_removed += 1
        else:
            break

    if chars_removed != overlap_len:
        return new_segments  # Can't remove on clean word boundaries

    target = set(id(head_zero_words[i]) for i in range(words_removed))
    for seg in new_segments:
        if seg.words:
            seg.words = [w for w in seg.words if id(w) not in target]
    new_segments = [s for s in new_segments if s.words]

    logger.warning(
        f"_remove_cross_call_text_overlap: removed {words_removed} duplicate "
        f"words ({chars_removed} chars) at cross-call boundary."
    )
    return new_segments


def align_transcript_to_audio(
    audio_file: Path,
    transcript: Union[Path, stable_whisper.result.WhisperResult],
    model: Union[str, WhisperModel] = "ivrit-ai/whisper-large-v3-turbo-ct2",
    device: str = "auto",
    align_model_compute_type: str = "int8",
    language: str = "he",
    pre_confusion_zone_backward_skip_search_duration_window: int = 30,
    max_pre_confusion_zone_tries_before_skip: int = 2,
    unaligned_start_text_match_search_radius: int = 270,
    zero_duration_segments_failure_ratio: float = 0.2,
    max_confusion_zone_skip_duration: int = 120,
    min_confusion_zone_skip_duration: int = 15,
    entry_id: str = None,
) -> stable_whisper.WhisperResult:
    """Align a transcript to audio using a robust, confusion-aware alignment algorithm.

    This function aligns a transcript to audio by breaking the alignment into pieces
    when it detects confusion zones. It uses a strategy of trying to align from before
    the confusion zone a few times, and if that fails, it skips the confusion zone
    and continues from after it.

    Note - Transcript is assumed to have relatively close (+-30s) timestamps across
    segments. If the transcripts has 0 timestamps - This algorithm would not work.
    Consider using stable_ts alignment directly.

    Args:
        audio_file: Path to the audio file to align to.
        transcript: Either a Path to a VTT file or a WhisperResult object containing
            the transcript to align.
        model: The model to use for alignment, either a path to a local model or a
            model identifier from the Hugging Face Hub.
        device: The device to use for inference, e.g., "cpu", "cuda", "auto".
            Can include a device index, e.g., "cuda:0".
        align_model_compute_type: The compute type to use for the model, e.g., "int8", "float16".
        language: The language code of the transcript.
        pre_confusion_zone_backward_skip_search_duration_window: Duration in seconds to look back
            when searching for a segment before a confusion zone.
        max_pre_confusion_zone_tries_before_skip: Maximum number of attempts to align from
            before a confusion zone before skipping it.
        unaligned_start_text_match_search_radius: Radius in characters to search for matching
            text when finding the start of a confusion zone in the unaligned transcript.
        zero_duration_segments_failure_ratio: Threshold for the ratio of zero-duration segments
            to total segments, above which alignment is considered failed.
        max_confusion_zone_skip_duration: When a skip over a confusion zone is needed -
            What is the mexium allowed skip in seconds.
        min_confusion_zone_skip_duration: When a skip over a confusion zone is needed -
            What is the min allowed skip in seconds.
        entry_id: helps with logging in parallel processing setting. Optional.

    Returns:
        A WhisperResult containing the aligned transcript.
    """
    if isinstance(transcript, Path):
        unaligned = vtt_to_whisper_result(str(transcript))
    else:
        unaligned = transcript

    # The Whisper tokenizer replaces newlines with spaces during encode/decode,
    # so the aligned output will never contain newlines.  Normalise the unaligned
    # text in-place now so that (a) the confusion-zone text-matching (find() calls
    # below) works correctly, and (b) character counts stay identical (\n and space
    # are both one character).
    #
    # Segment.text is a read-only property:
    #   - when has_words=True  it returns ''.join(word.word for word in words)
    #   - when has_words=False it returns _default_text
    # So we must write through the appropriate backing store.
    for seg in unaligned.segments:
        if seg.words:
            for word in seg.words:
                if '\n' in word.word:
                    word.word = word.word.replace('\n', ' ')
        else:
            if '\n' in seg._default_text:
                seg._default_text = seg._default_text.replace('\n', ' ')

    # If model is a string, load it using get_breakable_align_model
    if isinstance(model, str):
        model = get_breakable_align_model(model, device, align_model_compute_type)

    audio_metadata = stable_whisper.audio.utils.get_metadata(str(audio_file))
    audio_duration = audio_metadata["duration"] or 0

    # Initialize outer alignment loop
    slice_start = 0
    top_matched_unaligned_timestamp = 0
    done = False

    aligned_pieces: list[stable_whisper.Segment] = []
    min_confusion_zone_start = 0
    max_confusion_zone_end = 0
    current_pre_confusion_zone_tries = 0
    to_align_next = unaligned.text  # We align text not segments

    # Create a progress bar for the alignment process
    progress_bar = tqdm(total=audio_duration, unit="sec", desc=f"Aligning {entry_id or 'Entry'}")
    while not done:
        # Get the audio slice
        audio = SeekableAudioLoader(
            str(audio_file),
            sr=SAMPLE_RATE,
            stream=True,
            load_sections=[[slice_start, None]],
            test_first_chunk=False,
            # We expect long stretches of aligned audio and possible IO contention
            buffer_size=300 * SAMPLE_RATE,
        )

        # Align it until it breaks
        aligned: stable_whisper.WhisperResult = model.align(
            audio, to_align_next, language=language, failure_threshold=zero_duration_segments_failure_ratio
        )
        _sanitize_align_result(aligned, to_align_next)

        any_good_alignemnts = aligned.segments[0].start != aligned.segments[-1].end
        # If unable to do any proper alignment - assume a confusion zone up front
        if not any_good_alignemnts:
            confusion_zone_start = slice_start
            confusion_zone_end = confusion_zone_start + min_confusion_zone_skip_duration
        else:
            # find the confusion zone
            confusion_zone_start, confusion_zone_end = get_confusion_zone(aligned)

        # Check if done == No confusion zone exists
        if confusion_zone_start is None:
            # Keep aligned segments and stop aligning
            aligned_pieces.extend(aligned.segments)
            break

        # If the new confusion zone is outside the old one
        # we treat it as a new confusion zone
        if confusion_zone_start > max_confusion_zone_end:
            min_confusion_zone_start = confusion_zone_start
            max_confusion_zone_end = confusion_zone_end
            current_pre_confusion_zone_tries = 0
        else:
            # Expends the confusion zone with all previous tries
            # in this area
            min_confusion_zone_start = min(min_confusion_zone_start, confusion_zone_start)
            max_confusion_zone_end = max(max_confusion_zone_end, confusion_zone_end)

        # Pre-roll detection: if we haven't aligned anything yet and the
        # first unaligned segment starts after the confusion zone, the
        # audio before it is just dead air / pre-roll. No text needs to
        # be skipped; simply advance slice_start past the confusion zone
        # and retry.  We step by confusion-zone increments rather than
        # jumping to the first unaligned segment, since unaligned
        # timestamps are only approximate.
        if not aligned_pieces and unaligned.segments[0].start >= max_confusion_zone_end:
            slice_start = max_confusion_zone_end
            progress_bar.write(
                f"Pre-roll detected: advancing audio to {slice_start:.1f}s "
                f"(first speech estimated at {unaligned.segments[0].start:.1f}s)"
            )
            progress_bar.update(slice_start - progress_bar.n)
            min_confusion_zone_start = 0
            max_confusion_zone_end = 0
            continue

        probable_segment_before_confusion_zone = find_probable_segment_before_time(
            aligned,
            confusion_zone_start,
            pre_confusion_zone_backward_skip_search_duration_window,
        )

        # If there is a probable segment before confusion zone
        if probable_segment_before_confusion_zone is not None:
            # Keep properly aligned segments up to it including
            segments_already_aligned = aligned.segments[: probable_segment_before_confusion_zone.id + 1]
            aligned_pieces.extend(segments_already_aligned)

            # point to audio start for next try
            slice_start = probable_segment_before_confusion_zone.end

            # Update progress bar
            progress_bar.update(slice_start - progress_bar.n)

            # Prepare not aligned text for next try
            to_align_next = get_text_from_segments(aligned.segments[probable_segment_before_confusion_zone.id + 1 :])

            # If we have more tries left for the "pre confusion zone" retry strategy
            if current_pre_confusion_zone_tries < max_pre_confusion_zone_tries_before_skip:
                progress_bar.write(f"Retry alignment from before confusion zone: {slice_start}")
                # another pre confusion zone try is done
                current_pre_confusion_zone_tries += 1
                continue

        # Skipping forward - to_align_next is all the text we tried to align in this attempt
        # of course we will skip some of it after deciding where to skip to

        # Assume confusion starts where the audio starts.
        min_confusion_zone_start = slice_start

        # skip the confusion zone if no pre confusion zone retry
        # is possible or allowed.

        # Don't skip too much or too little forward
        max_confusion_zone_end = max(
            max_confusion_zone_end, min_confusion_zone_start + min_confusion_zone_skip_duration
        )
        max_confusion_zone_end = min(
            min_confusion_zone_start + max_confusion_zone_skip_duration, max_confusion_zone_end
        )

        progress_bar.write(f"Skipping confusion zone: {min_confusion_zone_start} - {max_confusion_zone_end}")
        logger.warning(f"[SKIP-TRACE] === Entering confusion zone skip ===")
        logger.warning(f"[SKIP-TRACE] min_confusion_zone_start={min_confusion_zone_start}, max_confusion_zone_end={max_confusion_zone_end}")
        logger.warning(f"[SKIP-TRACE] aligned_pieces count={len(aligned_pieces)}, total text in pieces={sum(len(s.text) for s in aligned_pieces)}")
        logger.warning(f"[SKIP-TRACE] to_align_next length={len(to_align_next)}, first 80 chars: {to_align_next[:80]!r}")
        # skipping - resets the jump back strategy allowed tries
        current_pre_confusion_zone_tries = 0

        # Discover which segments will be skipped
        # we don't want to lose any text - we want to skip aligning it
        # and keep the unaligned as a reasonable estimate

        # problem is - we have the text of the unaligned but the segmentation
        # does not match the unaligned which we use to pickup the estimated skip point.
        # thus, we will use text matching to find the locations we are skipping over
        # and which segments (or parts of them) are left unaligned.

        # first get some prefix from the text to align next - this is what we could not align
        # after the last try ended.
        # this usually covers the entire confusion zone up to the end
        # or the entire text for the audio file
        text_at_start_of_confusion_zone = to_align_next[: unaligned_start_text_match_search_radius // 3]

        search_tries_left = 3
        search_radius_to_try = unaligned_start_text_match_search_radius
        while search_tries_left > 0:
            search_tries_left -= 1

            # get segments from the unaligned around confusion zone to find the text within
            # we assume timing is off by not too much, so we expand the search area a little
            search_around_time = (
                min_confusion_zone_start
                if probable_segment_before_confusion_zone is None
                else probable_segment_before_confusion_zone.end
            )
            search_in_unaligned_window_time_start = max(
                # never search in unaligned parts already matched against
                # so we reduce the risk of matching to the past
                top_matched_unaligned_timestamp,
                search_around_time - search_radius_to_try,
            )
            search_in_unaligned_window_time_end = search_around_time + search_radius_to_try
            segments_around_confusion_zone = unaligned.get_content_by_time(
                (search_in_unaligned_window_time_start, search_in_unaligned_window_time_end),
                segment_level=True,
            )
            text_around_confusion_zone = get_text_from_segments(segments_around_confusion_zone)

            # where can we find the prefix text ?
            found_at_text_idx = text_around_confusion_zone.find(text_at_start_of_confusion_zone)
            logger.warning(f"[SKIP-TRACE] text search: search_radius={search_radius_to_try}, "
                           f"segments_around count={len(segments_around_confusion_zone)}, "
                           f"text_around len={len(text_around_confusion_zone)}, "
                           f"search_prefix len={len(text_at_start_of_confusion_zone)}, "
                           f"search_prefix={text_at_start_of_confusion_zone[:60]!r}, "
                           f"found_at_text_idx={found_at_text_idx}")

            # prepare for next try or break
            if found_at_text_idx == -1:
                search_radius_to_try *= 1.5
            else:
                break

        # If still cannot find - some data corruption is expected.
        # but we have no better way to recover at this point.
        # assume all content within the confusion zone span from the unaligned
        # is to be skipped (this may not contain all text in actual confusion zone. or may
        # contain text already aligned.
        # This is a hail-mary attempt
        if found_at_text_idx == -1:
            progress_bar.write(
                f"Could not find matching text in {entry_id or 'entry'} confusion zone, slice_start: {slice_start} - hard skip (+corruption) expected"
            )
            logger.warning(f"[SKIP-TRACE] HARD SKIP: could not find text, defaulting found_at_text_idx=0")
            found_at_text_idx = 0

        # Recall the latest known aligned timestamp - upcoming segments cannot
        # timestamp below it
        top_aligned_timestamp = aligned_pieces[-1].end if aligned_pieces else 0

        # in the confusion zone we cannot trust the aligned times - so we work on the unaligned
        # which is expected to have reasonable approximate timing
        # Find the segments we will continue aligning next
        segments_after_assumed_confusion_zone = unaligned.get_content_by_time(
            (max_confusion_zone_end, unaligned.segments[-1].end), segment_level=True
        )
        logger.warning(f"[SKIP-TRACE] segments_after_assumed_confusion_zone: count={len(segments_after_assumed_confusion_zone)}, "
                       f"first_id={segments_after_assumed_confusion_zone[0].id if segments_after_assumed_confusion_zone else None}, "
                       f"first_start={segments_after_assumed_confusion_zone[0].start if segments_after_assumed_confusion_zone else None}, "
                       f"text_len={sum(len(s.text) for s in segments_after_assumed_confusion_zone)}")

        # Ensure the segments we will align next start after the max_confusion_zone_end
        # so the skip will be effective.
        if (
            segments_after_assumed_confusion_zone
            and segments_after_assumed_confusion_zone[0].start < max_confusion_zone_end
        ):
            segments_after_assumed_confusion_zone = segments_after_assumed_confusion_zone[1:]

        # find the segment that contains the start idx
        # and the index of the text within that segment
        curr_matched_segment_idx = 0
        index_within_segment = found_at_text_idx
        logger.warning(f"[SKIP-TRACE] found_at_text_idx={found_at_text_idx}, "
                       f"segments_around count={len(segments_around_confusion_zone) if segments_around_confusion_zone else 0}")

        # if any segments found covering the confusion zone - analyze them to figure
        # out what exactly is to be skipped or already aligned
        if segments_around_confusion_zone:
            text_len_so_far = len(segments_around_confusion_zone[curr_matched_segment_idx].text)
            while text_len_so_far <= found_at_text_idx:
                index_within_segment = found_at_text_idx - text_len_so_far
                curr_matched_segment_idx += 1
                text_len_so_far += len(segments_around_confusion_zone[curr_matched_segment_idx].text)

            # get the initial segment
            initial_unaligned_segment_in_confusion_zone = segments_around_confusion_zone[curr_matched_segment_idx]
            initial_unaligned_segment_id_in_confusion_zone = initial_unaligned_segment_in_confusion_zone.id
            logger.warning(f"[SKIP-TRACE] initial_unaligned_segment_id={initial_unaligned_segment_id_in_confusion_zone}, "
                           f"index_within_segment={index_within_segment}, "
                           f"segment text len={len(initial_unaligned_segment_in_confusion_zone.text)}, "
                           f"segment start={initial_unaligned_segment_in_confusion_zone.start}, "
                           f"segment end={initial_unaligned_segment_in_confusion_zone.end}")

            # and if the text is mid-point within the segment. we need only the matched part.
            if index_within_segment > 0:
                initial_unaligned_segment_in_confusion_zone = stable_whisper.result.Segment(
                    # The start of the sliced segment is:
                    # Where we intended to start - if probable prev segment was found (slice_start)
                    # or the original start which had no probable segment at all (slice_start)
                    # and never less than the highest aligned timestamp since this
                    # will break the monotonicity of the aligned timestamps (top_aligned_timestamp)
                    start=min(
                        max(
                            top_aligned_timestamp,
                            slice_start,
                        ),
                        # But cannot go beyond the end of this segment
                        initial_unaligned_segment_in_confusion_zone.end,
                    ),
                    end=max(top_aligned_timestamp, initial_unaligned_segment_in_confusion_zone.end),
                    text=initial_unaligned_segment_in_confusion_zone.text[index_within_segment:],
                )

        # If no segments after the confusion zone, we are done
        if not segments_after_assumed_confusion_zone:
            done = True
            first_segment_to_continue_aligning = None
        else:
            first_segment_to_continue_aligning = segments_after_assumed_confusion_zone[0]

        # If segments found within/around confusion zone
        # Handle their skipping, and adding them to the "aligned" pieces
        # as if they were aligned.
        if segments_around_confusion_zone:
            # make sure all unaligned are added to results before existing
            if done:
                confusing_segments_to_skip = unaligned.segments[
                    initial_unaligned_segment_id_in_confusion_zone
                    # +1, since we will prepend this first segment (it might required prefix removal)
                    + 1 :
                ]
            else:
                # find the segments within the confusion zone that we will skip
                confusing_segments_to_skip = unaligned.segments[
                    initial_unaligned_segment_id_in_confusion_zone
                    # +1, since we will prepend this first segment (it might required prefix removal)
                    + 1 : first_segment_to_continue_aligning.id
                ]

            # prepend the initial segment (which might have had prefix removal)
            # containing only the part which is skipped
            confusing_segments_to_skip = [initial_unaligned_segment_in_confusion_zone] + confusing_segments_to_skip

            # Before committing the skipped segments - we will align them to the audio slice they reside over
            # this would not create high quality alignment probably - but will produce (arbitrary) word timings
            # that allow those segments to co-exist with the properly aligned segments.
            skipped_text_to_align = get_text_from_segments(confusing_segments_to_skip)
            logger.warning(f"[SKIP-TRACE] confusing_segments_to_skip: count={len(confusing_segments_to_skip)}, "
                           f"text_len={len(skipped_text_to_align)}, "
                           f"first_id={confusing_segments_to_skip[0].id if hasattr(confusing_segments_to_skip[0], 'id') else 'synth'}, "
                           f"last_id={confusing_segments_to_skip[-1].id if hasattr(confusing_segments_to_skip[-1], 'id') else 'synth'}, "
                           f"first_segment_to_continue_id={first_segment_to_continue_aligning.id if first_segment_to_continue_aligning else None}")
            # Log the text gap: text committed so far + skipped text + text to continue should == total
            _committed_so_far = sum(len(s.text) for s in aligned_pieces)
            _continue_text_len = sum(len(s.text) for s in segments_after_assumed_confusion_zone) if segments_after_assumed_confusion_zone else 0
            _total_accounted = _committed_so_far + len(skipped_text_to_align) + _continue_text_len
            _original_len = len(get_text_from_segments(unaligned.segments))
            logger.warning(f"[SKIP-TRACE] TEXT ACCOUNTING: committed={_committed_so_far} + skipped={len(skipped_text_to_align)} + continue={_continue_text_len} = {_total_accounted} vs original={_original_len} (gap={_original_len - _total_accounted})")
            align_skipped_start_from = max(top_aligned_timestamp, confusing_segments_to_skip[0].start)
            align_skipped_end_at = (
                first_segment_to_continue_aligning.start if first_segment_to_continue_aligning else None
            )
            audio = SeekableAudioLoader(
                str(audio_file),
                sr=SAMPLE_RATE,
                stream=True,
                load_sections=[[align_skipped_start_from, align_skipped_end_at]],
                test_first_chunk=False,
            )

            # this may break due to low prob - that's ok - we given up on those segments for now.
            aligned_skipped: stable_whisper.WhisperResult = model.align(
                audio, skipped_text_to_align, language=language, failure_threshold=zero_duration_segments_failure_ratio
            )
            _sanitize_align_result(aligned_skipped, skipped_text_to_align)

            # Ensure none of the segments has a start/end below the top aligned timestamp
            # or over the confusion zone audio slice end
            for segment in aligned_skipped:
                for word in segment.words:
                    word.start = max(word.start, top_aligned_timestamp)
                    word.end = max(word.end, word.start)
                    if align_skipped_end_at:
                        word.end = min(align_skipped_end_at, word.end)
                    # Start cannot be above the end
                    word.start = min(word.end, word.start)

            # Remove text that may have been duplicated across the main alignment
            # pass (zero-duration tail) and this skip pass (zero-duration head).
            deduped_skipped = _remove_cross_call_text_overlap(aligned_pieces, aligned_skipped.segments)
            _skipped_before_dedup = sum(len(s.text) for s in aligned_skipped.segments)
            _skipped_after_dedup = sum(len(s.text) for s in deduped_skipped)
            if _skipped_before_dedup != _skipped_after_dedup:
                logger.warning(f"[SKIP-TRACE] cross-call dedup removed {_skipped_before_dedup - _skipped_after_dedup} chars "
                               f"(before={_skipped_before_dedup}, after={_skipped_after_dedup})")
            aligned_pieces.extend(deduped_skipped)  # consider this done (although it's unaligned == estimated)

            # Mark the top text we took from the unaligned - so we cannot match earlier than that
            # on next iterations
            top_matched_unaligned_timestamp = confusing_segments_to_skip[-1].end

            # Advance segments_after_assumed_confusion_zone past any segments
            # that were already committed as confusing_segments_to_skip, so that
            # to_align_next (set below) does not include text already in
            # aligned_pieces.  This matters when unaligned segments start well
            # after max_confusion_zone_end (e.g. long audio pre-roll) causing
            # the "after" set to overlap with the "around" set.
            #
            # The skipped segments cover unaligned ids:
            #   initial_unaligned_segment_id_in_confusion_zone  ..  first_segment_to_continue_aligning.id - 1
            # (plus the initial segment itself which may have been prefix-trimmed).
            # We must ensure segments_after does not include any of these.
            last_skipped_original_id = max(
                initial_unaligned_segment_id_in_confusion_zone,
                first_segment_to_continue_aligning.id - 1 if first_segment_to_continue_aligning else initial_unaligned_segment_id_in_confusion_zone,
            )
            prev_after_count = len(segments_after_assumed_confusion_zone)
            segments_after_assumed_confusion_zone = [
                s for s in segments_after_assumed_confusion_zone if s.id > last_skipped_original_id
            ]
            if len(segments_after_assumed_confusion_zone) != prev_after_count:
                logger.warning(f"[SKIP-TRACE] overlap filter removed {prev_after_count - len(segments_after_assumed_confusion_zone)} segments from segments_after "
                               f"(last_skipped_original_id={last_skipped_original_id})")
                if not segments_after_assumed_confusion_zone:
                    done = True
                    first_segment_to_continue_aligning = None
                else:
                    first_segment_to_continue_aligning = segments_after_assumed_confusion_zone[0]

        logger.warning(f"[SKIP-TRACE] === End of skip logic: done={done}, "
                       f"aligned_pieces count={len(aligned_pieces)}, "
                       f"total text in pieces={sum(len(s.text) for s in aligned_pieces)}")

        # Prepare for next align attempt
        if not done:
            if not segments_around_confusion_zone and probable_segment_before_confusion_zone is not None:
                # Edge case: the skip logic found no unaligned segments
                # overlapping the confusion zone (e.g. the alignment ran
                # against a long pre-roll of wrong audio), but we already
                # committed segments at the probable-segment path above.
                # Keep to_align_next as set at line 180 (the text after the
                # probable segment) to avoid re-including text that was
                # already committed to aligned_pieces.  We only need to
                # fix slice_start to point to the correct audio position
                # for that text: use the first unaligned segment that
                # starts after the confusion zone.
                slice_start = first_segment_to_continue_aligning.start
            else:
                to_align_next = get_text_from_segments(segments_after_assumed_confusion_zone)
                logger.warning(f"[SKIP-TRACE] to_align_next set from segments_after: len={len(to_align_next)}, "
                               f"first 80={to_align_next[:80]!r}, "
                               f"slice_start will be={first_segment_to_continue_aligning.start}")

                # next audio start is the confusion zone end or the start of the
                # first segment to align - which ever comes first
                # slice_start = min(max_confusion_zone_end, first_segment_to_continue_aligning.start)
                slice_start = first_segment_to_continue_aligning.start

            # Update progress bar
            progress_bar.update(slice_start - progress_bar.n)

        # Forget prev confusion zone
        min_confusion_zone_start = 0
        max_confusion_zone_end = 0

    final_aligned = create_transcript_from_segments(aligned_pieces)

    # Close the progress bar
    progress_bar.close()

    return final_aligned
