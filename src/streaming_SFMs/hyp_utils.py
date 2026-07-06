import sys
import Levenshtein

import logging
logger = logging.getLogger(__name__)

from streaming_sfm import LOG_LEVEL
logger.setLevel(LOG_LEVEL)

# Punctuation marks that should not be duplicated at chunk boundaries.
PUNCT = [',', '.', '!', '?']

# _slcp_stable_flags lives in the SLCP helpers module (rapidfuzz / spacy).
# Imported lazily so the rest of hyp_utils has no hard dependency on those libs.
try:
    from segfreetk.states.sclp import _slcp_stable_flags
    _SLCP_AVAILABLE = True
except ImportError:
    _SLCP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Utility: group SentencePiece tokens into words
# ---------------------------------------------------------------------------

def clean_word(w):
    """Strip punctuation and whitespace for comparison."""
    return w.strip().lower().translate(str.maketrans('', '', "".join(PUNCT)))
    

def tokens_to_words(tokens):
    """
    Convert a list of token-level tuples (start, end, token_str) into
    word-level tuples (start, end, word_str).

    SentencePiece marks the beginning of a new word with the '▁' prefix.
    All tokens that follow a '▁'-prefixed token (without their own '▁') are
    sub-word continuations of the same word.

    Example:
        [▁hel, lo, ▁world]  →  [(t0,t1,'▁hello'), (t2,t3,'▁world')]

    The '▁' is preserved on the first sub-word so callers can still detect
    word boundaries (consistent with the token-level convention used
    throughout the codebase).
    """
    if not tokens:
        return []

    words = []
    word_start = None
    word_end = None
    word_text = ""

    for start, end, tok in tokens:
        is_new_word = tok.startswith('▁')

        if is_new_word and word_text:
            # Flush the previous word
            words.append((word_start, word_end, word_text))
            word_text = ""
            word_start = None

        if word_start is None:
            word_start = start
        word_end = end
        word_text += tok          # concatenate: keeps '▁' on the first sub-word

    # Flush the last word
    if word_text:
        words.append((word_start, word_end, word_text))

    return words


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ABSHypothesisBuffer:
    def __init__(self, word_level: bool = False, debug: bool = False):
        self.buffer = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.word_level = word_level
        self.debug = debug

    def _maybe_to_words(self, tokens):
        """Optionally convert token list to word list before processing."""
        if self.word_level:
            return tokens_to_words(tokens)
        return tokens

    def insert(self, new, offset=0):
        raise NotImplementedError   # BUG FIX #1: was `return`, must be `raise`

    def flush(self):
        raise NotImplementedError   # BUG FIX #1

    def complete(self):
        return self.buffer

    def reset(self):                # BUG FIX #2: was @staticmethod with self param
        self.buffer = []
        self.commited_in_buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None


# ---------------------------------------------------------------------------
# LCP (Longest Common Prefix) policy
# ---------------------------------------------------------------------------

class LCPHypothesisBuffer(ABSHypothesisBuffer):
    """
    Commits tokens/words that are confirmed by the longest common prefix
    between the previous hypothesis and the current one.
    """

    def __init__(self, uncased: bool = True, word_level: bool = False, debug: bool = False):
        super().__init__(word_level=word_level, debug=debug)
        self.commited_in_buffer = []
        self.new = []
        self.uncased = uncased

    def insert(self, new, offset=0):
        """
        Compare self.commited_in_buffer with new.  Only keep the tokens in
        new that extend beyond what is already committed.
        """
        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> insert] Received hypothesis: {new}')

        new = [(a + offset, b + offset, t) for a, b, t in new]
        new = self._maybe_to_words(new)                          # ← word-level hook

        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(c_len, n_len, 5) + 1):
                        c = ' '.join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = ' '.join(self.new[j - 1][2] for j in range(1, i + 1))

                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            if self.debug:
                                logger.debug(
                                    f'[LCPHypothesisBuffer -> insert] Removing last {i} items: '
                                    + ' '.join(words)
                                )
                            break

    def flush(self, forced=False, speculative=False):
        return self.flush_uncased(forced=forced, speculative=speculative) if self.uncased else self.flush_cased(forced=forced, speculative=speculative)

    def flush_uncased(self, forced, speculative=False):
        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> flush] Buffer n-1: {self.buffer}')
            logger.debug(f'[LCPHypothesisBuffer -> flush] Buffer n  : {self.new}')

        if forced:
            if self.new and self.buffer:
                while self.new[0][2].lower() != self.buffer[0][2].lower():
                    self.buffer.pop(0)
                    if not self.buffer:
                        break
            if self.debug:
                logger.debug(f'[LCPHypothesisBuffer -> flush] Buffer n-1 after cleanup: {self.buffer}')

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if not self.buffer:
                break

            if nt.lower() == self.buffer[0][2].lower():
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []

        # BUG FIX #3: `punct` was referenced but never defined in this method.
        if self.commited_in_buffer and self.commited_in_buffer[-1][2] in PUNCT:
            if commit and commit[0][2] in PUNCT:
                commit = commit[1:]

        if not commit:
            if speculative:
                return commit, self.buffer
            else:
                return commit

        self.commited_in_buffer.extend(commit)
        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.buffer
        return commit

    def flush_cased(self, forced, speculative=False):
        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> flush] Buffer n-1: {self.buffer}')
            logger.debug(f'[LCPHypothesisBuffer -> flush] Buffer n  : {self.new}')

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if not self.buffer:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        if self.debug:
            logger.debug(f'[LCPHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.buffer
        return commit

    def pop_commited(self, time_limit):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_limit:
            self.commited_in_buffer.pop(0)

    def commit_anyway(self):
        """
        Commit the hypothesis buffer overriding the emission policy. Useful
        when the policy gets stuck due to a certain token or timestamp blocking
        the emission.

        It checks commited_in_buffer to avoid commiting any already commited
        word, and then returns the rest of words in the buffer
        """
        if self.debug:
            logger.debug('[LCPHypothesisBuffer -> commit_anyway] Buffer n-1: {self.buffer}')
        if len(self.buffer) >= 1:
            a, b, t = self.buffer[0]
            if self.commited_in_buffer:
                c_len = len(self.commited_in_buffer)
                b_len = len(self.buffer)

                for i in range(1, min(c_len, n_len, 5) + 1):
                    c = ' '.join(
                        [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                    )
                    tail = ' '.join(self.buffer[j - 1][2] for j in range(1, i + 1))

                    if c == tail:
                        words = []
                        for j in range(i):
                            words.append(repr(self.buffer.pop(0)))
                        if self.debug:
                            logger.debug(
                                f'[LCPHypothesisBuffer -> commit_anyway] Last {i} items '
                                + f'are shared between buffer and commited_in_buffer. Removing: '
                                + ' '.join(words)
                            )
                        break
        

    def complete(self):
        return self.buffer


# ---------------------------------------------------------------------------
# LACP (Levenshtein-Approximate Common Prefix) policy
# ---------------------------------------------------------------------------

class LACPHypothesisBuffer(ABSHypothesisBuffer):
    """
    Like LCP but uses Levenshtein distance instead of exact matching, so
    minor transcription corrections between chunks don't block emission.
    """

    def __init__(self, threshold: int = 2, uncased: bool = True,
                 word_level: bool = False, debug: bool = False):
        super().__init__(word_level=word_level, debug=debug)
        self.commited_in_buffer = []
        self.new = []
        self.uncased = uncased
        self.threshold = threshold

    def insert(self, new, offset=0):
        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> insert] Received hypothesis: {new}')

        new = [(a + offset, b + offset, t) for a, b, t in new]
        new = self._maybe_to_words(new)                          # ← word-level hook

        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(c_len, n_len, 5) + 1):
                        c = ' '.join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = ' '.join(self.new[j - 1][2] for j in range(1, i + 1))

                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            if self.debug:
                                logger.debug(
                                    f'[LACPHypothesisBuffer -> insert] Removing last {i} items: '
                                    + ' '.join(words)
                                )
                            break

    def flush(self, forced=False, speculative=False):
        return self.flush_uncased(forced=forced, speculative=speculative) if self.uncased else self.flush_cased(forced=forced, speculative=speculative)

    def flush_uncased(self, forced, speculative=False):
        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> flush] Buffer n-1: {self.buffer}')
            logger.debug(f'[LACPHypothesisBuffer -> flush] Buffer n  : {self.new}')

        commit = []
        past = []
        present = []

        # if forced, returning commit early maintains the n-1 buffer for the next iteration
        if forced and not self.new:
            if speculative:
                return commit, []
            else:
                return commit

        # BUG FIX #4: use a proper for-loop. `while i in range(...)` works
        # but recomputes membership each iteration and is semantically misleading.
        limit = min(len(self.new), len(self.buffer))
        for i in range(limit):
            past.append(self.buffer[i])
            present.append(self.new[i])

            past_s    = ' '.join(t for _, _, t in past).lower()
            present_s = ' '.join(t for _, _, t in present).lower()

            if Levenshtein.distance(past_s, present_s) > self.threshold:
                break

            commit.append(self.new[i])
        # `i` is the last index checked; non-committed tail starts at len(commit).
        self.buffer = self.new[len(commit):]
        self.new = []

        if self.commited_in_buffer and self.commited_in_buffer[-1][2] in PUNCT:
            if commit and commit[0][2] in PUNCT:
                commit = commit[1:]

        if not commit:
            if speculative:
                return commit, self.buffer
            else:
                return commit

        self.last_commited_word = commit[-1][2]
        self.last_commited_time = commit[-1][1]
        self.commited_in_buffer.extend(commit)

        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.buffer
        return commit

    def flush_cased(self, forced, speculative=False):
        """
        BUG FIX #5: The original flush_cased was an exact-match copy of
        LCPHypothesisBuffer.flush_cased, completely ignoring the Levenshtein
        threshold.  This version applies the threshold correctly for the cased
        (case-sensitive) variant.
        """
        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> flush] Buffer n-1: {self.buffer}')
            logger.debug(f'[LACPHypothesisBuffer -> flush] Buffer n  : {self.new}')

        commit = []
        past = []
        present = []

        limit = min(len(self.new), len(self.buffer))
        for i in range(limit):
            past.append(self.buffer[i])
            present.append(self.new[i])

            # Case-sensitive: do NOT lowercase
            past_s    = ' '.join(t for _, _, t in past)
            present_s = ' '.join(t for _, _, t in present)

            if Levenshtein.distance(past_s, present_s) > self.threshold:
                break

            commit.append(self.new[i])

        self.buffer = self.new[len(commit):]
        self.new = []

        if not commit:
            if speculative:
                return commit, self.buffer
            else:
                return commit

        self.last_commited_word = commit[-1][2]
        self.last_commited_time = commit[-1][1]
        self.commited_in_buffer.extend(commit)

        if self.debug:
            logger.debug(f'[LACPHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.buffer
        return commit

    def pop_commited(self, time_limit):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_limit:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


# ---------------------------------------------------------------------------
# Wait-K policy
# ---------------------------------------------------------------------------

class WaitKHypothesisBuffer(ABSHypothesisBuffer):
    """
    Holds back K seconds of audio before emitting, giving the model time to
    refine its hypothesis.
    """

    def __init__(self, K: int = 1, features_per_second: int = 12,
                 subsampling_factor: int = 4, word_level: bool = False, debug: bool = False):
        super().__init__(word_level=word_level, debug=debug)
        self.new = []
        self.commited_in_buffer = []

        self.K = int(K * features_per_second / subsampling_factor)
        if self.debug:
            logger.debug(
                f'[WaitKHypothesisBuffer -> __init__] K={K}s → {self.K} encoder frames'
            )

    def insert(self, new, offset=0):
        if self.debug:
            logger.debug(f'[WaitKHypothesisBuffer -> insert] Received hypothesis: {new}')

        new = [(a + offset, b + offset, t) for a, b, t in new]
        new = self._maybe_to_words(new)                          # ← word-level hook

        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.debug:
            logger.debug(f'[WaitKHypothesisBuffer -> insert] Processed hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    c_len = len(self.commited_in_buffer)
                    n_len = len(self.new)

                    for i in range(1, min(c_len, n_len, 5) + 1):
                        c = ' '.join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = ' '.join(self.new[j - 1][2] for j in range(1, i + 1))

                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            if self.debug:
                                logger.debug(
                                    f'[WaitKHypothesisBuffer -> insert] Removing last {i} items: '
                                    + ' '.join(words)
                                )
                            break

    def flush(self, last_instant, speculative=False):
        last_valid_instant = last_instant - self.K

        if self.debug:
            logger.debug(f'[WaitKHypothesisBuffer -> flush] Last valid instant: {last_valid_instant}')
            logger.debug(f'[WaitKHypothesisBuffer -> flush] Committed buffer: {self.buffer}')
            logger.debug(f'[WaitKHypothesisBuffer -> flush] New buffer: {self.new}')

        commit = []
        if last_valid_instant > 0:
            for na, nb, nt in self.new:
                if na > last_valid_instant:
                    break
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb

            self.buffer.extend(commit)
            self.new = self.new[len(commit):]

        if self.debug:
            logger.debug(f'[WaitKHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.new
        return commit

    def pop_commited(self):
        return self.buffer

    def complete(self):
        return self.new


# ---------------------------------------------------------------------------
# Hold-N policy
# ---------------------------------------------------------------------------

class HoldNHypothesisBuffer(ABSHypothesisBuffer):
    """
    Always withholds the last N tokens/words before committing, ensuring the
    model has seen enough future context to be confident about earlier output.
    """

    def __init__(self, N: int = 3, word_level: bool = False, debug: bool = False):
        super().__init__(word_level=word_level, debug=debug)
        self.new = []
        self.commited_in_buffer = []
        self.N = N

    def insert(self, new, offset=0):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        new = self._maybe_to_words(new)                          # ← word-level hook

        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.debug:
            logger.debug(f'[HoldNHypothesisBuffer -> insert] Received hypothesis: {self.new}')

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.buffer:
                    c_len = len(self.buffer)
                    n_len = len(self.new)

                    for i in range(1, min(c_len, n_len, 10) + 1):
                        c = ' '.join(
                            [self.buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = ' '.join(self.new[j - 1][2] for j in range(1, i + 1))

                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            if self.debug:
                                logger.debug(
                                    f'[HoldNHypothesisBuffer -> insert] Removing last {i} items: '
                                    + ' '.join(words)
                                )
                            break

    def flush(self, speculative=False):
        if self.debug:
            logger.debug(f'[HoldNHypothesisBuffer -> flush] New buffer: {self.new}')
            logger.debug(f'[HoldNHypothesisBuffer -> flush] Committed buffer: {self.buffer}')

        if len(self.new) > self.N:
            commit = self.new[:-self.N]
            self.new = self.new[-self.N:]
            self.buffer.extend(commit)
            self.last_commited_time = commit[-1][1]
            self.last_commited_word = commit[-1][2]
        else:
            commit = []

        if self.debug:
            logger.debug(f'[HoldNHypothesisBuffer -> flush] Committing {commit}')
        if speculative:
            return commit, self.new
        return commit

    def pop_commited(self, time_limit):
        while self.buffer and self.buffer[0][1] <= time_limit:
            self.buffer.pop(0)

    def complete(self):
        return self.new

# ---------------------------------------------------------------------------
# SLCP (Stable LCP / Ripple) policy
# ---------------------------------------------------------------------------

class SLCPHypothesisBuffer(ABSHypothesisBuffer):
    """
    Stable LCP — a superset of LCP that also commits tokens that became
    stable because of what the model added *after* them.

    Stability is determined by three passes over the Levenshtein edit script
    between the previous and current full hypothesis (see _slcp_stable_flags):

      Pass 1 — tokens in 'equal' blocks → always stable.
      Pass 2 — tokens in right-anchored 'insert' blocks → stable because an
               'equal' block follows, confirming the context ("ripple effect").
      Pass 3 — tokens in 1-to-1 'replace' blocks that are linguistically or
               morphologically equivalent (optional; requires spacy or a
               surface-form similarity threshold).
      Pass 4 — gap bridging: propagate the stable prefix leftward from the
               rightmost valid anchor, filling gaps up to max_gap tokens.

    State that survives across flush() calls
    ----------------------------------------
    _committed_tokens : list[str]
        Running list of every token string committed so far for this sentence.
        NEVER shrinks (unlike commited_in_buffer which can be popped).
        Used as the cursor anchor and as the committed prefix in curr_hyp.

    _prev_hyp_tokens : list[str]
        Snapshot of the full hypothesis (committed + new) at the end of the
        last flush().  This is what the current hypothesis is diffed against.

    Parameters
    ----------
    semantic_threshold : float | None
        If set (and nlp is None), morphological similarity fallback is used
        in Pass 3.
    nlp : spacy Language | None
        If set, spacy annotations are used for Pass 3.
    linguistic_checks : _LinguisticChecks | None
        Bitfield controlling which spacy criteria are active in Pass 3.
        Required when nlp is not None.
    max_gap : int | None
        Maximum number of consecutive unstable tokens that can be bridged
        when propagating the stable prefix from the rightmost valid anchor.
    """

    def __init__(
        self,
        semantic_threshold=None,
        nlp=None,
        linguistic_checks=None,
        max_gap=None,
        word_level: bool = False,
        debug: bool = False,
        max_history: int = 20,  # Added: limit for the rolling window
    ):
        if not _SLCP_AVAILABLE:
            raise ImportError(
                "SLCPHypothesisBuffer requires 'streaming_sfm.slcp_helpers' "
                "(rapidfuzz and optionally spacy)."
            )
        super().__init__(word_level=word_level, debug=debug)

        self.commited_in_buffer = []   # (start, end, token) — may be popped
        self.new = []                  # current step's un-committed tokens

        # Full running history — never shrinks, used for cursor + diff.
        self._committed_tokens = []    # list[str]
        self._prev_hyp_tokens  = []    # list[str]

        self.semantic_threshold = semantic_threshold
        self.nlp = nlp
        self.linguistic_checks = linguistic_checks
        self.max_gap = max_gap
        self.max_history = max_history

    # ------------------------------------------------------------------
    # insert
    # ------------------------------------------------------------------

    def insert(self, new, offset=0):
        """
        Receive the latest hypothesis from the model and store it in self.new.

        Applies the same offset/time-filter as other buffers, then removes any
        leading tokens that overlap with the committed tail so that self.new
        truly represents the un-committed portion of the hypothesis.
        """
        if self.debug:
            logger.debug(f'[SLCPHypothesisBuffer -> insert] Received: {new}')

        logger.debug(self.last_commited_time)

        new = [(a + offset, b + offset, t) for a, b, t in new]
        new = self._maybe_to_words(new)
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]
        logger.debug(f'[SLCPHypothesisBuffer -> insert] Processed: {self.new}')
        
        # Overlap deduplication: remove tokens from self.new that are already
        # in the committed tail — same pattern as all other buffers.
        #(1113, 1117, '▁output,'), (1118, 1119, '▁that'), (1119, 1120, '▁is'), (1121, 1123, '▁the'), (1123, 1134, '▁cross-attention'), (1134, 1137, '▁mechanism.')
        #last 1113
        #1113 - 1113 = 0
        from difflib import SequenceMatcher
        #6 es new. commtie es el buffer, ponle 20 # el min es #5,6,30
        if self.new and self._committed_tokens:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                c_len = len(self._committed_tokens)
                n_len = len(self.new)
                for i in range(1, min(c_len, n_len, 5) + 1):
                    c    = ' '.join(self._committed_tokens[-i:])  # <- | ->
                    tail = ' '.join(tok for _, _, tok in self.new[:i])
                    #if c == tail:
                    if SequenceMatcher(None, c.replace('▁', ''), tail.replace('▁', '')).ratio() > 0.6:
                        for _ in range(i):
                            logger.debug(f"[SLCPHypothesisBuffer] -> Popping {self.new[0]}")
                            self.new.pop(0)
                        logger.debug(
                            f'[SLCPHypothesisBuffer -> insert] '
                            f'Removed {i} already-committed tokens'
                        )
                        break


    # ------------------------------------------------------------------
    # flush
    # ------------------------------------------------------------------

    def flush(self, speculative=False):
            """
            Calculates stability using a sliding window and commits stable tokens.
            """
            # 1. TRIMMING LOGIC: Keep the rolling window manageable
            if len(self._committed_tokens) > self.max_history:
                trim_size = len(self._committed_tokens) - self.max_history
                self._committed_tokens = self._committed_tokens[trim_size:]
                self._prev_hyp_tokens = self._prev_hyp_tokens[trim_size:]

            # 2. Construct current full hypothesis
            curr_tokens = self._committed_tokens + [t for _, _, t in self.new]
            prev_tokens = self._prev_hyp_tokens
            
            # The cursor marks where the uncommitted 'new' tokens begin in the curr_tokens list
            cursor = len(self._committed_tokens)

            logger.debug(f'[SLCPHypothesisBuffer -> flush] prev_hyp : {prev_tokens}')
            logger.debug(f'[SLCPHypothesisBuffer -> flush] curr_hyp : {curr_tokens}')
            logger.debug(f'[SLCP] Window Size: {len(curr_tokens)}, Cursor: {cursor}')

            # Update previous hypothesis snapshot for the next flush call
            self._prev_hyp_tokens = curr_tokens

            if not prev_tokens:
                logger.debug('[SLCPHypothesisBuffer -> flush] First hypothesis, emitting nothing')
                if speculative:
                    return [], []
                else:
                    return []

            # 3. Compute stable flags
            # This calls the helper that runs the Levenshtein-based stability passes
            stable = _slcp_stable_flags(
                prev_tokens=[ w.replace('▁', '') for w in prev_tokens],
                curr_tokens=[ w.replace('▁', '') for w in curr_tokens],
                semantic_threshold=self.semantic_threshold,
                nlp=self.nlp,
                linguistic_checks=self.linguistic_checks,
                max_gap=self.max_gap,
            )

            # 4. Apply clean_word comparison for Pass 1 (Exact/Normalized match)
            # We define a helper to strip SentencePiece artifacts and punctuation
            def clean_word(w):
                return w.replace('▁', '').strip(''.join(PUNCT)).lower()

            # 5. Determine how many tokens from 'self.new' are stable
            k = 0
            for i in range(cursor, len(curr_tokens)):
                # A token is stable if the _slcp_stable_flags says so, 
                # or if it's an identical match to the previous hypothesis at that position.
                
                # Check if we have a corresponding token in the previous hypothesis to compare
                is_identical = False
                if i < len(prev_tokens):
                    is_identical = clean_word(prev_tokens[i]) == clean_word(curr_tokens[i])

                if stable[i] or is_identical:
                    k += 1
                else:
                    # Stop committing at the first sign of instability/divergence
                    break

            # 6. Extract the stable prefix and update internal state
            commit = self.new[:k]
            self.new = self.new[k:]

            if commit:
                self.last_commited_word = commit[-1][2]
                self.last_commited_time = commit[-1][1]
                self._committed_tokens.extend(t for _, _, t in commit)
                self.commited_in_buffer.extend(commit)
                
            logger.debug(f'[SLCPHypothesisBuffer -> flush] stable flags: {stable}')
            logger.debug(f'[SLCPHypothesisBuffer -> flush] Committing {k} tokens: {commit}')

            if speculative:
                return commit, self.new
            return commit

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def pop_commited(self, time_limit):
        """
        Prune commited_in_buffer up to time_limit.

        Note: _committed_tokens is intentionally NOT pruned — it must remain
        intact so that cursor arithmetic stays correct across pop calls.
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_limit:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.new
