"""
Camelot wheel mapping + harmonic mixing compatibility, the same system every
working DJ software (rekordbox, Serato, Mixed In Key) uses for key display.

Wheel has 12 positions, each split into a major (B) and relative minor (A)
key. Adjacent numbers and same-number A/B pairs are "harmonically compatible,"
which is the whole trick to mixing two tracks in different keys without it
sounding like a trainwreck.
"""

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# pitch class -> camelot number, for major (B) keys, built off the circle of
# fifths starting at C = 8B
_CAMELOT_MAJOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}

# same thing for minor (A) keys, each minor key shares a number with its
# relative major, so this is just _CAMELOT_MAJOR reindexed by relative minor
# pitch class (minor root = major root - 3 semitones)
_CAMELOT_MINOR = {(pc - 3) % 12: num for pc, num in _CAMELOT_MAJOR.items()}


def to_camelot(pitch_class, is_minor):
    table = _CAMELOT_MINOR if is_minor else _CAMELOT_MAJOR
    letter = "A" if is_minor else "B"
    return f"{table[pitch_class % 12]}{letter}"


def key_name(pitch_class, is_minor):
    return f"{PITCH_CLASSES[pitch_class % 12]}{'m' if is_minor else ''}"


def _parse(code):
    return int(code[:-1]), code[-1]


def camelot_distance(code_a, code_b):
    """0 = identical key. Small = compatible. Large = clash.

    Rules, in order of how "safe" the transition is:
      same code              -> 0.0   (perfect match)
      adjacent number, same letter -> 0.2   (classic harmonic mix, shifts energy slightly)
      same number, other letter    -> 0.3   (relative major/minor swap)
      two steps, same letter       -> 0.6   (energy-boost move, used sparingly)
      anything else                -> 1.0   (treat as a clash)
    """
    n_a, l_a = _parse(code_a)
    n_b, l_b = _parse(code_b)

    if n_a == n_b and l_a == l_b:
        return 0.0

    ring_diff = min((n_a - n_b) % 12, (n_b - n_a) % 12)

    if l_a == l_b and ring_diff == 1:
        return 0.2
    if l_a != l_b and n_a == n_b:
        return 0.3
    if l_a == l_b and ring_diff == 2:
        return 0.6
    return 1.0


if __name__ == "__main__":
    print(to_camelot(0, False), "should be 8B")
    print(to_camelot(9, True), "should be 8A")
    print(to_camelot(7, False), "should be 9B")
    print("8A -> 8A:", camelot_distance("8A", "8A"))
    print("8A -> 9A:", camelot_distance("8A", "9A"))
    print("8A -> 8B:", camelot_distance("8A", "8B"))
    print("8A -> 2A:", camelot_distance("8A", "2A"))
