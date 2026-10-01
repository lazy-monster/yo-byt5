"""Evaluation metrics for Yorùbá diacritic restoration.

All metrics compare a list of reference sentences with a list of system
outputs. `evaluate` is the entry point used for every number in the paper.

  WER, CER       Levenshtein distance over words and characters.
  DER            Share of marked reference letters that the output marks
                 differently. The unit is the letter: a letter with a tone
                 mark and an underdot counts once.
  DER_tone       DER over tone marks (grave, acute, macron) only.
  DER_underdot   DER over underdots only.
  WDER           Share of marked reference words that the output gets wrong.

DER, WDER and the error typology first align reference and output words on
their unmarked base forms (Needleman-Wunsch), so an inserted or dropped word
is charged to that word alone. The alignment is one-to-one: a split or merged
word aligns as a substitution plus a gap.
"""

import re
import unicodedata
from collections import Counter, namedtuple

GRAVE, ACUTE, MACRON = "̀", "́", "̄"
DOT_BELOW = "̣"
# Some YAD references encode the underdot as U+0329, which no Unicode
# normalisation form maps to U+0323.
VERTICAL_LINE_BELOW = "̩"
TONE_MARKS = {GRAVE, ACUTE, MACRON}
UNDERDOTS = {DOT_BELOW}

TEXT_ALTERING = ("lexical", "deletion", "insertion")


# ------------------------------------------------------------------ encoding

def repair(s):
    """Return `s` in NFD with every underdot encoded as U+0323, once."""
    s = unicodedata.normalize("NFD", s)
    s = s.replace(VERTICAL_LINE_BELOW, DOT_BELOW)
    s = re.sub(DOT_BELOW + "{2,}", DOT_BELOW, s)
    return unicodedata.normalize("NFD", s)


def prepare(s, do_repair=True):
    return repair(s) if do_repair else unicodedata.normalize("NFD", s)


def base_of(s):
    """Return `s` with every combining mark removed."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def marks_by_base(s):
    """Split an NFD string into (base character, set of marks) units."""
    units = []
    for ch in s:
        if unicodedata.combining(ch):
            if units:
                units[-1][1].add(ch)
        else:
            units.append([ch, set()])
    return [(b, m) for b, m in units]


# ------------------------------------------------------------- edit distance

def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _rate(counts):
    errors = sum(e for e, _ in counts)
    total = sum(t for _, t in counts)
    return errors / total if total else 0.0


def wer_counts(refs, hyps):
    """Per-sentence (word edits, reference words)."""
    return [(_levenshtein(r.split(), h.split()), len(r.split())) for r, h in zip(refs, hyps)]


def cer_counts(refs, hyps):
    """Per-sentence (character edits, reference characters)."""
    return [(_levenshtein(list(r), list(h)), len(r)) for r, h in zip(refs, hyps)]


def wer(refs, hyps):
    return _rate(wer_counts(refs, hyps))


def cer(refs, hyps):
    return _rate(cer_counts(refs, hyps))


# ------------------------------------------------------------ word alignment

Sentence = namedtuple("Sentence", "rw hw rb hb ru hu pairs")


def align_words(rb, hb):
    """Needleman-Wunsch alignment of two lists of unmarked words.

    A match costs 0. A substitution, a dropped reference word and an inserted
    output word each cost 1. Returns (ref_index | None, hyp_index | None)
    pairs in reference order.
    """
    n, m = len(rb), len(hb)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "U"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0 if rb[i - 1] == hb[j - 1] else 1)
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            back[i][j] = "D" if best == sub else "U" if best == dele else "L"

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "D":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "U":
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def align_sentence(r, h):
    rw, hw = r.split(), h.split()
    rb, hb = [base_of(w) for w in rw], [base_of(w) for w in hw]
    return Sentence(rw, hw, rb, hb,
                    [marks_by_base(w) for w in rw],
                    [marks_by_base(w) for w in hw],
                    align_words(rb, hb))


def align_all(refs, hyps):
    """Align every sentence pair once, for reuse across metrics."""
    return [align_sentence(r, h) for r, h in zip(refs, hyps)]


def _sentences(refs, hyps, aligned):
    if aligned is None:
        return align_all(refs, hyps)
    if len(aligned) != len(refs):
        raise ValueError(f"aligned covers {len(aligned)} sentences, refs has {len(refs)}")
    return aligned


# ------------------------------------------------------------------ DER family

def _want(kind):
    if kind == "tone":
        return TONE_MARKS
    if kind == "underdot":
        return UNDERDOTS
    return TONE_MARKS | UNDERDOTS


def der_counts(refs, hyps, kind="all", aligned=None):
    """Per-sentence (wrong letters, marked reference letters) for DER.

    A reference word with no aligned partner, or whose partner has a
    different base form, counts all of its marked letters as wrong. A letter
    that the reference leaves unmarked is outside DER even if the output
    marks it.
    """
    want = _want(kind)
    counts = []
    for sent in _sentences(refs, hyps, aligned):
        wrong = total = 0
        for ri, hi in sent.pairs:
            if ri is None:
                continue
            ru = sent.ru[ri]
            rletters = sum(1 for _, m in ru if m & want)
            if not rletters:
                continue
            total += rletters
            if hi is None or sent.hb[hi] != sent.rb[ri]:
                wrong += rletters
                continue
            for (_, rm), (_, hm) in zip(ru, sent.hu[hi]):
                rs = rm & want
                if rs and (hm & want) != rs:
                    wrong += 1
        counts.append((wrong, total))
    return counts


def der(refs, hyps, kind="all", aligned=None):
    """Diacritic error rate. `kind` is 'all', 'tone' or 'underdot'."""
    return _rate(der_counts(refs, hyps, kind, aligned))


def wder_counts(refs, hyps, aligned=None):
    """Per-sentence (wrong marked words, marked reference words)."""
    counts = []
    for sent in _sentences(refs, hyps, aligned):
        bad = total = 0
        for ri, hi in sent.pairs:
            if ri is None:
                continue
            if not any(m & (TONE_MARKS | UNDERDOTS) for _, m in sent.ru[ri]):
                continue
            total += 1
            if hi is None or sent.hw[hi] != sent.rw[ri]:
                bad += 1
        counts.append((bad, total))
    return counts


def wder(refs, hyps, aligned=None):
    """Share of marked reference words that the output does not reproduce."""
    return _rate(wder_counts(refs, hyps, aligned))


# ------------------------------------------------------------ error typology

def taxonomy(refs, hyps, aligned=None):
    """Count aligned words by error class.

    correct        output word equals the reference word
    tone_only      same base form, a tone mark differs
    underdot_only  same base form, an underdot differs
    mixed          same base form, both kinds differ
    lexical        different base form
    deletion       reference word with no output partner
    insertion      output word with no reference partner
    other          same base form and same tone marks and underdots, but a
                   different string (for example a mark outside both sets)
    """
    tally = Counter()
    for sent in _sentences(refs, hyps, aligned):
        for ri, hi in sent.pairs:
            if ri is None:
                tally["insertion"] += 1
                continue
            if hi is None:
                tally["deletion"] += 1
                continue
            if sent.hw[hi] == sent.rw[ri]:
                tally["correct"] += 1
                continue
            if sent.hb[hi] != sent.rb[ri]:
                tally["lexical"] += 1
                continue
            tone_bad = underdot_bad = False
            for (_, rm), (_, hm) in zip(sent.ru[ri], sent.hu[hi]):
                if (rm & TONE_MARKS) != (hm & TONE_MARKS):
                    tone_bad = True
                if (rm & UNDERDOTS) != (hm & UNDERDOTS):
                    underdot_bad = True
            if tone_bad and underdot_bad:
                tally["mixed"] += 1
            elif tone_bad:
                tally["tone_only"] += 1
            elif underdot_bad:
                tally["underdot_only"] += 1
            else:
                tally["other"] += 1
    return tally


# ---------------------------------------------------------------- evaluation

def evaluate(refs, hyps, do_repair=True):
    """Score outputs against references.

    Both sides are converted to NFD. With `do_repair`, U+0329 underdots and
    doubled underdots are also corrected on both sides; this is the corrected
    YAD test set used in the paper. `typology` gives each error class as a
    percentage of all wrong words, and `text_altering_rate` is the share of
    wrong words that change the base text.
    """
    R = [prepare(x, do_repair) for x in refs]
    H = [prepare(x, do_repair) for x in hyps]
    aligned = align_all(R, H)
    tax = taxonomy(R, H, aligned=aligned)
    errors = sum(v for k, v in tax.items() if k != "correct")
    typology = {k: (v / errors * 100 if errors else 0.0)
                for k, v in tax.items() if k != "correct"}
    return {
        "WER": wer(R, H),
        "CER": cer(R, H),
        "DER": der(R, H, "all", aligned=aligned),
        "DER_tone": der(R, H, "tone", aligned=aligned),
        "DER_underdot": der(R, H, "underdot", aligned=aligned),
        "WDER": wder(R, H, aligned=aligned),
        "typology": typology,
        "text_altering_rate": sum(typology.get(k, 0.0) for k in TEXT_ALTERING) / 100,
    }


def bleu_chrf(refs, hyps):
    """Corpus BLEU and ChrF from sacrebleu defaults, on a 0-1 scale."""
    import sacrebleu
    return {
        "BLEU": sacrebleu.corpus_bleu(hyps, [refs]).score / 100,
        "ChrF": sacrebleu.corpus_chrf(hyps, [refs]).score / 100,
    }


# ------------------------------------------------------------------ selftest

def selftest():
    """Check the metric definitions on small cases with known answers."""
    ref = ["Ẹ̀kọ́ ni kọ́kọ́rọ́ àṣeyọrí.", "Ọmọ tí ó bá kàwé rẹ̀ á jàre."]

    # Copying the reference scores 0 everywhere.
    oracle = evaluate(ref, ref)
    for k in ("WER", "CER", "DER", "DER_tone", "DER_underdot", "WDER", "text_altering_rate"):
        assert oracle[k] == 0.0, f"oracle {k} = {oracle[k]}"

    # Copying the unmarked input scores DER 100%.
    src = ["Eko ni kokoro aseyori.", "Omo ti o ba kawe re a jare."]
    assert evaluate(ref, src)["DER"] == 1.0

    # One reference letter with an underdot and an acute.
    marked = prepare("ọ́")
    tone_lost = prepare("ọ")
    underdot_lost = prepare("ó")
    tone_wrong = prepare("ọ̀")

    assert der([marked], [marked]) == 0.0
    # A wrong tone mark is one wrong letter, so DER never exceeds 100%.
    assert der([marked], [tone_wrong], "tone") == 1.0
    # Losing one or both marks of a letter is one wrong letter.
    assert der([marked], [tone_lost]) == 1.0
    assert der([marked], [underdot_lost]) == 1.0
    assert der([marked], [tone_wrong]) == 1.0
    # Each family ignores the other.
    assert der([marked], [tone_lost], "underdot") == 0.0
    assert der([marked], [underdot_lost], "tone") == 0.0
    assert der([marked], [tone_wrong], "underdot") == 0.0

    # An added mark and a removed mark cost the same.
    single = prepare("ó")
    assert der([single], [prepare("ọ́")]) == der([single], [prepare("o")]) == 1.0

    # A dropped, inserted or changed word costs only that word.
    # Six marked letters in four words, two of them on "ọjà".
    full = prepare("Ilé ọjà títí dé")
    dropped = prepare("Ilé títí dé")
    inserted = prepare("Ilé ọjà kan títí dé")
    changed = prepare("Ilé ọjà tít dé")
    assert der([full], [dropped]) == 2 / 6
    assert dict(taxonomy([full], [dropped])) == {"correct": 3, "deletion": 1}
    assert der([full], [inserted]) == 0.0
    assert dict(taxonomy([full], [inserted])) == {"correct": 4, "insertion": 1}
    assert der([full], [changed]) == 2 / 6
    assert dict(taxonomy([full], [changed])) == {"correct": 3, "lexical": 1}


if __name__ == "__main__":
    selftest()
    print("selftest passed")
