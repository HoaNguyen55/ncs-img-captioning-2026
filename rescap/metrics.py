"""Image-captioning evaluation (PHASE 14).

Wraps `pycocoevalcap` so every experiment reports the same numbers computed the
same way.  Never report a single metric -- each one is blind to something:

    BLEU-n    n-gram precision against references.  Rewards fluent copying,
              correlates weakly with human judgement, saturates.  BLEU-4 is
              reported for comparability with the literature, not because it is
              a good metric.
    METEOR    unigram matching with stemming + WordNet synonyms.  Better
              correlation than BLEU, but needs a JVM and is slow.
    ROUGE-L   longest-common-subsequence recall.  Sensitive to length.
    CIDEr     TF-IDF-weighted n-gram consensus across all references.  The de
              facto primary captioning metric; corpus-level, so it is
              meaningless on a single sentence and unstable on tiny test sets.
    SPICE     scene-graph (object/attribute/relation) F1.  Closest to semantic
              adequacy, catches hallucinated objects that CIDEr misses, but
              needs a JVM, is very slow, and depends on a parser.

    CLIPScore / BERTScore are reference-free / embedding-based complements --
    useful when references are sparse, but they inherit their backbone's biases.

METEOR and SPICE require Java.  When it is absent they are reported as `None`,
never silently dropped, never substituted with another number.
"""

from __future__ import annotations

import contextlib
import glob
import os
from pathlib import Path
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def java_available() -> bool:
    return shutil.which("java") is not None


def legacy_java_home() -> str | None:
    """Locate a JDK <= 15 for SPICE.

    SPICE 1.0 (2016) serialises through the FST library, which the Java module
    system blocks from JDK 16 onward -- on a modern JVM it dies with an
    InaccessibleObjectException before scoring anything. METEOR is unaffected
    and runs fine on the default JVM.

    Install one with:  sudo apt-get install -y openjdk-11-jre-headless
    """
    for pattern in ("/usr/lib/jvm/java-11-*", "/usr/lib/jvm/java-8-*", "/usr/lib/jvm/java-1.11.*"):
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(path, "bin", "java")):
                return path
    return None


@contextlib.contextmanager
def _java(home: str | None):
    """Temporarily put a specific JVM first on PATH.

    pycocoevalcap invokes a bare `java`, so overriding PATH is the only way to
    steer SPICE at a different JVM without patching the installed package.
    """
    if not home:
        yield
        return
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + original
    try:
        yield
    finally:
        os.environ["PATH"] = original


# ---------------------------------------------------------------------------
_RDR_CACHE: dict[str, Any] = {}


def _rdrsegmenter():
    """`(segment_fn, version)` for VnCoreNLP's RDRSegmenter.

    Loaded once per process and cached: it starts a JVM, which is slow and must
    not happen per caption. Needs the **JDK**, not the JRE -- `py_vncorenlp`
    goes through `jnius`, which looks for `javac` and fails with
    "Unable to find javac" against a JRE-only install.
    """
    if "fn" in _RDR_CACHE:
        return _RDR_CACHE["fn"], _RDR_CACHE["version"]

    import os

    import py_vncorenlp

    save_dir = os.environ.get(
        "VNCORENLP_DIR",
        str(Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data")) / "vncorenlp"),
    )
    os.makedirs(save_dir, exist_ok=True)

    # Both halves, checked separately. A partial download -- models present, jar
    # missing -- makes `VnCoreNLP()` spawn a java process that dies immediately
    # and leaves Python waiting on a defunct child **forever**. That happened:
    # an evaluation sat for eleven minutes after generation finished, printing
    # nothing, with `[java] <defunct>` in the process table. A hang is the worst
    # failure mode available, because it looks like slow progress.
    # The FILE, not the directory. Checking only that `models/wordsegmenter`
    # exists passed on a machine whose copy of it was missing
    # `wordsegmenter.rdr`, and the failure moved from here into the JVM.
    jar = os.path.join(save_dir, "VnCoreNLP-1.2.jar")
    rdr = os.path.join(save_dir, "models", "wordsegmenter", "wordsegmenter.rdr")
    vocab = os.path.join(save_dir, "models", "wordsegmenter", "vi-vocab")
    if not (os.path.exists(jar) and os.path.exists(rdr)):
        py_vncorenlp.download_model(save_dir=save_dir)
    missing = [name for name, path in (("VnCoreNLP-1.2.jar", jar),
                                       ("models/wordsegmenter/wordsegmenter.rdr", rdr),
                                       ("models/wordsegmenter/vi-vocab", vocab))
               if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            f"VnCoreNLP is incomplete in {save_dir}: missing {missing}. "
            f"`download_model` calls wget — check that wget is installed. "
            f"Not initialising, to avoid hanging forever on a dead java process."
        )
    # py_vncorenlp resolves paths relative to the CURRENT WORKING DIRECTORY,
    # not to save_dir. Started from anywhere else the JVM comes up, prints
    # "Loading Word Segmentation model", and never returns -- no error, no
    # timeout, just a process that looks busy.
    #
    # This is what actually caused the stalled evaluations. I first blamed the
    # tokenize=False path, then a partial download, then a CUDA/JVM conflict,
    # and each guess survived because a hang gives no evidence. What separated
    # it was that every run from `/root` worked in 0.3 s and every run from the
    # repo directory hung.
    cwd = os.getcwd()
    try:
        os.chdir(save_dir)
        model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    finally:
        os.chdir(cwd)

    def segment(text: str) -> str:
        return " ".join(model.word_segment(text))

    _RDR_CACHE["fn"] = segment
    _RDR_CACHE["version"] = "VnCoreNLP-1.2/RDRSegmenter"
    return segment, _RDR_CACHE["version"]


@dataclass
class CaptionMetrics:
    """Compute the standard captioning metric suite.

    Args:
        use_meteor / use_spice: attempt the JVM-backed metrics.  Automatically
            disabled (and reported as `None`) when no `java` binary is found.
        tokenize: run the PTB tokenizer that the COCO leaderboard uses.  Turn
            it off only if your captions are already tokenized identically to
            the reference implementation -- mismatched tokenization is the most
            common cause of "my BLEU doesn't match the paper".
    """

    use_meteor: bool = True
    use_spice: bool = False  # slow (~minutes) -- opt in for final numbers only
    tokenize: bool = True
    language: str = "en"  # "en" -> PTB tokenizer; "vi" -> Vietnamese segmenter
    #: "rdrsegmenter" | "underthesea" | "pyvi", recorded in warnings.
    #: Default is RDRSegmenter because it is the one that MATCHES: run over
    #: KTVIC's raw captions it reproduces the corpus's own `segment_caption`
    #: field on 100.0% of 2,000 captions, where underthesea manages 82.6%
    #: (`MEASUREMENTS-17-08.md` §11, `DECISIONS.md` ). Segmenting 18% of
    #: captions differently from the corpus shifts every n-gram metric in a
    #: direction nobody can predict, and never raises an error.
    segmenter: str = "rdrsegmenter"
    warnings: list[str] = field(default_factory=list)

    def compute(
        self,
        ground_truth: dict[str, Sequence[str]],
        predictions: dict[str, Sequence[str]],
    ) -> dict[str, float | None]:
        """
        Args:
            ground_truth: {image_id: [ref1, ref2, ...]}  (multiple references)
            predictions:  {image_id: [caption]}          (exactly one hypothesis)

        Returns a flat dict of metric -> score.  Missing metrics are `None`.
        """
        self.warnings = []
        gts, res = self._align(ground_truth, predictions)

        if not gts:
            raise ValueError("no overlapping image ids between references and predictions")
        if len(gts) < 50:
            self.warnings.append(
                f"only {len(gts)} images scored -- CIDEr/SPICE are corpus-level "
                "statistics and are unreliable below a few hundred samples"
            )

        if self.tokenize:
            gts, res = self._tokenize(gts, res)

        scores: dict[str, float | None] = {}
        scores.update(self._bleu(gts, res))
        scores.update(self._rouge(gts, res))
        scores.update(self._cider(gts, res))
        scores.update(self._meteor(gts, res))
        scores.update(self._spice(gts, res))
        scores["num_images"] = len(gts)
        return scores

    # -- individual metrics -------------------------------------------------
    def _bleu(self, gts, res) -> dict[str, float | None]:
        try:
            from pycocoevalcap.bleu.bleu import Bleu

            score, _ = Bleu(4).compute_score(gts, res)
            return {f"BLEU-{i + 1}": float(s) for i, s in enumerate(score)}
        except Exception as exc:
            self.warnings.append(f"BLEU failed: {exc}")
            return {f"BLEU-{i}": None for i in range(1, 5)}

    def _rouge(self, gts, res) -> dict[str, float | None]:
        try:
            from pycocoevalcap.rouge.rouge import Rouge

            score, _ = Rouge().compute_score(gts, res)
            return {"ROUGE-L": float(score)}
        except Exception as exc:
            self.warnings.append(f"ROUGE-L failed: {exc}")
            return {"ROUGE-L": None}

    def _cider(self, gts, res) -> dict[str, float | None]:
        try:
            from pycocoevalcap.cider.cider import Cider

            score, _ = Cider().compute_score(gts, res)
            return {"CIDEr": float(score)}
        except Exception as exc:
            self.warnings.append(f"CIDEr failed: {exc}")
            return {"CIDEr": None}

    def _meteor(self, gts, res) -> dict[str, float | None]:
        if not self.use_meteor:
            return {"METEOR": None}
        if not java_available():
            self.warnings.append(
                "METEOR skipped: no `java` on PATH "
                "(sudo apt-get install -y default-jre)"
            )
            return {"METEOR": None}
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            score, _ = Meteor().compute_score(gts, res)
            return {"METEOR": float(score)}
        except Exception as exc:
            self.warnings.append(f"METEOR failed: {exc}")
            return {"METEOR": None}

    def _spice(self, gts, res) -> dict[str, float | None]:
        if not self.use_spice:
            return {"SPICE": None}
        if not java_available():
            self.warnings.append(
                "SPICE skipped: no `java` on PATH "
                "(sudo apt-get install -y default-jre)"
            )
            return {"SPICE": None}

        home = legacy_java_home()
        if home is None:
            self.warnings.append(
                "SPICE skipped: needs a JDK <= 15 (FST serialisation is blocked by the "
                "Java module system from JDK 16). "
                "Install: sudo apt-get install -y openjdk-11-jre-headless"
            )
            return {"SPICE": None}

        try:
            from pycocoevalcap.spice.spice import Spice

            with _java(home):
                score, _ = Spice().compute_score(gts, res)
            return {"SPICE": float(score)}
        except Exception as exc:
            self.warnings.append(f"SPICE failed (JVM {home}): {exc}")
            return {"SPICE": None}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _align(ground_truth, predictions):
        gts, res = {}, {}
        for key in ground_truth:
            skey = str(key)
            if key not in predictions:
                continue
            refs = ground_truth[key]
            hyp = predictions[key]
            if isinstance(refs, str):
                refs = [refs]
            if isinstance(hyp, str):
                hyp = [hyp]
            gts[skey] = [{"caption": r} for r in refs]
            res[skey] = [{"caption": hyp[0]}]
        return gts, res

    def _tokenize(self, gts, res):
        if self.language == "vi":
            return self._tokenize_vietnamese(gts, res)
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            tokenizer = PTBTokenizer()
            return tokenizer.tokenize(gts), tokenizer.tokenize(res)
        except Exception as exc:
            self.warnings.append(
                f"PTB tokenizer unavailable ({exc}); falling back to lowercase whitespace "
                "split -- scores will NOT be directly comparable to published numbers"
            )
            simple = lambda d: {  # noqa: E731
                k: [c["caption"].lower().strip() for c in v] for k, v in d.items()
            }
            return simple(gts), simple(res)

    def _tokenize_vietnamese(self, gts, res):
        """Word-segment Vietnamese before any n-gram metric.

        Vietnamese writes whitespace between SYLLABLES, not words: `người đàn
        ông` is one word written as three whitespace-separated syllables. Every
        token-based metric (BLEU, ROUGE-L, CIDEr, METEOR) is therefore measuring
        the wrong unit on raw Vietnamese, and the PTB tokenizer -- built for
        English -- makes it worse.

        The chosen segmenter and its version are recorded in `warnings` and MUST
        be reported alongside any score: segmenters disagree, and the choice
        moves the numbers.
        """
        import importlib

        try:
            if self.segmenter == "whitespace":
                # Syllable level: Vietnamese orthography already puts a space
                # between syllables, so this is a real tokenization choice and
                # not a degraded one. It goes through the same path as the rest
                # because skipping `_tokenize` hands the scorers the raw
                # `[{"caption": ...}]` structure instead of strings, and
                # pycocoevalcap then hangs rather than failing.
                segment = lambda t: t  # noqa: E731
                version = "âm tiết / dấu trắng"
            elif self.segmenter == "rdrsegmenter":
                segment, version = _rdrsegmenter()
            elif self.segmenter == "underthesea":
                module = importlib.import_module("underthesea")
                segment = lambda t: module.word_tokenize(t, format="text")  # noqa: E731
            else:
                module = importlib.import_module("pyvi")
                from pyvi import ViTokenizer

                segment = ViTokenizer.tokenize
            # Only the importlib-backed segmenters carry a module to read a
            # version from. `rdrsegmenter` and `whitespace` already set theirs,
            # and reaching for `module` here raised UnboundLocalError -- which
            # the except clause below then reported as "segmenter unavailable"
            # and silently fell back. The whitespace numbers were right by
            # accident, because the fallback IS whitespace.
            if self.segmenter in ("underthesea", "pyvi"):
                version = getattr(module, "__version__", None)
            if self.segmenter in ("underthesea", "pyvi") and version is None:  # pyvi exposes no __version__
                from importlib.metadata import PackageNotFoundError, version as pkg_version

                try:
                    version = pkg_version(self.segmenter)
                except PackageNotFoundError:
                    version = "unknown"
        except Exception as exc:
            self.warnings.append(
                f"Vietnamese segmenter {self.segmenter!r} unavailable ({exc}); "
                "falling back to WHITESPACE split -- scores are NOT valid "
                "Vietnamese n-gram metrics, and are NOT comparable to any "
                "published number. Install with: apt-get install "
                "default-jdk-headless && uv pip install py_vncorenlp"
            )
            # The label must tell the truth: fallback scores must not sit under
            # the name of the segmenter that was requested. A scoring run on a
            # machine without VnCoreNLP once recorded whitespace scores under
            # the name `rdrsegmenter`, and the number slipped into the official
            # table past every `segmenter == "rdrsegmenter"` check.
            self.segmenter = f"{self.segmenter}!whitespace-fallback"
            simple = lambda d: {  # noqa: E731
                k: [c["caption"].lower().strip() for c in v] for k, v in d.items()
            }
            return simple(gts), simple(res)

        self.warnings.append(
            f"Vietnamese word segmentation: {self.segmenter} {version} "
            "-- report this alongside every score (doc 04 §0)"
        )

        def apply(d):
            return {
                k: [segment(c["caption"].strip()).lower() for c in v] for k, v in d.items()
            }

        return apply(gts), apply(res)


def evaluate_captions(
    ground_truth: dict[str, Sequence[str]],
    predictions: dict[str, Sequence[str]],
    use_spice: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: compute the suite and print a readable table."""
    scorer = CaptionMetrics(use_spice=use_spice)
    scores = scorer.compute(ground_truth, predictions)

    if verbose:
        print("\n" + "=" * 46)
        print(f"{'metric':<14}{'score':>12}")
        print("-" * 46)
        for key in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"):
            value = scores.get(key)
            shown = f"{value:.4f}" if isinstance(value, float) else "n/a"
            print(f"{key:<14}{shown:>12}")
        print("-" * 46)
        print(f"{'images':<14}{scores.get('num_images', 0):>12}")
        for warning in scorer.warnings:
            print(f"  ! {warning}")
        print("=" * 46 + "\n")

    return {"scores": scores, "warnings": scorer.warnings}


# ---------------------------------------------------------------------------
# Optional, reference-free complements
# ---------------------------------------------------------------------------