"""Insert APA-style in-text citations into the Capturing-Science-Talk docx.

The .docx was already given a `References` section by
`scripts/append_references.py`. This script adds the matching parenthetical
in-text citations to the body so each reference is actually cited somewhere.

We do this by string-replacing inside `word/document.xml`. Each anchor phrase
below is unique to a single <w:t> element (verified before writing this
script), so we are not at risk of double-substituting. The two comment ranges
(Whisper, Jameel & Dungen) are left untouched: every insertion is either in a
different paragraph or in a run that sits after `commentRangeEnd`.

Run from the project root:
    python scripts/add_intext_citations.py
"""

from pathlib import Path
import shutil
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / "Captureing Science Talk in Musuem_5.12.26+DI.docx"

# (anchor_text, replacement_text). The anchor must occur exactly once in the
# document.xml stream; the script asserts this. `&amp;` is the XML-encoded
# form of `&` (required for inline ampersands inside Word's XML).
REPLACEMENTS: list[tuple[str, str]] = [
    # Overview: justify the parent-child museum framing. The anchor sits
    # inside an existing `(e.g., ...)` parenthetical, so we fold the citation
    # into the outer parens with a semicolon instead of nesting `(...)` inside
    # `(...)`. APA-style convention for nested-parens citations.
    (
        "they offer each other",
        "they offer each other; Haden et al., 2014",
    ),
    # Audio capture: LENA recorder.
    (
        "Audio system [LENA])",
        "Audio system [LENA]) (Gilkerson et al., 2017)",
    ),
    # Audio capture: Reel beacons (this run sits AFTER commentRangeEnd for
    # comment id=0, so we don't touch the comment anchor).
    (
        "to track when individuals",
        "(Jameel &amp; Dungen, 2015) to track when individuals",
    ),
    # ASR section: Whisper-large-v3 paper.
    (
        "Whisper-large-v3, OpenAI&apos;s open-source ASR model.",
        "Whisper-large-v3 (Radford et al., 2022), OpenAI&apos;s open-source ASR model.",
    ),
    # Detector section: bi-encoder retrieval architecture.
    (
        "fine-tuned bi-encoder",
        "fine-tuned bi-encoder (Reimers &amp; Gurevych, 2019)",
    ),
    # Training section, sub-type labeling: LLM zero-shot fallback.
    (
        "LLM zero-shot fallback",
        "LLM zero-shot fallback (Grattafiori et al., 2024)",
    ),
    # Training section, negative mining: hard negatives via LLM (DPR).
    (
        "LLM-generated hard negatives",
        "LLM-generated hard negatives (Karpukhin et al., 2020)",
    ),
]

DOC_XML_NAME = "word/document.xml"


def _patch_document_xml(xml: str) -> str:
    for anchor, replacement in REPLACEMENTS:
        count = xml.count(anchor)
        assert count == 1, (
            f"Expected exactly one occurrence of {anchor!r} in document.xml, "
            f"found {count}. Aborting to avoid an ambiguous edit."
        )
        xml = xml.replace(anchor, replacement, 1)
    return xml


def add_citations(doc_path: Path = DOC_PATH) -> Path:
    tmp_path = doc_path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(doc_path, "r") as zin:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == DOC_XML_NAME:
                    xml = data.decode("utf-8")
                    xml = _patch_document_xml(xml)
                    data = xml.encode("utf-8")
                zout.writestr(item, data)
    shutil.move(tmp_path, doc_path)
    return doc_path


if __name__ == "__main__":
    out = add_citations()
    print(f"Updated {out}")
