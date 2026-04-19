"""
Invoice information extraction.

Runs Tesseract OCR on an invoice image, then applies regex patterns to
extract structured fields. Designed to be called after classification
confirms the document is an invoice.

CLI usage:
  python scripts/extract.py --image path/to/invoice.png

Returns JSON with:
  invoice_number, invoice_date, due_date,
  issuer_name, recipient_name, total_amount
Missing fields are null.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(img: np.ndarray) -> str:
    """Run Tesseract on a grayscale uint8 numpy array."""
    import pytesseract
    import cv2

    # Light preprocessing: deskew + threshold to improve OCR accuracy
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, config="--psm 6")
    return text


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # 01/12/2024  or  01-12-2024
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    # 2024-01-12
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
    # January 12, 2024  or  12 January 2024
    r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{2,4})\b",
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{2,4})\b",
]

_DATE_RE = re.compile(
    "|".join(_DATE_PATTERNS),
    re.IGNORECASE,
)


def find_dates(text: str) -> list[str]:
    matches = []
    for m in _DATE_RE.finditer(text):
        val = next(g for g in m.groups() if g is not None)
        matches.append(val.strip())
    return matches


def extract_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r"(?:invoice|inv|invoice\s*no\.?|inv\.?\s*no\.?|invoice\s*#|inv\s*#)\s*[:#]?\s*([A-Z0-9][\w\-\/]{1,20})",
        r"(?:bill\s*(?:no|number|#))[:\s]+([A-Z0-9][\w\-\/]{1,20})",
        r"#\s*([A-Z0-9]{4,20})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_invoice_date(text: str) -> Optional[str]:
    """Find the date nearest to 'invoice date' / 'date' keywords."""
    # Prefer labeled date
    m = re.search(
        r"(?:invoice\s*date|date\s*of\s*invoice|date\s*issued?)[:\s]+(.{1,30})",
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip()
        dates = find_dates(candidate)
        if dates:
            return dates[0]

    # Fall back: first date in document
    dates = find_dates(text)
    return dates[0] if dates else None


def extract_due_date(text: str) -> Optional[str]:
    m = re.search(
        r"(?:due\s*(?:date|on|by)|payment\s*due)[:\s]+(.{1,40})",
        text, re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        dates = find_dates(candidate)
        if dates:
            return dates[0]
        # plain text fallback e.g. "Due: 30 days"
        return candidate[:30]

    # Second date in document often the due date
    dates = find_dates(text)
    return dates[1] if len(dates) > 1 else None


def extract_total_amount(text: str) -> Optional[str]:
    patterns = [
        r"(?:grand\s*total|total\s*(?:amount\s*)?due|total\s*payable|amount\s*due|total)[:\s]+([£$€¥]?\s*[\d,]+(?:\.\d{2})?)",
        r"([£$€¥]\s*[\d,]+\.\d{2})\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def extract_names(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Heuristic name extraction using spaCy NER.
    Falls back to first ORG/PERSON entity for issuer, last for recipient.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:2000])  # process first 2000 chars only
        entities = [(ent.label_, ent.text.strip()) for ent in doc.ents
                    if ent.label_ in ("ORG", "PERSON")]
        orgs    = [t for l, t in entities if l == "ORG"]
        persons = [t for l, t in entities if l == "PERSON"]

        issuer    = orgs[0]    if orgs    else (persons[0]    if persons    else None)
        recipient = orgs[1]    if len(orgs) > 1 else (persons[1] if len(persons) > 1 else None)
        return issuer, recipient
    except Exception:
        pass

    # Fallback: look for labeled lines
    issuer = recipient = None
    for pat, key in [
        (r"(?:from|bill\s*from|issued?\s*by)[:\s]+(.+)", "issuer"),
        (r"(?:to|bill\s*to|sold\s*to|client)[:\s]+(.+)",  "recipient"),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()[:60]
            if key == "issuer":
                issuer = val
            else:
                recipient = val

    return issuer, recipient


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def _get_word_boxes(img: np.ndarray) -> list[dict]:
    """Return list of {text, x, y, w, h} for each OCR word."""
    import pytesseract
    import cv2
    import pandas as pd

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(binary, config="--psm 6", output_type=pytesseract.Output.DATAFRAME)
    data = data[data["conf"] > 0]
    boxes = []
    for _, row in data.iterrows():
        txt = str(row["text"]).strip()
        if txt:
            boxes.append({
                "text": txt,
                "x": int(row["left"]),
                "y": int(row["top"]),
                "w": int(row["width"]),
                "h": int(row["height"]),
            })
    return boxes


def find_field_bboxes(img: np.ndarray, fields: dict) -> list[dict]:
    """Find bounding boxes for extracted field values on the image."""
    word_boxes = _get_word_boxes(img)
    if not word_boxes:
        return []

    bboxes = []
    for field_name, value in fields.items():
        if not value:
            continue
        value_lower = str(value).lower()
        value_words = value_lower.split()
        if not value_words:
            continue

        # Find consecutive words that match the field value
        for i, wb in enumerate(word_boxes):
            if wb["text"].lower().startswith(value_words[0][:4]):
                # Try to match the full value across consecutive words
                end = min(i + len(value_words) + 2, len(word_boxes))
                span = word_boxes[i:end]
                span_text = " ".join(w["text"] for w in span).lower()
                if value_words[0][:4] in span_text:
                    x = min(w["x"] for w in span)
                    y = min(w["y"] for w in span)
                    x2 = max(w["x"] + w["w"] for w in span)
                    y2 = max(w["y"] + w["h"] for w in span)
                    bboxes.append({
                        "field": field_name,
                        "x": x, "y": y,
                        "w": x2 - x, "h": y2 - y,
                    })
                    break
    return bboxes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_invoice_info(img: np.ndarray, pdf_text: Optional[str] = None) -> dict:
    # Use embedded PDF text when available (cleaner than OCR)
    if pdf_text and len(pdf_text.strip()) > 30:
        text = pdf_text
    else:
        text = ocr_image(img)
    issuer, recipient = extract_names(text)
    return {
        "invoice_number":  extract_invoice_number(text),
        "invoice_date":    extract_invoice_date(text),
        "due_date":        extract_due_date(text),
        "issuer_name":     issuer,
        "recipient_name":  recipient,
        "total_amount":    extract_total_amount(text),
    }


def extract_invoice_from_path(image_path: str) -> dict:
    """Convenience wrapper: load image and extract."""
    from PIL import Image
    p = Path(image_path)
    pdf_text = None
    if p.suffix.lower() == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            pil = pdf.pages[0].to_image(resolution=150).original.convert("L")
            pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        pil = Image.open(p).convert("L")
    img = np.array(pil, dtype=np.uint8)
    return extract_invoice_info(img, pdf_text=pdf_text)


def main():
    parser = argparse.ArgumentParser(description="Extract invoice fields from an image")
    parser.add_argument("--image", required=True, help="Path to invoice image or PDF")
    parser.add_argument("--pretty", action="store_true", default=True)
    args = parser.parse_args()

    result = extract_invoice_from_path(args.image)
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
