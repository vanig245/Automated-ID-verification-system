# VeriDoc — Automated KYC ID Verification

> Instantly extract name, Aadhaar number, and date of birth from a scanned ID card using computer vision and OCR — no manual data entry, no third-party APIs, fully local.

---

## The Problem

Banks, telecom companies, and fintech apps spend millions of dollars and countless man-hours on **manual KYC (Know Your Customer)** verification. An agent physically looks at a user's ID, types the details into a system, and checks for inconsistencies. This process is:

- **Slow** — takes minutes to hours per user
- **Error-prone** — manual transcription mistakes are common
- **Expensive** — requires dedicated verification staff
- **Unscalable** — bottleneck during high user onboarding periods

Fraud is also rampant: forged IDs, edited PDFs, and photo-manipulated cards slip through human review regularly.

---

## The Solution

VeriDoc is a **local, privacy-first KYC pipeline** that automates ID verification in under a second:

1. User uploads a photo of their Aadhaar card via a web interface
2. OpenCV detects and perspective-corrects the card boundary
3. Tesseract OCR extracts all text from the preprocessed image
4. spaCy NER + regex patterns identify name, Aadhaar number, and DOB
5. Results are returned as a structured JSON profile — no data leaves the machine

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI (Python) | REST API server |
| Computer Vision | OpenCV | Card detection, perspective correction, preprocessing |
| OCR Engine | Tesseract + pytesseract | Text extraction from ID image |
| NLP | spaCy (`en_core_web_sm`) | Named Entity Recognition for person names |
| Frontend | Vanilla HTML/CSS/JS | Drag-and-drop upload UI |
| Serving | Uvicorn | ASGI server for FastAPI |

---

## Project Structure

```
automated-id-verification/
│
├── main.py               # FastAPI app — routes, validation, response models
├── process.py            # Core pipeline — OpenCV + OCR + NLP logic
│
├── templates/
│   └── index.html        # Frontend UI (drag-and-drop, results display)
│
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

Make sure the following are installed on your system:

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/vanig245/automated-id-verification.git
cd automated-id-verification

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Start the server
uvicorn main:app --reload
```

Open your browser at **http://localhost:8000**

---

## Requirements

```
fastapi
uvicorn
opencv-python
pytesseract
numpy
spacy
python-multipart
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

*Built with OpenCV · Tesseract · spaCy · FastAPI*