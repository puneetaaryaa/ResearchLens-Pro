from flask import Flask, render_template, request, jsonify, send_file
import fitz  # PyMuPDF
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gunicorn setup ke liye port dynamically uthana padta hai
PORT = int(os.environ.get("PORT", 5000))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/bart-large-cnn"

_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _model is None:
        # CPU low-resource environment ke liye optimized loading
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        if DEVICE == "cpu":
            _model.config.forced_bos_token_id = None # Memory optimizer

def get_targeted_context(full_text, section_type):
    text_lower = full_text.lower()
    anchors = {
        "problem": ["introduction", "motivation", "background", "problem statement"],
        "summary": ["abstract", "executive summary", "overview"],
        "gaps": ["limitations", "research gap", "future work", "challenges"],
        "scope": ["conclusion", "future directions", "future work"]
    }
    selected = anchors.get(section_type, [])
    start_pos = 0
    for kw in selected:
        match = re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        if match:
            start_pos = match.start()
            break
    return full_text[start_pos : start_pos + 3500]

def extract_metrics(text):
    patterns = [
        r'\d+\.?\d*\%', 
        r'(accuracy|f1-score|precision|recall|auc|map)\s*[:=]?\s*\d?\.?\d+',
        r'p\s*[<=]\s*\d?\.?\d+',
        r'(dataset|benchmark)\s*:\s*[\w\d\s-]+'
    ]
    found = []
    for p in patterns:
        matches = re.finditer(p, text, re.IGNORECASE)
        for m in matches:
            found.append(m.group().upper())
    return list(set(found))[:8]

def neural_extract(text, section):
    _load_model()
    context = get_targeted_context(text, section)
    inputs = _tokenizer(context, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    with torch.no_grad():
        ids = _model.generate(**inputs, max_new_tokens=150, min_new_tokens=60, num_beams=3, no_repeat_ngram_size=3) # Beams 5 se 3 kiye for RAM safety
    return _tokenizer.decode(ids[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'pdf' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['pdf']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    try:
        doc = fitz.open(path)
        full_text = " ".join(p.get_text() for p in doc)
        meta = doc.metadata
        title = meta.get('title') or file.filename
        author = meta.get('author') or "Author Not Found"
        doc.close()

        cite_key = re.sub(r'\W+', '', author.split()[0].lower()) if author != "Author Not Found" else "research"
        citation = f"@article{{{cite_key}2026,\n  author = {{{author}}},\n  title = {{{title}}},\n  journal = {{ResearchLens AI Analysis}},\n  year = {{2026}}\n}}"

        results = {
            "summary": neural_extract(full_text, "summary"),
            "problem": neural_extract(full_text, "problem"),
            "gaps": neural_extract(full_text, "gaps"),
            "scope": neural_extract(full_text, "scope"),
            "metrics": extract_metrics(full_text),
            "citation": citation,
            "filename": file.filename
        }
        app.config['LATEST'] = results
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)

@app.route("/export")
def export():
    data = app.config.get('LATEST')
    if not data: return "No data", 400
    report = "Research_Insight_Pro.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"--- RESEARCH ANALYSIS REPORT ---\n\nFILE: {data['filename']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"KEY FINDINGS: {', '.join(data['metrics'])}\n\n")
        f.write(f"1. PROBLEM:\n{data['problem']}\n\n")
        f.write(f"2. SUMMARY:\n{data['summary']}\n\n")
        f.write(f"3. RESEARCH GAPS:\n{data['gaps']}\n\n")
        f.write(f"4. FUTURE SCOPE:\n{data['scope']}\n\n")
        f.write(f"CITATION (BibTeX):\n{data['citation']}")
    return send_file(report, as_attachment=True)

if __name__ == "__main__":
    # Local run karne ke liye default port 5000 rahega
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
