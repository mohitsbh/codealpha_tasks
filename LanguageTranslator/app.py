"""
Flask web UI for AI Language Translator (deep-translator)
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES

app = Flask(__name__)

# Prepare language lists (name -> code)
LANGUAGE_ITEMS = sorted(GOOGLE_LANGUAGES_TO_CODES.items(), key=lambda x: x[0])
NAME_BY_CODE = {code: name for name, code in LANGUAGE_ITEMS}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", languages=LANGUAGE_ITEMS)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    source = (data.get("source") or "auto").strip() or "auto"
    target = (data.get("target") or "en").strip() or "en"

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        translator = GoogleTranslator(source=source, target=target)
        translated = translator.translate(text)
        return jsonify(
            {
                "text": translated,
                "source": source,
                "target": target,
                "source_name": NAME_BY_CODE.get(source, source),
                "target_name": NAME_BY_CODE.get(target, target),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
