# BERTopic + Streamlit (Minimal Visual Demo)
- Upload CSV → pilih kolom teks (+ opsional tanggal)
- Jalankan BERTopic → tampilkan visual (topics, barchart, topics over time)
- Unduh hasil topic_info.csv

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Jika tidak mengupload CSV, app akan memuat `data/sample.csv`.
