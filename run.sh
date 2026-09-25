#!/usr/bin/env bash
# Kullanım:
#   ./run.sh           kurulum + testler + pipeline (cachedeki OCR ve LLM cevaplarıyla. Yaklaşık olarak 10 saniye civarında sürüyor)
#   ./run.sh --fresh   cache'i silip her şeyi baştan hesaplar (OCR yaklaşık olarak 5 dk, LLM için OPENAI_API_KEY)
set -e
cd "$(dirname "$0")"

if [ "$1" == "--fresh" ]; then
    echo "Cache siliniyor, OCR ve LLM çağrıları baştan yapılacak."
    rm -rf cache/ocr cache/llm
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Uyarı: OPENAI_API_KEY yok. Yöntem B (LLM) yerine yöntem A sonuçları kullanılacak."
    fi
fi

if [ ! -d .venv ]; then
    echo "venv kuruluyor"
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

echo
echo "Test"
.venv/bin/python test_numbers.py

echo
echo "Pipeline"
.venv/bin/python main.py config.yaml

echo
echo "Çıktılar: outputs/  (sonuç: outputs/final_output.json)"
