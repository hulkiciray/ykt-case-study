# Finansal Tablo ve Dipnot İlişkilendirme

Rapor taranmış bir PDF olduğundan dolayı OCR ile başladım.

## Çalıştırma

```bash
./run.sh
```

Virtual environment'ı kurup ipeline'ı çalıştırmak için yukarıdaki komutu çalıştırabilirsiniz.
OCR sonuçlarıyla LLM cevapları `cache/` içinde olduğu için API anahtarını paylaşmadım. Eğer bunları tekrar sıfırdan başlatmasını isterseniz `./run.sh --fresh` (LLM adımı için `OPENAI_API_KEY` gerekir).

Case study'de belirtildiği gibi sayfaları ve dipnot numarasını `config.yaml`'dan gelecek şekilde düzenledim. Sonuç `outputs/final_output.json` dosyasına kaydediliyor.

## Yaklaşım

| Aşama | Dosya | Ne yapıyor |
|---|---|---|
| 1 | `table_extraction.py` | EasyOCR ile okuma, sütun ve dönem tespiti, sayı ayrıştırma, ana kalem / alt kalem / toplam ilişkisi |
| 2 | `footnote_page_finder.py` | Dipnotun sayfalarını içindekiler sayfası ve dipnot başlığıyla bulma |
| 3 | `note_rows.py` | Dipnot tablolarındaki satırları çıkarma |
| 4-5 | `linking.py` | Aday üretimi, iki yöntemle eşleştirme |
| 6 | `validation.py` | Kontroller, nihai confidence, sonuç dosyası |

Özet olarak, düzenin kurallı olduğu yerde kural, anlam gereken yerde model kullandım.

Bazı notlar:
- Sayılar ödevdeki kurallarla ayrıştırılır (parantez negatif, nokta binlik ayırıcı, tire / boş / sıfır ayrı). OCR tire işaretini okumadığı için boş hücreleri pikselden kontrol ettim.
- Hiyerarşiyi aritmetikle bulmak istedim çünkü bilançoda ana kalem önce gelir, gelir tablosunda ise toplam parçalarından sonra gelir.
- Dipnotlar standart bir düzene sahip olmadığı için dipnot tablolarını yeniden kurmak istemedim, sadece satırları OCR ile çıkardım.

## Modeller

**EasyOCR (Türkçe):** sayfa 300 dpi'da okunuyor. Temiz ayrışmayan tutarlar sadece rakamlara izin verilerek ikinci kez okunuyo.

**Yöntem A, `paraphrase-multilingual-MiniLM-L12-v2`:**
- Skor = 0.5 etiket benzerliği + 0.3 tutar eşleşmesi + 0.2 dönem eşleşmesi.
- Dipnot tarafında metin "dipnot başlığı - satır etiketi - sütun başlığı" olarak verilir.
- Eşik (0.47) 16 elle etiketlenmiş kontrol örneğinden (`data/control_links.json`) seçildi.

**Yöntem B, OpenAI `gpt-4.1-mini` (LLM judge):**
- Her kalem için tek çağrı yapılıyor.
- Model her aday için ilişki tipi, confidence ve kısa gerekçe döner (strict JSON, temperature 0 çünkü biraz daha reproducable olmasını istedim).
- Cevaplar cache'lenir. API hata verirse o kalem için yöntem A'yı kullanıyor.

## Confidence

Model skoru tek başına kullanılmadı.
- **Değer:** OCR güveni + format düzeltmesi gerekti mi + toplam kontrolünü geçiyor mu.
- **İlişki:** 0.35 model skoru + 0.25 tutar uyumu + 0.15 dönem uyumu + 0.15 iki yöntemin uyumu + 0.10 OCR güveni.

0.70 altı `low_confidence`, tutarı ya da dönemi tutmayan ilişkiler `rejected` olarak işaretlenir.

Bunları aslında biraz da intuitional olarak belirledim.

## Sonuçlar

- Özet tablolar: 152 değer, 40/40 toplam kontrolü, aktif = pasif (iki yıl için).
- Dipnot 11: PDF 53-54. Sayfa bulma 31 dipnotun hepsinde doğru sayfaları veriyor.
- Dipnot satırları: 19 satır, 11/11 toplam kontrolü.
- Doğrulama: 12/12 kontrol geçti (yapısal, format, finansal). 16 ilişki kabul edildi.

| Yöntem | Precision | Recall |
|---|---|---|
| A | 1.00 | 1.00 |
| B | 1.00 | 0.75 |

## Hata analizi

1. **LLM'in yanlış ilişkisi (ilk deneme):** 151.350.000'lik değerleme farkını 423.580.000'lik kapanış bakiyesine 0.95 güvenle bağladı. Neden: prompt kalemin bakiye mi akış mı olduğunu söylemiyordu. Doğrulamadaki tutar kontrolü bunu reddetti, promptu revize ettim.
2. **B'nin 2012 dağılımını atlaması:** model gerçeğe uygun değer ile net defter değerini farklı saydı. İyileştirme: dipnotun toplam yapısını prompt'a vermek.
3. **OCR hataları:** dipnot numarasını `11` `Il` olarak okudu. Harf → rakam dönüşümüyle düzeltildi. `7,.932.336` doğru ayrıştı ama düşük OCR güveni yüzünden düşük confidence ile işaretleniyor.

## Sınırlamalar

- Yatay basılmış sayfalar (mesela dipnot 12) döndürülmüyor, bu durumda sayfalar sadece içindekilerden bulunuyor.
- Çok sütunlu geniş tabloları (mesela özkaynak değişim tablosu) test etmedim.
