# Player tracking and individual performance analytics

## Motivation

Tek sabit kameradan kaydedilen saha videosu üzerinden oyuncu takibi ve analizi çalışması.

## Objectives

- Oyuncu tespiti ve takibi
- Metre cinsinden konum tahmini
- Mesafe, hız ve heatmap çıktıları

## Dataset

Adaylar: SoccerTrack v2, TeamTrack, Alfheim, Sosyal Halısaha kayıtları.

Bence gerekli bazı şartlar:

- 20-25 fps arası görüntü
- Sabit kamera **ŞART!**
- Çözünürlük değişkenlik gösterebilir
- Panoramik olursa parça parça işlenebilir

Kaynaklar hakkında notlar:

- Halısaha videoları kalite açısından kötü; çok fazla afiş, poster ve pano var.
- En iyi kalite SoccerTrack v2'de var (4K panoramik, full-pitch maçlar), ama
  panoramik olduğu için tek videoda uzaklarda ve köşelerde detection belki
  sıkıntı çıkarır.
- Alfheim dataseti amaca uygun ama eski ve kalitesi yetersiz olabilir.
- TeamTrack'te fisheye + drone (top-view) görüntüleri mevcut.
- YouTube'da da birkaç video var ama kameralar sabit değil.

## Video bilgisi (118575, 1. yarı)

- Çözünürlük: 4096 × 1080 (panoramik), 25 fps
- Süre: 2874 s (47:54), 71850 kare
- Çalışma klibi: ilk 60 s → 1500 kare (`data/processed/test_60s.mp4`)
- Gözlem: kenarlarda belirgin panoramik bükülme; uzak taraftaki oyuncular çok küçük.

## Araçlar

- Python
- Ultralytics YOLO11
- OpenCV
- Grafik ve analiz çıktıları için pandas, matplotlib

## Sınırlamalar

- Panoramik kayıtta köşe bükülmeleri.
- Korner gibi kalabalık durumlarda tek kameradan oyuncuların birbirini kapatması.
- Kameranın sabit olması bu senaryoda daha çok işimize yarıyor.

Proje sürecinde elde edilen tüm gerçek ve hatalı veriler kayıt edilip projenin
sınırları sayısal olarak belirlenecektir.
