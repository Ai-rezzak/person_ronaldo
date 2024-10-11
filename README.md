# İnsan ve Ronaldo Tanıma Projesi
Bu proje, ResNet modelini kullanarak insan yüzlerini ve özel olarak Cristiano Ronaldo'yu tanımak için geliştirilmiştir. Kaggle'da 933 insan ve 933 Ronaldo resmi ile 25 epoch süresince eğitilmiştir.

## Açıklama
Bu projede, ResNet mimarisi kullanılarak yüz tanıma işlemi gerçekleştirilmiştir. Proje, iki ana Python dosyası içerir:
- **test_with_image.py**: Belirli bir resim üzerinde yüz tanıma işlemi yapar.
- **found_face.py**: Webcam üzerinden gerçek zamanlı yüz tespiti yapar.

## Gereksinimler
Projenin çalışabilmesi için aşağıdaki kütüphanelerin kurulu olması gerekmektedir:
- TensorFlow veya PyTorch (kullanılan kütüphaneye bağlı olarak)
- OpenCV
- NumPy
- Matplotlib (isteğe bağlı, görselleştirme için)

Bu bağımlılıkları yüklemek için aşağıdaki komutu kullanabilirsiniz:
```bash
pip install -r requirements.txt
