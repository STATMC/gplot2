import pandas as pd

# Dosyayı yükleme
df = pd.read_csv('/path/to/your/file.csv')

# Eksik değerleri kontrol etme
print(df['haber_paragraflar'].isnull().sum())

# Eksik değerleri doldurma
df.loc[df['haber_paragraflar'].isnull(), 'haber_paragraflar'] = ''

# Metin temizleme fonksiyonu
def preprocess_text(text):
    text = str(text).strip()  # Metni stringe çevirip boşlukları temizleme
    return text

# Temizlenmiş metin sütunu ekleme
df.loc[:, 'Cleaned_Text'] = df['haber_paragraflar'].apply(preprocess_text)







from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# FinBERT modelini yükleme
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Duygu analizi pipeline'ı oluşturma
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Duygu analizini uygulama
df.loc[:, 'Sentiment'] = df['Cleaned_Text'].apply(lambda x: nlp(x)[0]['label'])

# Duygu dağılımını görselleştirme
sns.countplot(x='Sentiment', data=df)
plt.title('Duygu Dağılımı')
plt.show()











#tam


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Dosyayı yükleme
df = pd.read_csv('/path/to/your/file.csv')

# Eksik değerleri kontrol etme ve doldurma
df.loc[df['haber_paragraflar'].isnull(), 'haber_paragraflar'] = ''

# Metin temizleme fonksiyonu
def preprocess_text(text):
    text = str(text).strip()  # Metni stringe çevirip boşlukları temizleme
    return text

# Temizlenmiş metin sütunu ekleme
df.loc[:, 'Cleaned_Text'] = df['haber_paragraflar'].apply(preprocess_text)

# FinBERT modelini yükleme
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Duygu analizi pipeline'ı oluşturma
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Duygu analizini uygulama
df.loc[:, 'Sentiment'] = df['Cleaned_Text'].apply(lambda x: nlp(x)[0]['label'])

# Duygu dağılımını görselleştirme
sns.countplot(x='Sentiment', data=df)
plt.title('Duygu Dağılımı')
plt.show()
