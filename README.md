tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Duygu analizi pipeline'ı oluşturma
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Duygu analizini uygulama
df['Sentiment'] = df['Cleaned_Text'].apply(lambda x: nlp(x)[0]['label'])

# Duygu dağılımını görselleştirme
sns.countplot(x='Sentiment', data=df)
plt.title('Duygu Dağılımı')
plt.show()
