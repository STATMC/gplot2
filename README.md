# a = pd.DataFrame(list(newsCollection.find()))0
# duplicateList = a[a.duplicated(subset = ["headline", "date","source"], keep = False)]["_id"].tolist()
# newsCollection.delete_many({'_id':{'$in':duplicateList}})
# a = pd.DataFrame(list(newsCollection.find({"check" : False})))
# a["date"] = pd.to_datetime(a["date"])
# a = a.drop_duplicates(subset = ["headline"])
# a.date = a.date.apply(lambda x : np.nan if x == "" else x )
# a.fillna(method = "ffill", inplace = True)
# a.reset_index(inplace = True, drop = True)
# #a["date"] = a["date"].apply(lambda x:  x[:10] if type(x) != type(datetime.datetime(2022,11,11)) else x)
# #a["date"] = a["date"].apply(lambda x:  datetime.datetime.strptime(x, '%Y-%m-%d') if type(x) != type(datetime.datetime(2022,11,11)) else x)
# a["date"] = a["date"].apply(lambda x: x.replace(hour = 0, minute = 0, second = 0, microsecond = 0))
# #a = a[a["date"] >= datetime.datetime(2023,8,19)]
# a = a.reset_index().drop("index", axis = 1)

# a = a.iloc[-1000:]
# array = np.array(a.headline)
# print(len(a))
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", proxies = proxies)
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", proxies = proxies)
# def historicalSentiment(df):
#     tempTickers = []
#     tempPosList = []
#     tempNegList = []
#     tempNeuList = []
#     sentence_count = 0
#     ticker_count = 0
#     for i in range(len(df)):
#         sentenceList = split_into_sentences(df.alltext[i])
#         for sentence in sentenceList:
#             sentence_count = sentence_count + 1 
#             for ticker in usTickers:
#                 if ticker in sentence:
#                     ticker_count = ticker_count +1 
#                     tempTickers.append(ticker)
#                     sentiment = calculateSentiment(sentence)
#                     tempPosList.append(sentiment[0])
#                     tempNegList.append(sentiment[1])
#                     tempNeuList.append(sentiment[2])
#     resultDf = pd.DataFrame(list(zip(tempTickers,tempPosList,tempNegList,tempNeuList)), columns = ["ticker", "positive", "negative", "neutral"])
#     resultDf["positive"] = resultDf["positive"]*100
#     resultDf["negative"] = resultDf["negative"]*100
#     resultDf["neutral"] = resultDf["neutral"]*100
#     resultDf["date"] = df.date[i]
#     resultDf = resultDf.groupby(["ticker"]).mean()
#     return resultDf













########################












import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Veri çerçevenizin yüklenmesi (varsayım)
newsCollection = pd.read_csv('your_dataframe.csv')

# Duplicate haber başlıklarını sil
newsCollection["date"] = pd.to_datetime(newsCollection["date"])
newsCollection = newsCollection.drop_duplicates(subset=["headline", "date", "source"])

# Tarih işlemleri
newsCollection["date"] = newsCollection["date"].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))

# Son 1000 kaydı al
newsCollection = newsCollection.iloc[-1000:]

# Tokenizer ve model yükleme
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Duygu analizi fonksiyonu
def historicalSentiment(df):
    tempTickers = []
    tempPosList = []
    tempNegList = []
    tempNeuList = []
    for index, row in df.iterrows():
        sentenceList = split_into_sentences(row['alltext'])  # Cümlelere ayırma fonksiyonu gerekebilir
        for sentence in sentenceList:
            for ticker in usTickers:  # usTickers listesi tanımlanmalı
                if ticker in sentence:
                    sentiment = calculateSentiment(sentence)  # Sentiment hesaplama fonksiyonu gerekebilir
                    tempTickers.append(ticker)
                    tempPosList.append(sentiment[0])
                    tempNegList.append(sentiment[1])
                    tempNeuList.append(sentiment[2])
    resultDf = pd.DataFrame(list(zip(tempTickers, tempPosList, tempNegList, tempNeuList)), columns=["ticker", "positive", "negative", "neutral"])
    resultDf["date"] = df.date.iloc[-1]  # Son tarihi kullan
    resultDf = resultDf.groupby(["ticker"]).mean().reset_index()
    return resultDf

# Duygu analizi uygulama
sentiment_results = historicalSentiment(newsCollection)
print(sentiment_results)



########################################

import pandas as pd

# Veri çerçevenizin yüklenmesi (varsayım)
newsCollection = pd.read_csv('your_dataframe.csv')

# Farklı kaynaklara göre tarih formatlama
def custom_date_parser(date_str, source):
    if source == 'first':
        # Format: 'Jul 9, 2024 at 4:51AM'
        return pd.to_datetime(date_str, format='%b %d, %Y at %I:%M%p', errors='coerce')
    elif source == 'second':
        # Format: 'Tue, Jul 9, 2024, 4:51AM'
        return pd.to_datetime(date_str, format='%a, %b %d, %Y, %I:%M%p', errors='coerce')
    elif source == 'third':
        # Format: 'July 9, 2024 4:51AM'
        return pd.to_datetime(date_str, format='%B %d, %Y %I:%M%p', errors='coerce')
    else:
        return pd.NaT

# Tarih sütununu özel parser ile dönüştürme
newsCollection['parsed_date'] = newsCollection.apply(lambda x: custom_date_parser(x['date'], x['source']), axis=1)

# Dönüşüm sonrası kontrol
print(newsCollection[['date', 'source', 'parsed_date']].head())

# Hatalı dönüşümleri kontrol et
print("Entries with NaT in 'parsed_date':", newsCollection['parsed_date'].isna().sum())

#---------------------------------------------------------------------------------------------------#


a = pd.DataFrame(list(newsCollection.find()))
duplicateList = a[a.duplicated(subset = ["headline", "date","source"], keep = False)]["_id"].tolist()
newsCollection.delete_many({'_id':{'$in':duplicateList}})
a = pd.DataFrame(list(newsCollection.find({"check" : False})))
a["date"] = pd.to_datetime(a["date"])
a = a.drop_duplicates(subset = ["headline"])
a.date = a.date.apply(lambda x : np.nan if x == "" else x )
a.fillna(method = "ffill", inplace = True)
a.reset_index(inplace = True, drop = True)
#a["date"] = a["date"].apply(lambda x:  x[:10] if type(x) != type(datetime.datetime(2022,11,11)) else x)
#a["date"] = a["date"].apply(lambda x:  datetime.datetime.strptime(x, '%Y-%m-%d') if type(x) != type(datetime.datetime(2022,11,11)) else x)
a["date"] = a["date"].apply(lambda x: x.replace(hour = 0, minute = 0, second = 0, microsecond = 0))
#a = a[a["date"] >= datetime.datetime(2023,8,19)]
a = a.reset_index().drop("index", axis = 1)

a = a.iloc[-50:]
array = np.array(a.headline)
print(len(a))
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", proxies = proxies)
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", proxies = proxies)
def historicalSentiment(df):
    tempTickers = []
    tempPosList = []
    tempNegList = []
    tempNeuList = []
    sentence_count = 0
    ticker_count = 0
    for i in range(len(df)):
        sentenceList = split_into_sentences(df.alltext[i])
        for sentence in sentenceList:
            sentence_count = sentence_count + 1 
            for ticker in usTickers:
                if ticker in sentence:
                    ticker_count = ticker_count +1 
                    tempTickers.append(ticker)
                    sentiment = calculateSentiment(sentence)
                    tempPosList.append(sentiment[0])
                    tempNegList.append(sentiment[1])
                    tempNeuList.append(sentiment[2])
    resultDf = pd.DataFrame(list(zip(tempTickers,tempPosList,tempNegList,tempNeuList)), columns = ["ticker", "positive", "negative", "neutral"])
    resultDf["positive"] = resultDf["positive"]*100
    resultDf["negative"] = resultDf["negative"]*100
    resultDf["neutral"] = resultDf["neutral"]*100
    resultDf["date"] = df.date[i]
    resultDf = resultDf.groupby(["ticker"]).mean()
    return resultDf

df_list = []
for date in a.date.drop_duplicates():
    print(date)
    tempDf = a[a.date == date].copy()
    tempDf.reset_index(inplace = True)
    tempDf.drop("index", axis = 1, inplace = True)
    resultDf = historicalSentiment(tempDf)
    resultDf["date"] = date
    df_list.append(resultDf)












