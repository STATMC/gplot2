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
