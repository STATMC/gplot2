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


#####################################################################################


# Duygu analizi için yardımcı fonksiyonlar
def calculateSentiment(sentence):
    # Modelinizi ve tokenizer'ınızı kullanarak duygu analizi yapan fonksiyon
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    return probs  # [pozitif, negatif, nötr] varsayımıyla

def split_into_sentences(text):
    # Metni cümlelere ayıran fonksiyon
    import nltk
    nltk.download('punkt')
    return nltk.tokenize.sent_tokenize(text)

def historicalSentiment(newsCollection, usTickers):
    tempTickers = []
    tempPosList = []
    tempNegList = []
    tempNeuList = []
    dates = []
    
    for i, row in newsCollection.iterrows():
        sentenceList = split_into_sentences(row['alltext'])
        for sentence in sentenceList:
            for ticker in usTickers:
                if ticker in sentence:
                    sentiment = calculateSentiment(sentence)
                    tempTickers.append(ticker)
                    tempPosList.append(sentiment[0])
                    tempNegList.append(sentiment[1])
                    tempNeuList.append(sentiment[2])
                    dates.append(row['date'])
    
    resultDf = pd.DataFrame({
        "date": dates,
        "ticker": tempTickers,
        "positive": np.array(tempPosList)*100,
        "negative": np.array(tempNegList)*100,
        "neutral": np.array(tempNeuList)*100
    })
    return resultDf

# Model ve tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# US tickers listesi (örnek, gerçek listeyi buraya ekleyin)
usTickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Haber koleksiyonu DataFrame (newsCollection) üzerinde duygu analizi yapın
final_df = historicalSentiment(newsCollection, usTickers)
print(final_df)




#############3ERROR##################################################3

---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
File ~/.local/lib/python3.8/site-packages/urllib3/connection.py:174, in HTTPConnection._new_conn(self)
    173 try:
--> 174     conn = connection.create_connection(
    175         (self._dns_host, self.port), self.timeout, **extra_kw
    176     )
    178 except SocketTimeout:

File ~/.local/lib/python3.8/site-packages/urllib3/util/connection.py:95, in create_connection(address, timeout, source_address, socket_options)
     94 if err is not None:
---> 95     raise err
     97 raise socket.error("getaddrinfo returns an empty list")

File ~/.local/lib/python3.8/site-packages/urllib3/util/connection.py:85, in create_connection(address, timeout, source_address, socket_options)
     84     sock.bind(source_address)
---> 85 sock.connect(sa)
     86 return sock

OSError: [Errno 101] Network is unreachable

During handling of the above exception, another exception occurred:

NewConnectionError                        Traceback (most recent call last)
File ~/.local/lib/python3.8/site-packages/urllib3/connectionpool.py:715, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
    714 # Make the request on the httplib connection object.
--> 715 httplib_response = self._make_request(
    716     conn,
    717     method,
    718     url,
    719     timeout=timeout_obj,
    720     body=body,
    721     headers=headers,
    722     chunked=chunked,
    723 )
    725 # If we're going to release the connection in ``finally:``, then
    726 # the response doesn't need to know about the connection. Otherwise
    727 # it will also try to release it and we'll have a double-release
    728 # mess.

File ~/.local/lib/python3.8/site-packages/urllib3/connectionpool.py:404, in HTTPConnectionPool._make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
    403 try:
--> 404     self._validate_conn(conn)
    405 except (SocketTimeout, BaseSSLError) as e:
    406     # Py2 raises this as a BaseSSLError, Py3 raises it as socket timeout.

File ~/.local/lib/python3.8/site-packages/urllib3/connectionpool.py:1058, in HTTPSConnectionPool._validate_conn(self, conn)
   1057 if not getattr(conn, "sock", None):  # AppEngine might not have  `.sock`
-> 1058     conn.connect()
   1060 if not conn.is_verified:

File ~/.local/lib/python3.8/site-packages/urllib3/connection.py:363, in HTTPSConnection.connect(self)
    361 def connect(self):
    362     # Add certificate verification
--> 363     self.sock = conn = self._new_conn()
    364     hostname = self.host

File ~/.local/lib/python3.8/site-packages/urllib3/connection.py:186, in HTTPConnection._new_conn(self)
    185 except SocketError as e:
--> 186     raise NewConnectionError(
    187         self, "Failed to establish a new connection: %s" % e
    188     )
    190 return conn

NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x7fae5c351ee0>: Failed to establish a new connection: [Errno 101] Network is unreachable

During handling of the above exception, another exception occurred:

MaxRetryError                             Traceback (most recent call last)
File ~/.local/lib/python3.8/site-packages/requests/adapters.py:440, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
    439 if not chunked:
--> 440     resp = conn.urlopen(
    441         method=request.method,
    442         url=url,
    443         body=request.body,
    444         headers=request.headers,
    445         redirect=False,
    446         assert_same_host=False,
    447         preload_content=False,
    448         decode_content=False,
    449         retries=self.max_retries,
    450         timeout=timeout
    451     )
    453 # Send the request.
    454 else:

File ~/.local/lib/python3.8/site-packages/urllib3/connectionpool.py:799, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
    797     e = ProtocolError("Connection aborted.", e)
--> 799 retries = retries.increment(
    800     method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
    801 )
    802 retries.sleep()

File ~/.local/lib/python3.8/site-packages/urllib3/util/retry.py:592, in Retry.increment(self, method, url, response, error, _pool, _stacktrace)
    591 if new_retry.is_exhausted():
--> 592     raise MaxRetryError(_pool, url, error or ResponseError(cause))
    594 log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)

MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /ProsusAI/finbert/resolve/main/model.safetensors (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7fae5c351ee0>: Failed to establish a new connection: [Errno 101] Network is unreachable'))

During handling of the above exception, another exception occurred:

ConnectionError                           Traceback (most recent call last)
Cell In[6], line 248
    246 # Model ve tokenizer yükleme
    247 tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
--> 248 model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    250 # US tickers listesi (örnek, gerçek listeyi buraya ekleyin)
    251 # usTickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    252 
    253 # Haber koleksiyonu DataFrame (newsCollection) üzerinde duygu analizi yapın
    254 final_df = historicalSentiment(newsCollection, usTickers)

File ~/.local/lib/python3.8/site-packages/transformers/models/auto/auto_factory.py:564, in _BaseAutoModelClass.from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    562 elif type(config) in cls._model_mapping.keys():
    563     model_class = _get_model_class(config, cls._model_mapping)
--> 564     return model_class.from_pretrained(
    565         pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
    566     )
    567 raise ValueError(
    568     f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
    569     f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
    570 )

File ~/.local/lib/python3.8/site-packages/transformers/modeling_utils.py:3494, in PreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   3475         has_file_kwargs = {
   3476             "revision": revision,
   3477             "proxies": proxies,
   (...)
   3480             "local_files_only": local_files_only,
   3481         }
   3482         cached_file_kwargs = {
   3483             "cache_dir": cache_dir,
   3484             "force_download": force_download,
   (...)
   3492             **has_file_kwargs,
   3493         }
-> 3494         if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
   3495             Thread(
   3496                 target=auto_conversion,
   3497                 args=(pretrained_model_name_or_path,),
   3498                 kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
   3499                 name="Thread-autoconversion",
   3500             ).start()
   3501 else:
   3502     # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
   3503     # We try those to give a helpful error message.

File ~/.local/lib/python3.8/site-packages/transformers/utils/hub.py:655, in has_file(path_or_repo, filename, revision, proxies, token, local_files_only, cache_dir, repo_type, **deprecated_kwargs)
    653 # Check if the file exists
    654 try:
--> 655     response = get_session().head(
    656         hf_hub_url(path_or_repo, filename=filename, revision=revision, repo_type=repo_type),
    657         headers=build_hf_headers(token=token, user_agent=http_user_agent()),
    658         allow_redirects=False,
    659         proxies=proxies,
    660         timeout=10,
    661     )
    662 except OfflineModeIsEnabled:
    663     return has_file_in_cache

File ~/.local/lib/python3.8/site-packages/requests/sessions.py:564, in Session.head(self, url, **kwargs)
    556 r"""Sends a HEAD request. Returns :class:`Response` object.
    557 
    558 :param url: URL for the new :class:`Request` object.
    559 :param \*\*kwargs: Optional arguments that ``request`` takes.
    560 :rtype: requests.Response
    561 """
    563 kwargs.setdefault('allow_redirects', False)
--> 564 return self.request('HEAD', url, **kwargs)

File ~/.local/lib/python3.8/site-packages/requests/sessions.py:529, in Session.request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
    524 send_kwargs = {
    525     'timeout': timeout,
    526     'allow_redirects': allow_redirects,
    527 }
    528 send_kwargs.update(settings)
--> 529 resp = self.send(prep, **send_kwargs)
    531 return resp

File ~/.local/lib/python3.8/site-packages/requests/sessions.py:645, in Session.send(self, request, **kwargs)
    642 start = preferred_clock()
    644 # Send the request
--> 645 r = adapter.send(request, **kwargs)
    647 # Total elapsed time of the request (approximately)
    648 elapsed = preferred_clock() - start

File ~/.local/lib/python3.8/site-packages/huggingface_hub/utils/_http.py:66, in UniqueRequestIdAdapter.send(self, request, *args, **kwargs)
     64 """Catch any RequestException to append request id to the error message for debugging."""
     65 try:
---> 66     return super().send(request, *args, **kwargs)
     67 except requests.RequestException as e:
     68     request_id = request.headers.get(X_AMZN_TRACE_ID)

File ~/.local/lib/python3.8/site-packages/requests/adapters.py:519, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
    515     if isinstance(e.reason, _SSLError):
    516         # This branch is for urllib3 v1.22 and later.
    517         raise SSLError(e, request=request)
--> 519     raise ConnectionError(e, request=request)
    521 except ClosedPoolError as e:
    522     raise ConnectionError(e, request=request)

ConnectionError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /ProsusAI/finbert/resolve/main/model.safetensors (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7fae5c351ee0>: Failed to establish a new connection: [Errno 101] Network is unreachable'))"), '(Request ID: 6d85cef5-d084-4582-b4ae-f9e36b0757a9)')




