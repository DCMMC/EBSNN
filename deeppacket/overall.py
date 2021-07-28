BATCH_SIZE = 128
EPOCHS = 120
LR = 0.001

NGRAM = 10
NUM_CLASS = 20
EMBEDDING_DIM = 100

CUDA = True
DEVICE = 3

DEBUG = False

# FILENAME = '../bsnn/data/tor_erase_tls2.traffic'
FILENAME = '../bsnn/data/20_header_payload_all.traffic'

# LABELS ={'facebook': 0, 'yahoomail': 1, 'Youtube': 2, 'itunes': 3, 'mysql': 4, 'filezilla': 5, 'amazon': 6, 'skype': 7, 'google': 8, 'gmail': 9, 'vimeo': 10, 'twitter': 11, 'spotify': 12, 'netflix': 13, 'aimchat': 14, 'voipbuster': 15, 'jd': 16, 'taobao': 17, 'pp': 18, 'weibo': 19, 'baidu': 20, 'thunder': 21, 'sohu': 22, 'youku': 23, 'tudou': 24, 'KG': 25, 'sinaUC': 26, 'cloudmusic': 27, 'qq': 28, 'ftps': 29, 'snmp': 30, 'ssh': 31, 'https': 32, 'smtp': 33, 'dns': 34, 'bittorrent': 35}
# LABELS = {'vimeo': 0, 'spotify': 1, 'voipbuster': 2, 'sinauc': 3, 'cloudmusic': 4, 'weibo': 5, 'baidu': 6, 'tudou': 7, 'amazon': 8, 'thunder': 9, 'gmail': 10, 'pplive': 11, 'qq': 12, 'taobao': 13, 'yahoomail': 14, 'itunes': 15, 'twitter': 16, 'jd': 17, 'sohu': 18, 'youtube': 19, 'youku': 20, 'netflix': 21, 'aimchat': 22, 'kugou': 23, 'skype': 24, 'facebook': 25, 'google': 26, 'mssql': 27, 'ms-exchange': 28}
# LABELS = {'audio': 0, 'browsing': 1, 'chat': 2,
#           'file': 3, 'mail': 4,
#           'p2p': 5, 'video': 6, 'voip': 7}
LABELS = {'reddit': 0, 'facebook': 1, 'NeteaseMusic': 2,
          'twitter': 3, 'qqmail': 4, 'instagram': 5,
          'weibo': 6, 'iqiyi': 7, 'imdb': 8,
          'TED': 9, 'douban': 10,
          'amazon': 11, 'youtube': 12, 'JD': 13,
          'youku': 14, 'baidu': 15,
          'google': 16, 'tieba': 17, 'taobao': 18,
          'bing': 19}

test_percent = 0.2

# save model
load_model_name = 'mymodel/ulti.model1'
save_model_name = 'mymodel/ulti.model2'
