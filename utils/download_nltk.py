import nltk
import os
import ssl

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

nltk.data.path.append('/data/miniconda/envs/vsrn/nltk_data') 
nltk.download('punkt')
nltk.download('punkt_tab')