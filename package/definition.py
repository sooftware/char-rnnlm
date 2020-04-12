import logging
import sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
from package.data_loader import load_label
char2id, id2char = load_label('./data/label/train_labels.csv', encoding='utf-8')
SOS_token = int(char2id['<s>'])
EOS_token = int(char2id['</s>'])
PAD_token = int(char2id['_'])
train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}