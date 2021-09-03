import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='../bert')
parser.add_argument('--save_path', help='path to save checkpoints', default='../train')
parser.add_argument('--train_file', help='training data', default='../data/train.tsv')
parser.add_argument('--valid_file', help='valid data', default='../data/test.tsv')
parser.add_argument('--intent_label_vocab', help='training file', default='../data/cls_vocab')
parser.add_argument('--slot_label_vocab', help='training file', default='../data/slot_vocab')

parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=30)
parser.add_argument('--batch_split', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF", default=True)
parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from transformers import BertConfig, BertTokenizer, AdamW
from NLU_model import NLUModule
import torch
from NLU_model import NLUModule
import numpy as np
import utils


train_path = os.path.join(args.save_path, 'train')
model_path = os.path.join(train_path, 'model_crf.ckpt')
max_lengths = args.max_length
tokz = BertTokenizer.from_pretrained(args.bert_path)
_, intent2index, index2intent = utils.load_vocab(args.intent_label_vocab)
_, slot2index, index2slot = utils.load_vocab(args.slot_label_vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def class_pred(intent_logit):
    intent_pred = intent_logit.argmax(axis=1)
    return intent_pred


def string_pred(intent_logit, ner_pred, mask):
    intent_pred = class_pred(intent_logit)
    print(intent_pred)
    print(ner_pred)
    intent_pred = [index2intent[i.item()] for i in intent_pred]
    # 在预测的时候第一个位置[CLS]，最后一个位置[SEP]不做预测
    ner_pred_mask = [np.array(j)[np.array(i) == 1][1:-1] for i, j in zip(mask, ner_pred)]
    ner_pred_string = [[index2slot[j.item()] for j in i] for i in ner_pred_mask]
    return intent_pred, ner_pred_string


def encode(text):
    '''
    从dataset拷贝的代码
    :param text:
    :return:
    '''
    utt = tokz.convert_tokens_to_ids(list(text)[:max_lengths])
    utt = [tokz.cls_token_id] + utt + [tokz.sep_token_id]
    mask = [1] * len(utt)
    token_type_ids = [0] * len(utt)
    return ({'utt': torch.LongTensor([utt]),
                 'mask': torch.LongTensor([mask]),
                 'token_type_ids': torch.LongTensor([token_type_ids])
                 })


def predict(text):
    encoded_data = encode(text)
    input_ids = encoded_data['utt']
    mask = encoded_data['mask']
    token_type_ids  = encoded_data['token_type_ids']
    bert_config = BertConfig.from_pretrained(args.bert_path)
    bert_config.num_intent_labels = len(intent2index)
    bert_config.num_slot_labels = len(slot2index)
    bert_config.use_crf = args.use_crf
    model = NLUModule.from_pretrained(args.bert_path, config=bert_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    intent_logits, slot_logits, _ = model(input_ids, mask, token_type_ids)

    slot_preds = np.array(model.crf.decode(slot_logits))
    print('crf predicat ',slot_preds)

    intent_pred, ner_pred = string_pred(intent_logits, slot_preds, mask)
    print(intent_pred)
    print(ner_pred)



if __name__ == '__main__':
    '''
    data = pickle.load(open('small_batch_train.pkl', 'rb'))
    input_ids = data['input_ids']
    mask = data['mask']
    token_type_ids = data['token_type_ids']
    ner_target = data['ner_target']
    intent_target = data['intent_target']
    print('input_ids.shape = ', input_ids.shape)
    print('ner_target.shape = ', ner_target.shape)
    print('intent_target.shape = ', intent_target.shape)
    model = NLU_model()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    intent_logit, ner_logit = model(input_ids, mask, token_type_ids)
    intent_pred , ner_pred = string_pred(intent_logit, ner_logit, mask)

    print('真实', config.intent_encoder.inverse_transform(intent_target).tolist())
    print('预测且字符串化', intent_pred)
    print('============================')
    ner_target_string = []
    for i, item in enumerate(ner_target):
        ner_target_string.append(config.ner_encoder.inverse_transform(item).tolist())
    print('真实', ner_target_string)
    print('预测且字符串化', ner_pred) 
    '''

    text = '酸菜粉条肉的做法'
    #text = '童子鸡的做法。'
    #text = '现在电视台在放什么节目'
    #text = '安徽电视台12月18号晚上10:10的电视剧'
    #text = '找山西卫视'
    #text = '高清电影'
    #text = '从东莞去北海的汽车'
    predict(text)

