from allennlp.data.fields import *
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.nn.util import get_text_field_mask
from allennlp.data.tokenizers import Token
from allennlp.models import BasicClassifier, Model
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, Average, Metric
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics.metric import Metric
from allennlp.nn import util

from typing import *
from overrides import overrides
import jieba
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import cv2 as cv
import os
from PIL import Image
import PIL

torch.manual_seed(123)

check_list = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F',
              'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L',
              'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R',
              's', 'S', 't', 'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X',
              'y', 'Y', 'z', 'Z',]# '⊙', '∠', '△', '▱', ]#'∥', '⊥',]

check_list_cap = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',]# '⊙', '∠', '△', '▱', '⊿']
check_list_sma = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',]# '⊙', '∠', '△', '▱', '⊿']
check_nx_list = ['N_0', 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6', 'N_7', 'N_8', 'N_9', 'N_10', 'N_11']

check_num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
check_num_list_ = ['l', '∠']

def process_image_ori(img, min_side=224):  # 等比例缩放与填充
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    # 下右填充
    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w

    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255]) # 从图像边界向上,下,左,右扩的像素数目

    return pad_img

def process_image(img, min_side=224):  # 等比例缩放与填充
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))

    top = int((min_side-new_h)/2)
    bottom = int((min_side-new_h)/2) +1 
    left = int((min_side-new_w)/2)
    right = int((min_side-new_w)/2) + 1
    
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255]) # 从图像边界向上,下,左,右扩的像素数目

    return pad_img



def resize(image, size=224, resample=PIL.Image.BILINEAR, default_to_square=True, max_size=None):
    if isinstance(size, list):
        size = tuple(size)

    if isinstance(size, int) or len(size) == 1:
        if default_to_square:
            size = (size, size) if isinstance(size, int) else (size[0], size[0])
        else:
            width, height = image.size
            # specified size only for the smallest edge
            short, long = (width, height) if width <= height else (height, width)
            requested_new_short = size if isinstance(size, int) else size[0]

            if short == requested_new_short:
                return image

            new_short, new_long = requested_new_short, int(requested_new_short * long / short)

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            size = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize(size, resample=resample)



@DatasetReader.register("s2s_manual_reader")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "./GeoQA-Data/GeoQA-Pro/sub_dataset_dict.pk"
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        self.all_points = ['切线', '垂径定理', '勾股定理', '同位角', '平行线', '三角形内角和', '三角形中位线', '平行四边形',
                  '相似三角形', '正方形', '圆周角', '直角三角形', '距离', '邻补角', '圆心角', '圆锥的计算', '三角函数',
                  '矩形', '旋转', '等腰三角形', '外接圆', '内错角', '菱形', '多边形', '对顶角', '三角形的外角', '角平分线',
                  '对称', '立体图形', '三视图', '圆内接四边形', '垂直平分线', '垂线', '扇形面积', '等边三角形', '平移',
                  '含30度角的直角三角形', '仰角', '三角形的外接圆与外心', '方向角', '坡角', '直角三角形斜边上的中线', '位似',
                  '平行线分线段成比例', '坐标与图形性质', '圆柱的计算', '俯角', '射影定理', '黄金分割', '钟面角', '多边形内角和', '外接圆', '弦长', '长度', '中垂线',
                  '相交线', '全等三角形', '梯形', '锐角', '补角', '比例线段', '比例角度', '圆形', '正多边形', '同旁内角', '余角', '三角形的重心', '旋转角', '中心对称',
                  '三角形的内心', '投影', '对角线','弧长的计算' , '平移的性质' , '位似变换' ,'菱形的性质' ,'正方形的性质']
        #弧长的计算  平移的性质  位似变换 菱形的 性质 正方形的性质


        #self.all_points = ['切线']
    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = Image.fromarray(np.uint8(image))
        img_rgb = image.convert("RGB")
        img_rgb = resize(img_rgb, size=224)
        fields['image'] = MetadataField(img_rgb)
     
        texts = ['[CLS]'] + sample['token_list'] + ['[SEP]']
        while '\n' in texts:
            texts.remove('\n')
        while '\r' in texts:
            texts.remove('\r')
        s_token = self._tokenizer.tokenize(' '.join(texts))


        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)

        ans = sample['manual_program']

        for ind, token in enumerate(ans):
            if len(token) >= 3:
                if token[:2] == 'c_':
                    ans[ind] = 'C_' + token[2:]
                if token[:2] == 'v_':
                    ans[ind] = 'V_' + token[2:]


        for token in ans:
            if len(token) == 3:
                if token[:2] == 'c_':
                    print(ans)
                    break
                if token[:2] == 'v_':
                    print(ans)
                    break        


        t_token = self._tokenizer.tokenize(' '.join(ans))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)


        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])
        equ_list = []




        equ = ans
        equ_token = self._tokenizer.tokenize(' '.join(equ))
        equ_token.insert(0, Token(START_SYMBOL))
        equ_token.append(Token(END_SYMBOL))
        equ_token = TextField(equ_token, self._source_token_indexer)
        equ_list.append(equ_token)

        fields['equ_list'] = ListField(equ_list)
        fields['manual_program'] = MetadataField(ans)

        point_label = np.zeros(77, np.float32)
        exam_points = sample['formal_point']
        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))


        # Merge Mask
        # texts = sample['token_list']
        merge_mask = [0] * len(texts)

        for ind, token in enumerate(texts):
            if token in check_list_cap: # It is a single word
                if ind < (len(texts)-1): # question end with: ()m
                    if texts[ind+1] in check_list_cap: # and the next is also     
                        merge_mask[ind] = 1
                        merge_mask[ind+1] = 1
                        # merge_pos_id[ind] = merge_pos_id[ind-1] + 1
                        # merge_pos_id[ind+1] = merge_pos_id[ind] + 1
        fields['merge_mask'] = MetadataField(merge_mask)



        # has_num = False

        # Merge Cap Char Mask
        merge_cap_mask = [1] * len(texts)
        merge_pos_id = np.array([0] * len(s_token))
        for ind, token in enumerate(texts):
            if token in check_list_cap: # It is a single word
                if texts[ind+1] in check_list_sma:
                    pass               
                else:
                    merge_cap_mask[ind] = 0
                    merge_pos_id[ind] = merge_pos_id[ind-1] + 1

            if token in check_num_list:     # 
                if texts[ind-1] in check_num_list_: # ∠1, l2
                    merge_cap_mask[ind] = 0
                    merge_pos_id[ind] = merge_pos_id[ind-1] + 1
                    # print(sample['id'])


        fields['merge_cap_mask'] = ArrayField(np.array(merge_cap_mask)) # only A letters and nums like 'l1' '∠2', Attention to image
        fields['merge_pos_id'] = ArrayField(merge_pos_id)
        

        # print('===============')
        # print(merge_cap_mask)
        # print(merge_pos_id)

        
        # if has_num:
        #     print(sample['token_list'])


        # if sample['id'] > 10000:
        #     print('-------------------------')
        #     print(texts)

        #     print(merge_cap_mask)
            
        #     print(s_token)
        #     exit()

        # print('=================')
        # print(texts)
        # print(merge_cap_mask)
        # print(merge_mask)


        return Instance(fields)

