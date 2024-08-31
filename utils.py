
import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label1=None, label2=None):
        self.guid = guid
        self.text = text 
        self.label1 = label1
        self.label2 = label2

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id1, label_id2, ori_tokens):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id1 = label_id1
        self.label_id2 = label_id2
        self.ori_tokens = ori_tokens


class NerProcessor(object):
    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            labels1 = []
            labels2 = []
            
            for line in f.readlines():   
                contents = line.strip()
                tokens = line.strip().split("\t")

                if len(tokens) == 3:
                    words.append(tokens[0])
                    labels1.append(tokens[1])
                    labels2.append(tokens[2])
                else:
                    if len(contents) == 0 and len(words) > 0:
                        label1 = []
                        label2 = []
                        word = []
                        for l1, l2, w in zip(labels1, labels2, words):
                            if len(l1) > 0 and len(l2) > 0 and len(w) > 0:
                                label1.append(l1)
                                label2.append(l2)
                                word.append(w)
                        lines.append([' '.join(label1), ' '.join(label2), ' '.join(word)])
                        words = []
                        labels1 = []
                        labels2 = []
            
            return lines
    
    def get_labels(self, args):
        labels1 = set()
        labels2 = set()
        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
            logger.info(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
                (labels1, labels2) = pickle.load(f)
        else:
            # get labels from train data
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            with open(args.train_file, encoding='utf8') as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")

                    if len(tokens) == 3:
                        labels1.add(tokens[1])
                        labels2.add(tokens[2])

            labels2.add('O')
            labels2.add('B-DEG')
            labels2.add('E-DEG')

            if len(labels1) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump((labels1, labels2), f)
            else:
                logger.info("loading error and return the default labels B,I,O")
                # labels = {"O", "B", "I"}
                labels = {"O", "B", "M", "E", "S"}
        
        return labels1, labels2

    def get_examples(self, input_file):
        examples = []
        
        lines = self.read_data(input_file)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[2]
            label1 = line[0]
            label2 = line[1]

            examples.append(InputExample(guid=guid, text=text, label1=label1, label2=label2))
        
        return examples


def convert_examples_to_features(args, examples, label_list1, label_list2, max_seq_length, tokenizer):

    label_map1 = {label : i for i, label in enumerate(label_list1)}
    label_map2 = {label : i for i, label in enumerate(label_list2)}

    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        textlist = example.text.split(" ")
        labellist1 = example.label1.split(" ")
        labellist2 = example.label2.split(" ")
        assert len(textlist) == len(labellist1)
        tokens = []
        labels1 = []
        labels2 = []
        ori_tokens = []

        for i, word in enumerate(textlist):
            # 防止wordPiece情况出现，不过貌似不会
            
            # token = tokenizer.tokenize(word)
            # tokens.extend(token)
            token = word
            tokens.append(word)
            label_1 = labellist1[i]
            label_2 = labellist2[i]
            ori_tokens.append(word)

            # # 单个字符不会出现wordPiece
            # for m in range(len(token)):
            #     if m == 0:
            #         labels.append(label_1)
            #     else:
            #         if label_1 == "O":
            #             labels.append("O")
            #         else:
            #             labels.append("I")
            labels1.append(label_1)
            labels2.append(label_2)
            
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels1 = labels1[0:(max_seq_length - 2)]
            labels2 = labels2[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
        
        ntokens = []
        segment_ids = []
        label_ids1 = []
        label_ids2 = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids1.append(label_map1["O"])
        label_ids2.append(label_map2["O"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids1.append(label_map1[labels1[i]])
            label_ids2.append(label_map2[labels2[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids1.append(label_map1["O"])
        label_ids2.append(label_map2["O"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)   
        
        input_mask = [1] * len(input_ids)

        assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}, {ntokens}"

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids1.append(0)
            label_ids2.append(0)
            ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids1) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in ntokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids1: %s" % " ".join([str(x) for x in label_ids1]))
            logger.info("label_ids2: %s" % " ".join([str(x) for x in label_ids2]))

        # if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        #     with open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
        #         pickle.dump(label_map, w)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id1=label_ids1,
                              label_id2=label_ids2,
                              ori_tokens=ori_tokens))

    return features


def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_examples(filepath)
    label_list1 = args.label_list1
    label_list2 = args.label_list2

    features = convert_examples_to_features(
        args, examples, label_list1, label_list2, args.max_seq_length, tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids1 = torch.tensor([f.label_id1 for f in features], dtype=torch.long)
    all_label_ids2 = torch.tensor([f.label_id2 for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids1, all_label_ids2)

    return examples, features, data

