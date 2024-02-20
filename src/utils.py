import collections
import numpy as np
from operator import itemgetter
from datasets.formatting.formatting import LazyBatch
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
import os
import yaml

with open('./config.yaml') as f:
    my_dict = yaml.safe_load(f)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config() -> Struct:
    args = my_dict
    s = Struct(**args)
    return s


cnf = get_config()


# create directory
def make_dirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


# Normalize box to 0,1
def normalize_box(bbox: list, width: int, height: int) -> list:
    return [
        float((bbox[0] / width)),
        float((bbox[1] / height)),
        float((bbox[2] / width)),
        float((bbox[3] / height)),
    ]


# Normalize box to 0,1000
def normalize_box_1000(bbox: list, width: int, height: int) -> list:
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


# Denormalize box to width, height
def unnormalize_box(bbox: list, width: int, height: int) -> list:
    return [
        int(width * (bbox[0])),
        int(height * (bbox[1])),
        int(width * (bbox[2])),
        int(height * (bbox[3])),
    ]


# Denormalize box to width, height
def unnormalize_box_1000(bbox: list, width: int, height: int) -> list:
    return [
        int(width * (bbox[0]) / 1000),
        int(height * (bbox[1]) / 1000),
        int(width * (bbox[2]) / 1000),
        int(height * (bbox[3]) / 1000),
    ]


# Denormalize image to 01 --  assuming 8bit
def normalize_image(image):
    return image / 255.0


# get bbox in original image width and height
def original_box(bbox: list, original_width: int, original_height: int, coco_width: int, coco_height: int) -> list:
    return [
        int(original_width * (bbox[0] / coco_width)),
        int(original_height * (bbox[1] / coco_height)),
        int(original_width * (bbox[2] / coco_width)),
        int(original_height * (bbox[3] / coco_height)),
    ]


# convert bbox coordinates in (x1, y1, x2, y2)
def convert_box(bbox: list) -> list:
    x, y, w, h = tuple(bbox)
    return [x, y, x + w, y + h]


# function to sort bounding boxes
def get_sorted_boxes(bboxes):
    # sort by y from page top to bottom
    sorted_bboxes = sorted(bboxes, key=itemgetter(1), reverse=False)
    y_list = [bbox[1] for bbox in sorted_bboxes]

    # sort by x from page left to right when boxes with same y
    if len(list(set(y_list))) != len(y_list):
        y_list_duplicates_indexes = dict()
        y_list_duplicates = [item for item, count in collections.Counter(y_list).items() if count > 1]
        for item in y_list_duplicates:
            y_list_duplicates_indexes[item] = [i for i, e in enumerate(y_list) if e == item]
            bbox_list_y_duplicates = sorted(
                np.array(sorted_bboxes, dtype=object)[y_list_duplicates_indexes[item]].tolist(), key=itemgetter(0),
                reverse=False)
            np_array_bboxes = np.array(sorted_bboxes)
            np_array_bboxes[y_list_duplicates_indexes[item]] = np.array(bbox_list_y_duplicates)
            sorted_bboxes = np_array_bboxes.tolist()

    return sorted_bboxes


def sort_data(bboxes, categories, texts):
    sorted_bboxes = get_sorted_boxes(bboxes)
    sorted_bboxes_indexes = [bboxes.index(bbox) for bbox in sorted_bboxes]
    sorted_categories = np.array(categories, dtype=object)[sorted_bboxes_indexes].tolist()
    sorted_texts = np.array(texts, dtype=object)[sorted_bboxes_indexes].tolist()
    return sorted_bboxes, sorted_categories, sorted_texts


def prepare_examples(example: LazyBatch, tokenizer: LlamaTokenizerFast, cls_box: list = [0, 0, 0, 0],
                     sep_box: list = [0, 0, 0, 0],
                     label_pad_token_id: int = -100) -> object:
    input_ids_list, attention_mask_list, bb_list, ll_list, page_hash_list, original_image_list = list(), list(), list(), list(), list(), list()

    # get batch
    batch_page_hash = example["page_hash"]
    batch_bboxes_block = example["bboxes_block"]
    batch_categories = example["categories"]
    batch_texts = example["texts"]
    batch_images = example["image"]
    batch_original_width, batch_original_height = example["original_width"], example["original_height"]
    batch_coco_width, batch_coco_height = example["coco_width"], example["coco_height"]

    # add a dimension if not a batch but only one image
    if not isinstance(batch_page_hash, list):
        batch_page_hash = [batch_page_hash]
        batch_bboxes_block = [batch_bboxes_block]
        batch_categories = [batch_categories]
        batch_texts = [batch_texts]
        batch_images = [batch_images]
        batch_original_width, batch_original_height = [batch_original_width], [batch_original_height]
        batch_coco_width, batch_coco_height = [batch_coco_width], [batch_coco_height]

    # process all images of the batch
    for num_batch, (
            page_hash, boxes, labels, texts, image, coco_width, coco_height, original_width,
            original_height) in enumerate(
        zip(batch_page_hash, batch_bboxes_block, batch_categories, batch_texts, batch_images, batch_coco_width,
            batch_coco_height, batch_original_width, batch_original_height)):
        tokens_list = []
        bboxes_list = []
        labels_list = []

        # resize image to input width, height
        original_image = image.resize((cnf.input_size, cnf.input_size)).convert("RGB")

        # add a dimension if only on image
        if not isinstance(texts, list):
            texts, boxes, labels = [texts], [boxes], [labels]

        normalize_bboxes_block = [normalize_box_1000((convert_box(box)), coco_width, coco_height) for box in boxes]
        boxes, labels, texts = sort_data(normalize_bboxes_block, labels, texts)

        count = 0
        for box, label, text in zip(boxes, labels, texts):
            tokens = tokenizer.tokenize(text)
            num_tokens = len(tokens)  # get number of tokens
            tokens_list.extend(tokens)

            bboxes_list.extend([box] * num_tokens)  # number of boxes must be the same as the number of tokens
            labels_list.extend([label if token.startswith('▁') else label_pad_token_id for token in
                                tokens])  # WARNING: check the tokenizer to get the string to search

        encodings = tokenizer(" ".join(texts),
                              truncation=True,
                              padding="max_length",
                              max_length=cnf.max_length,
                              stride=cnf.doc_stride,
                              return_overflowing_tokens=True,
                              return_offsets_mapping=True
                              )

        _ = encodings.pop("overflow_to_sample_mapping")
        offset_mapping = encodings.pop("offset_mapping")

        # Let's label those examples and get their boxes
        sequence_length_prev = 0
        for i, offsets in enumerate(offset_mapping):
            # truncate tokens, boxes and labels based on length of chunk - 2 (special tokens <s> and </s>)
            sequence_length = len(encodings.input_ids[i]) - 2
            if i == 0:
                start = 0
            else:
                start += sequence_length_prev - cnf.doc_stride
            end = start + sequence_length
            sequence_length_prev = sequence_length

            # get tokens, boxes and labels of this image chunk
            bb = [cls_box] + bboxes_list[start:end] + [sep_box]
            # get labels for this chunck
            ll = [label_pad_token_id] + labels_list[start:end] + [label_pad_token_id]

            # as the last chunk can have a length < max_length
            # we must to add [tokenizer.pad_token] (tokens), [sep_box] (boxes) and [label_pad_token_id] (labels)
            if len(bb) < cnf.max_length:
                bb = bb + [sep_box] * (cnf.max_length - len(bb))
                ll = ll + [label_pad_token_id] * (cnf.max_length - len(ll))

            # append results
            input_ids_list.append(encodings["input_ids"][i])
            attention_mask_list.append(encodings["attention_mask"][i])
            bb_list.append(bb)
            ll_list.append(ll)
            page_hash_list.append(page_hash)
            original_image_list.append(np.array(original_image))

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": ll_list,
        "bbox": bb_list,
        "pixel_values": original_image_list,
    }
