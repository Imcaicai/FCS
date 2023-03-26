import os
import csv
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import InputExample, InputFeatures
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bert_model = (BertConfig, BertForSequenceClassification, BertTokenizer)


def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    del lines[0]
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        # label = int(line[1])
        # CNM!!@!!
        text_a = line[2].replace("YZYHUST", ',')
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
    return examples


def Load_data(args, tokenizer):
    csv.field_size_limit(500 * 1024 * 1024)
    with open(os.path.join(args.data_dir, 'val.csv'), 'r') as f:
        examples = create_examples(list(csv.reader(f)), 'predict')
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    features = features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode="classification",
    )
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                      dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                      dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids)
    return DataLoader(dataset, batch_size=16)


#all in main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="./url/predict",
                        type=str,
                        required=False,
                        help="Locate the dataset")
    parser.add_argument("--model_dir",
                        default='./model',
                        type=str,
                        required=False,
                        help="Locate the pre-trained model")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        required=False)
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = bert_model

    # Load Bert
    config = config_class.from_pretrained(args.model_dir, num_labels=13)
    tokenizer = tokenizer_class.from_pretrained(args.model_dir)
    model = model_class.from_pretrained(args.model_dir,
                                        from_tf=False,
                                        config=config)
    model.to(device)
    #Load Data
    #pipeline: file->example->features
    pred_dataloader = Load_data(args, tokenizer)
    #Predict
    preds = None
    for batch in tqdm(pred_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)

            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    m = torch.nn.Softmax(dim=1)
    prob = np.array(m(torch.Tensor(preds)))
    pred = np.argmax(preds, axis=1)
    prob = np.max(prob, axis=1)
    pred = pd.DataFrame(pred)
    prob = pd.DataFrame(prob)
    prediction = pd.concat([pred, prob], axis=1)
    prediction.columns = ['pred', 'prob']
    # save results
    prediction.to_csv(index=None, path_or_buf='./predict.csv')


if __name__ == "__main__":
    print("current device:", device)
    main()
