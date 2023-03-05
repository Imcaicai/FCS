import csv
import torch
import argparse
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
        text_a = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
    return examples


def Load_data(args, tokenizer):
    with open(args.data_dir + 'pred.csv', 'r') as f:
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
                        default="./url",
                        type=str,
                        required=False,
                        help="Locate the dataset")
    parser.add_argument("--model_dir",
                        default='./outs',
                        type=str,
                        required=True,
                        help="Locate the pre-trained model")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        required=False)
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = bert_model

    # Load Bert
    config = config_class.from_pretrained(
        args.model_dir,
        num_labels=13,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(
        args.model_dir,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(device)
    #Load Data
    #TODO:process data
    #file->example->features
    pred_dataloader = Load_data(args, tokenizer)
    #Predict
    preds = None
    for batch in tqdm(pred_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in [
                    'bert', 'xlnet'
                ] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)


if __name__ == "__main__":
    main()
