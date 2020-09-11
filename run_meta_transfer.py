import logging
import os
import random
import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from maml import MetaLearner
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_multiple_choice import MetaMultipleChoiceDataset, MultipleChoiceDataset, Split, processors
from sklearn.metrics import accuracy_score

import torch
from torch.optim import Adam
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F



logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class MetaTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    outer_batch_size: int = field(
        default=2,
        metadata={"help":""},
    )
    # Same to k_support
    inner_batch_size: int = field(
        default=5,
        metadata={"help":""},
    )
    # learning rate : lambda 
    mtl_update_lr: float = field(
        default=5e-5,
        metadata={"help":""},
    )
    # learning rate : beta
    outer_update_lr: float = field(
        default=5e-5,
        metadata={"help":""},
    )
    # learning rate : alpha
    inner_update_lr: float = field(
        default=5e-5,
        metadata={"help":""},
    )
    inner_update_step: int = field(
        default=1,
        metadata={"help":""},
    )
    inner_update_step_eval: int = field(
        default=1,
        metadata={"help":""},
    )
    bert_model: str = field(
        default='bert-base-uncased',
        metadata={"help": "" },
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MetaTrainingArguments))
    model_args, data_args, training_args, metatraining_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # BertForMultipleChoice
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    s1_train_dataset = (
        MetaMultipleChoiceDataset(
            data_dir=os.path.join(data_args.data_dir, 'swag'),
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            num_task=20,
            k_support=5,
            k_query=1,
        )
        if training_args.do_train
        else None
    )

    # s2_train_dataset = (
    #     MetaMultipleChoiceDataset(
    #         data_dir=os.path.join(data_args.data_dir, 'ComVE_A'),
    #         tokenizer=tokenizer,
    #         task=data_args.task_name,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.train,
            # num_task=100,
            # k_support=5,
            # k_query=1,
    #     )
    #     if training_args.do_train
    #     else None
    # )

    # s3_train_dataset = (
    #     MetaMultipleChoiceDataset(
    #         data_dir=os.path.join(data_args.data_dir, 'ComVE_B'),
    #         tokenizer=tokenizer,
    #         task=data_args.task_name,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.train,
            # num_task=100,
            # k_support=5,
            # k_query=1,
    #     )
    #     if training_args.do_train
    #     else None
    # )
    # s1_train_dataset = (
    #     MultipleChoiceDataset(
    #         data_dir=os.path.join(data_args.data_dir, 'swag'),
    #         tokenizer=tokenizer,
    #         task=data_args.task_name,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.train,
    #     )
    #     if training_args.do_train
    #     else None
    # )
    # eval_dataset = (
    #     MultipleChoiceDataset(
    #         data_dir=data_args.data_dir,
    #         tokenizer=tokenizer,
    #         task=data_args.task_name,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.test,
    #     )
    #     if training_args.do_eval
    #     else None
    # )

    target_train_dataset = (
        MultipleChoiceDataset(
            data_dir=os.path.join(data_args.data_dir, 'cqa'), 
            tokenizer=tokenizer,
            task='cqa_clf',
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )

    # [TODO]:Modify this...
    # target_test_dataset = (
    #     MultipleChoiceDataset(
    #         data_dir=os.path.join(data_args.data_dir, 'cqa'), 
    #         tokenizer=tokenizer,
    #         task='cqa_clf',
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.test,
    #     )
    #     if training_args.do_train
    #     else None
    # )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}


    # Initialize our Trainer


    # Create meta batch
    s1_db = create_batch_of_tasks(s1_train_dataset, is_shuffle = True, batch_size = metatraining_args.outer_batch_size) 
    # s2_db = create_batch_of_tasks(s2_train_dataset, is_shuffle = True, batch_size = metatraining_args.outer_batch_size) 
    # s3_db = create_batch_of_tasks(s3_train_dataset, is_shuffle = True, batch_size = metatraining_args.outer_batch_size) 

    # Define Data Loader

    def _get_train_sampler(train_dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                RandomSampler(train_dataset)
            )

    # s1_train_sampler = _get_train_sampler(s1_train_dataset)

    # s1_train_dataloader = DataLoader(s1_tarin_dataset,
    #  batch_size=args.train_batch_size,
    #  sampler=s1_train_sampler,
    #  collate_fn=DataCollatorWithPadding(tokenizer),
    #  drop_last=args.dataloader_drop_last)
    
    target_train_sampler = _get_train_sampler(target_train_dataset)

    target_train_dataloader = DataLoader(target_train_dataset,
    batch_size=training_args.train_batch_size,
    sampler=target_train_sampler,
    collate_fn=default_data_collator, #DataCollatorWithPadding(tokenizer),
    drop_last=training_args.dataloader_drop_last)

    
    metalearner = MetaLearner(metatraining_args, tokenizer)
    mtl_optimizer = Adam(metalearner.model.parameters(), lr=metatraining_args.mtl_update_lr)
   

    for source_idx, db in enumerate([s1_db]): # , s2_db, s3_db]):

        for step, task_batch in enumerate(db):
            # Meta-Training(FOMAML)
            f = open('log.txt', 'a')
            # print("\n")
            # print(task_batch)
            # print("\n")
            acc, loss = metalearner(task_batch)
            print('Step:', step, '\tTraining Loss | Acc:', loss, " | ",acc)
            f.write(str(acc) + '\n')

        # Fine-tuning on Target Set
        # target_batch = iter(target_train_dataloader).next()
        target_train_loss = []
        target_train_acc = []
        metalearner.model.cuda()
        metalearner.model.train()
        print(metalearner.model.parameters())

        for target_batch in tqdm.tqdm(target_train_dataloader):
            target_batch = metalearner.prepare_inputs(target_batch)
            outputs = metalearner.model(**target_batch)
            loss = outputs[0]
            loss.backward()
            metalearner.outer_optimizer.step()
            metalearner.outer_optimizer.zero_grad()
            target_train_loss.append(loss.item())

            # Compute Acc for target
            logits = F.softmax(outputs[1], dim=1)
            target_label_id = target_batch.get('labels')
            pre_label_id = torch.argmax(logits,dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            target_label_id = target_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id,target_label_id)
            target_train_acc.append(acc)



        print("Target Loss: ", np.mean(target_train_loss))
        print("Target Acc: ", np.mean(target_train_acc))
            
            # end fine tuning
        
    # end MML 
    
    # MTL : Normal fine tuning
    target_finetune_loss = []
    for target_batch in target_train_dataloader:
        metalearner.model.train()
        target_batch = metalearner.prepare_inputs(target_batch)
        outputs = metalearner.model(**target_batch)
        loss = outputs[0]              
        loss.backward()
        mtl_optimizer.step()
        mtl_optimizer.zero_grad()
        target_finetune_loss.append(loss.item())

    print("Target Loss: ", np.mean(target_finetune_loss))

if __name__ == '__main__':
    main()