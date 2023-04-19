# https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
# we use the multilingual version because PILE also has multiple languages
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class ToxicCommentTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None,  model_name='bert-base-uncased', device="None"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:

            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # for i, name in enumerate(LABEL_COLUMNS):

        #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
        #     self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
         )

        return dict(
        optimizer=optimizer,
        lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
         )
        )

    def predict(self, sequences, device="cuda", batch_size=32):
        if not isinstance(sequences, list):
            sequences = sequences.tolist()

        predictions = []

        for start in range(0, len(sequences), batch_size):
            end = min(start+batch_size, len(sequences))

            batch_sequences = sequences[start:end]
            encoding = self.bert_tokenizer.batch_encode_plus(
                batch_sequences,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            _, test_prediction = self(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            predictions.extend(test_prediction.cpu().detach().numpy())

        return np.vstack(predictions)