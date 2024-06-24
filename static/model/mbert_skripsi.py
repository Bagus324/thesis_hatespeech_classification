import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.metrics import classification_report

from transformers import AutoModel, XLMForSequenceClassification, BertForSequenceClassification

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy

import csv
    
  
class Multilabelmbert(pl.LightningModule):
    def __init__(self,  
                 labels, 
                 lr = 1e-5,
                 num_classes = 13,
                 dropout = 0.22
                 ) -> None:
        super(Multilabelmbert, self).__init__()
        self.lr = lr
        self.labels = labels
        torch.manual_seed(1)
        random.seed(43)
        self.criterion = nn.BCELoss()

        ks = 3
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.xlm_model = XLMForSequenceClassification.from_pretrained('FacebookAI/xlm-mlm-100-1280', num_labels=num_classes, problem_type="multi_label_classification")
        self.mbert = AutoModel.from_pretrained(
            'google-bert/bert-base-multilingual-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # apply dropout

        self.test_output_step = []
        
        print("MODEL SKRIPSI MBERT")
  
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.auroc_macro = MultilabelAUROC(num_labels = len(self.labels), average = "macro")
        self.f1_macro = MultilabelF1Score(num_labels=len(self.labels), average='macro')
        self.acc = MultilabelAccuracy(num_labels=len(self.labels), average='macro')
        self.auroc_micro = MultilabelAUROC(num_labels = len(self.labels), average = "micro")
        self.f1_micro = MultilabelF1Score(num_labels=len(self.labels), average='micro')
        self.acc_micro = MultilabelAccuracy(num_labels=len(self.labels), average='micro')
        # self.auroc = AUROC(num_classes = 12, task = 'multilabel')


    def forward(self, input_ids, token_type_ids, attention_mask):
        mbert_out = self.mbert(input_ids = input_ids, 
                                   attention_mask = attention_mask, 
                                   token_type_ids = token_type_ids,
                                   return_dict = True)

        hidden_state = mbert_out[0]
        logits = hidden_state[:, 0]
        logits = self.pre_classifier(logits)
        logits = self.tanh(logits)
        logits = self.dropout(logits)
        logits = self.l1(logits)

        logits = self.sigmoid(logits)
        return logits
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch
        x_input_ids, x_token_type_ids, x_attention_mask, y = x_input_ids, x_token_type_ids, x_attention_mask, y

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        # preds = torch.argmax(out, dim = 1)

        #XLM for seq classification
        #out.loss
        #buat sigmoid(out.logits) untuk metrics

  
        loss = self.criterion(out, y.float())
        acc = self.acc(out, y)
 
        metrics = {'train_loss': loss,
                  'train_acc': acc
                }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
     
        return {"loss": loss, "predictions": out, "labels": y}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch
        x_input_ids, x_token_type_ids, x_attention_mask, y = x_input_ids, x_token_type_ids, x_attention_mask, y

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        # preds = torch.argmax(out, dim = 1)

    
          
            

  
        
        loss = self.criterion(out, y.float())
        # acc = self.acc(preds, y.float())
        acc = self.acc(out, y)
        f1_macro = self.f1_macro(out, y)
        auroc_macro = self.auroc_macro(out, y)
        acc_micro = self.acc_micro(out, y)
        f1_micro = self.f1_micro(out, y)
        auroc_micro = self.auroc_micro(out, y)
 
        metrics = {'val_loss': loss,
                  'val_acc': acc,
                  'val_f1_macro_': f1_macro,
                  'val_auroc_macro': auroc_macro,
                  'val_acc_micro': acc_micro,
                  'val_f1_micro': f1_micro,
                  'val_auroc_micro': auroc_micro
                }
        # self.log_dict(metrics, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch
        x_input_ids, x_token_type_ids, x_attention_mask, y = x_input_ids, x_token_type_ids, x_attention_mask, y
        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        loss = self.criterion(out, y.float())

    
        result = {"predict": out.tolist(), "y": y.tolist()}
        self.test_output_step.append(result)
    
    
    
  
        acc = self.acc(out, y)
        f1_macro = self.f1_macro(out, y)
        auroc_macro = self.auroc_macro(out, y)
        acc_micro = self.acc_micro(out, y)
        f1_micro = self.f1_micro(out, y)
        auroc_micro = self.auroc_micro(out, y)

 
        metrics = {'test_loss': loss,
                  'test_acc': acc,
                  'test_f1_macro_': f1_macro,
                  'test_auroc_macro': auroc_macro,
                  'test_acc_micro': acc_micro,
                  'test_f1_micro': f1_micro,
                  'test_auroc_micro': auroc_micro
                }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return out
    
    def on_test_epoch_end(self):
        fieldnames = self.test_output_step[0].keys()
        with open('/kaggle/working/twitter_sentimen/skripsi/output/test_results.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.test_output_step)
    # def training_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []
    #     loss = []
    #     for output in outputs:
    #         for out_lbl in output["labels"]:
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"]:
    #             predictions.append(out_pred)
    #         loss.append(output['loss'])
    #     loss_mean = torch.mean(torch.Tensor(loss))
    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)
    #     # print(predictions.shape)
    #     # print(labels.shape)
    #     print(f"====================Loss Of Epoch==================== {self.current_epoch} : {loss_mean}")
    #     aurocs_label = self.auroc_label(predictions, labels)
    #     f1_labels = self.f1_labels(predictions, labels)
    #     acc_label = self.acc_labels(predictions, labels)
    #     print("====================auroc_score====================")
    #     for i, gits in enumerate(aurocs_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_roc_auc/Train", gits, self.current_epoch)
    
    #     print("====================f1_score====================")
    #     for i, gits in enumerate(f1_labels) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_F1/Train", gits, self.current_epoch)
    #     print("====================Accuracy====================")
    #     for i, gits in enumerate(acc_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_Acc/Train", gits, self.current_epoch)
        # for i, name in enumerate(self.labels):
        #     class_roc_auc = self.auroc_label(predictions[:, i], labels[:, i])
        #     print(f"{name} \t : {class_roc_auc}")
        #     self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    # def on_predict_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []
    #     loss = []
    #     for output in outputs:
    #         for out_lbl in output["labels"]:
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"]:
    #             predictions.append(out_pred)
    #         loss.append(output['loss'])
    #     loss_mean = torch.mean(torch.Tensor(loss))
    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)
    #     # print(predictions.shape)
    #     # print(labels.shape)
    #     print(f"Loss Of Epoch {self.current_epoch} : {loss_mean}")
    #     aurocs_label = self.auroc_label(predictions, labels)
    #     auroc_macro = self.auroc_macro(predictions, labels)
    #     accuracy = self.acc(predictions, labels)
    #     self.log("Accuracy_test", accuracy, prog_bar=True, on_epoch=True)
    #     self.log("Auroc_Macro_test", auroc_macro, prog_bar=True, on_epoch=True)
    #     f1_labels = self.f1_labels(predictions, labels)
    #     f1_macro = self.f1_macro(predictions, labels)
    #     acc_label = self.acc_labels(predictions, labels)
    #     self.log("F1_test", f1_macro, prog_bar=True, on_epoch=True)
    #     print("auroc score")
    #     for i, gits in enumerate(aurocs_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_roc_auc/Test", gits, self.current_epoch)
    
    #     print("f1 score")
    #     for i, gits in enumerate(f1_labels) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_F1/Test", gits, self.current_epoch)
    #     print("Accuracy")
    #     for i, gits in enumerate(acc_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_Acc/Test", gits, self.current_epoch)