# HiWestSum

This is the repository for Final Year Project: Hierarchical Document Representation for Summarization submitted to Nanyang Technological University(https://hdl.handle.net/10356/157571).
Much of the codes are referenced from [github repository from NLPYang](https://github.com/nlpyang/PreSumm) used for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345). Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Data Preparation for CNN/DM
Please refer to the [referenced repository](https://github.com/nlpyang/PreSumm) to download or preprocess the required dataset. Please note that the raw dataset should be preprocessed differently for ALBERT because of the difference in vocab file.

Please note that there are some added codes for processing ALBERT since the vocab file is different for ALBERT(vocab size 30000) as compared to BERT or DistilBERT (vocab size 30522). The only difference is in step 5:
####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log -pretrained_model albert
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)
* `-pretrained_model` is added with arguments `bert` or `albert` or `distilbert` options where `albert` will have a difference in preprocessing in our repo.


## Environment & Packages
Please refer to requirements.txt
Important packages:
<ul>
<li>torch==1.1.0</li>
<li>transformers==4.16.2</li>
<li>sentencepiece==0.1.96</li>
<li>pyrouge</li>
<li>tensorboardX==1.9</li>
<li>multiprocess==0.70.9</li>
<li>pytorch-transformers==1.2.0</li>
<li>nltk==3.7</li>
  <li>sentencepiece==0.1.96</li>
</ul>

## Model Training
### HIWESTSUM - ALBERT
```
python train.py -task ext -mode train -bert_data_path ALBERT_DATA_PATH -ext_dropout 0.1 -model_path INTENDED_CHECKPOINT_SAVED_DIR -lr 2e-3 -visible_gpus 0,1,2,3 -report_every 100 -save_checkpoint_steps 1000 -train_steps 10000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 2000 -max_pos 512 -other_bert albert -batch_size 300 -doc_weight 0.4 -extra_attention False -sharing True
```
Arguments:
<ul>
  <li><b>-task ext</b> (fixed to extractive summarization only for HIWESTSUM)</li>
  <li><b>-mode train</b> (train/validate/test)</li>
  <li><b>-bert_data_path ALBERT_DATA_PATH</b> (IMPORTANT, change ALBERT_DATA_PATH to your preprocessed data directory, eg: ../albert_data/albert_data)</li>
  <li><b>-ext_dropout 0.1</b> (dropout rate)</li>
  <li><b>-model_path INTENDED_CHECKPOINT_SAVED_DIR</b> (IMPORTANT, change INTENDED_CHECKPOINT_SAVED_DIR to the intended output directory for model checkpoints, eg: ./hiwest/albert0.4) </li>
  <li><b>-lr 2e-3</b> (learning rates, suggested learning rates for BERT are <a href="https://arxiv.org/pdf/1706.03762.pdf">5e-5, 3e-5, 2e-5</a>)</li>
  <li><b>-visible_gpus 0,1,2,3</b> (IMPORTANT, set the GPU(s) to be used)</li>
  <li><b>-save_checkpoint_steps 1000</b> (set checkpoint to be saved every X step, X is the argument)</li>
  <li><b>-report_every 100</b> (set progress to be reported by logging every X step, X is the argument)</li>
  <li><b>-train_steps 10000</b> (set total training steps)</li>
  <li><b>-log_file ../logs/ext_bert_cnndm</b> (log file location) </li>
  <li><b>-max_pos 512</b> (set max position.For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.) </li>
  <li><b>-other_bert albert</b> (IMPORTANT, set the pretrained model to be used, albert/bert/distilbert) </li>
  <li><b>-batch_size 300<b> (set batch size) </li>  
  <li><b>-doc_weight 0.4</b> (IMPORTANT, set the document weight. document_weight + sent_weight = 1 </li>
   <li><b>-sharing True</b> (IMPORTANT, determine if weights are shared between the bert layer and extractive layers. </li>
</ul>

## Model Evaluation
```
python train.py -task ext -mode validate -batch_size 300 -test_batch_size 500 -bert_data_path ../albert_data/albert_data -log_file /home/students/s121md102_06/bertsum_experiment/PreSummWithMobileBert/logs/val_hiwest_distilbert_cnndm -model_path ./hiwest/albert0.4 -sep_optim true -use_interval true -visible_gpus 0,1,2,3 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/hiwestsum_al0.4 -other_bert albert -architecture hiwest -doc_weight 0.4 -extra_attention False -sharing True
```

## Architecture of HiWestSum
![architecture](/architecture.JPG)
Most extractive summarization models usually employ a hierarchical encoder for document summarization. However, these extractive models are solely using document-level information to classify and select sentences which may not be the most effective way.
In addition, most state-of-the-art (SOTA) models will be using huge number of parameters to learn from a large amount of data, and this causes the computational costs to be very
expensive.

In this project, Hierarchical Weight Sharing Transformers for Summarization (HIWESTSUM) is proposed for document summarization. HIWESTSUM is very light in weight with parameter size over 10 times smaller than current existing models that [finetune BERT for summarization](https://arxiv.org/abs/1908.08345). Moreover, the proposed model is faster than SOTA models with shorter training and inference time. It learns effectively from both sentence
and document level representations with weight sharing mechanisms.

By adopting weight sharing and hierarchical learning strategies, it is proven in this project that the proposed model HIWESTSUM may reduce the usage of computational resources
for summarization and achieve comparable results as SOTA models when trained on smaller datasets.
