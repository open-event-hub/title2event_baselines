# Title2Event
This is the repository for the paper [Title2Event: Benchmarking Open Event Extraction with a Large-scale Chinese Title Dataset](https://arxiv.org/abs/2211.00869)
## Quick Start
### Download the dataset
You can obtain the dataset from our [webpage](https://open-event-hub.github.io/title2event/) \
Note that the dataset is provided in both `csv` and `json` format, but currently the baseline code reads `csv` files by default. \
You can also find `tagging_train.csv`,`tagging_dev.csv` ,`tagging_test.csv`, these files contain the `BIO` labels needed to train tagging-based models, and are used by the `SeqTag` model.     
### Requirements
The code is modified from [examples of huggingface transformers](https://github.com/huggingface/transformers/tree/main/examples) \
In your preferred environment, run
```
pip3 install -r requirements.txt
```
### Trigger Extraction
#### Sequence-tagging model
Note that the trigger prediction file is needed for pipeline inference.
```
cd seqtag
bash run_trigger_extraction.sh
```
### Argument Extraction
All the following scripts will output two files: \
**arg_predictions.csv**: the model predictions with golden triggers \
**pipeline_predictions.csv**: the model predictions given the triggers predicted by the Trigger Extraction model \
The above files are used in **Evaluation**
#### Sequence-tagging model
```
cd seqtag
bash run_argument_extraction.sh
```
#### Span MRC model
```
cd mrc
bash run_spanmrc.sh
```
#### Seq2Seq MRC model
```
cd mrc
bash run_seq2seqmrc.sh
```

### Evaluation
```
python3 evaluate.py -f [path of file1] [path of file2] ...
```