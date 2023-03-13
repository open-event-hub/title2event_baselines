# Title2Event
This is the repository for the paper [Title2Event: Benchmarking Open Event Extraction with a Large-scale Chinese Title Dataset](https://aclanthology.org/2022.emnlp-main.437/)
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
## Citation
```
@inproceedings{deng-etal-2022-title2event,
    title = "{T}itle2{E}vent: Benchmarking Open Event Extraction with a Large-scale {C}hinese Title Dataset",
    author = "Deng, Haolin  and
      Zhang, Yanan  and
      Zhang, Yangfan  and
      Ying, Wangyang  and
      Yu, Changlong  and
      Gao, Jun  and
      Wang, Wei  and
      Bai, Xiaoling  and
      Yang, Nan  and
      Ma, Jin  and
      Chen, Xiang  and
      Zhou, Tianhua",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.437",
    pages = "6511--6524",
    abstract = "Event extraction (EE) is crucial to downstream tasks such as new aggregation and event knowledge graph construction. Most existing EE datasets manually define fixed event types and design specific schema for each of them, failing to cover diverse events emerging from the online text. Moreover, news titles, an important source of event mentions, have not gained enough attention in current EE research. In this paper, we present Title2Event, a large-scale sentence-level dataset benchmarking Open Event Extraction without restricting event types. Title2Event contains more than 42,000 news titles in 34 topics collected from Chinese web pages. To the best of our knowledge, it is currently the largest manually annotated Chinese dataset for open event extraction. We further conduct experiments on Title2Event with different models and show that the characteristics of titles make it challenging for event extraction, addressing the significance of advanced study on this problem. The dataset and baseline codes are available at https://open-event-hub.github.io/title2event.",
}

```
