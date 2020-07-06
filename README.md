

### Pun Detection and Location

```
“The Boating Store Had Its Best Sail Ever”: Pronunciation-attentive Contextualized Pun Recognition
Yichao Zhou, Jyun-yu Jiang, Jieyu Zhao, Kai-Wei Chang and Wei Wang
Computer Science Department, University of California, Los Angeles
```

# Requirements

- `python3`
- `pip3 install -r requirements.txt`

# Run

```Bash
python run_ner.py 
--data_dir=data/ 
--bert_model=bert-base-cased 
--task_name=ner 
--output_dir=out 
--max_seq_length=128 
--do_train 
--num_train_epochs 5 
--do_eval --warmup_proportion=0.4
```

# Inference



