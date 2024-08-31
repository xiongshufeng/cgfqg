python ner.py --model_name_or_path ./pretrained_bert/bert-base-chinese --do_train True --do_eval True --do_test True --max_seq_length 256 --train_file ./data/csfqg/train.txt --eval_file ./data/csfqg/dev.txt --test_file ./data/csfqg/test.txt --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 10 --do_lower_case --logging_steps 200 --need_birnn True --rnn_dim 256 --clean True --output_dir ./output