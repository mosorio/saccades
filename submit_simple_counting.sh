# #!/bin/bash

# # * Simple Counting *
#  save activations
python3 main.py --model_type=rnn_classifier2stream --save_act --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=19 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=0 --grid=6 

echo Simple Counting