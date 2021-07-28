#!/bin/bash
# Due to the limited resources of codeocean, please run the code in your own
# GPU server with dataset downloaded from Google driver.
labels=reddit,facebook,NeteaseMusic,twitter,qqmail,instagram,weibo,iqiyi,
labels+=imdb,TED,douban,amazon,youtube,JD,youku,baidu,google,tieba,taobao,bing

for n in {1..8}; do
    echo 'Start instance '$n' of lstm and bilstm'
    python -W ignore ./train.py --filename 20_header_payload_all.hdf5 \
    --epochs 30 --labels $labels \
    --batch_size 128 --gpu 0 --gamma 1 --embedding_dim 64 \
    --model EBSNN_LSTM --segment_len 16 \
    --no_bidirectional \
    --log_filename log_20/log_train_rebuttal_lstm_$n.txt \
    --shuffle 2>&1 >/dev/null &
    pids[$(( $n * 2 - 1 ))]=$!
    #################################
    python -W ignore ./train.py --filename 20_header_payload_all.hdf5 \
    --epochs 30 --labels $labels \
    --batch_size 128 --gpu 0 --gamma 1 --embedding_dim 64 \
    --model EBSNN_LSTM --segment_len 16 \
    --log_filename log_20/log_train_rebuttal_bilstm_$n.txt \
    --shuffle 2>&1 >/dev/null &
    pids[$(( $n * 2 ))]=$!
    #################################
    mod=$(( $n % 2 ))
    if (( mod == 0 )); then
        for pid in ${pids[*]}; do
            echo 'Wait for '$pid
            wait $pid
        done
        unset pids
    fi
done