for expected_batchsize in 5000
do

for epsilon in 8
do

for EPOCH in 40
do

for lr in 0.1
do

python main.py --expected_batchsize $expected_batchsize --epsilon $epsilon --EPOCH $EPOCH --lr $lr --log_dir logs 


done
done
done
done

