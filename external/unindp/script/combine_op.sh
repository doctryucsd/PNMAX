rm -f ./*.csv
dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for i in $dir
do
    echo Name,Baseline,UniNDP,SpeedUp > ./${i}.csv
    for j in ${i}/csv/*
    do
        # echo ${j}
        while read line
        do
            name=$(echo ${line} | cut -d , -f 1)
            # optype=$(echo ${line} | cut -d , -f 2)
            optimal=$(echo ${line} | cut -d , -f 3)
            baseline=$(echo ${line} | cut -d , -f 8)
            echo ${name},${baseline},${optimal},$(awk 'BEGIN{printf("%.2f",'$baseline'/'$optimal')}') >> ./${i}.csv
            # break
        done < ${j}
        # cat ../script/head_short.csv > ./${i}.csv
        # cat ${i}/csv/*.csv >> ./${i}.csv
    done
done