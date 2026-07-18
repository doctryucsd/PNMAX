rm -f ./*.csv
dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for i in $dir
do
    cat ${i}/csv/*.csv > ./${i}.csv
done