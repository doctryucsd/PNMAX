rm -f ./*.csv
dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for i in $dir
do
    # add header
    cat ../script/head_short.csv > ./${i}.csv
    cat ${i}/csv/*.csv >> ./${i}.csv
done
cp ../script/combine_e2e.py ./
python combine_e2e.py
find . -maxdepth 1 -type f -name "*.csv" ! -name "*combine*" -exec rm -f {} +