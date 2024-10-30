cd ..

for ds in CelebA_test lfw
do

    for e in 0 1 10 100 1000 10000
    do

        for t in 0 30 60 90 120 150 180
        do

            od=$ds"_eps"$e"_theta"$t
            python3.8 anonymize_dataset.py --input_dir /Datasets/$ds --output_dir /Anonymized\ Datasets/$od --epsilon $e --theta $t
        done
    done
done