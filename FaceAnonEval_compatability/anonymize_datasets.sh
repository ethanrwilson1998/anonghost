cd ..

for ds in CelebA_test lfw
do

    for e in -1 0 1 10 100 1000 10000
    do

        for t in 0 30 45 60 90 120 135 150 180
        do

            od="anonghost/"$ds"_eps"$e"_theta"$t
            python3.8 anonymize_dataset.py --input_dir /Datasets/$ds --output_dir /Anonymized\ Datasets/$od --epsilon $e --theta $t
        done
    done
done
