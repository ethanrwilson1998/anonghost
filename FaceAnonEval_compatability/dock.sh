cd ..

winpty docker run --rm -i -t --gpus all \
-v 'D:\\github\anonghost':/ghost \
-v 'D:\\github\FaceAnonEval\Datasets':/Datasets \
-v 'D:\\github\FaceAnonEval\Anonymized Datasets':'/Anonymized Datasets' \
anonghost:latest