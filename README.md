# dynamic-hand-gestures-classification

Anonymous repository to host code and data to run the 3D hand gestures classification pipeline based on Vispy and a ResNet-50 trained with Fast.ai


## How to perform online inference on the SFINGE 3D dataset

- tested with Ubuntu 18.04 with CUDA already installed - inference performed on CPU (slower)

cd /tmp
git clone git@github.com:SFINGE3D/DatasetV1.git
mkdir /tmp/dynamic-hand-gestures-venv
python3 -m venv /tmp/dynamic-hand-gestures-venv
source /tmp/dynamic-hand-gestures-venv/bin/activate
git clone git@github.com:dynamic-hand-gestures-classification/dynamic-hand-gestures-classification.git
pip install --upgrade pip
cd /tmp/dynamic-hand-gestures-classification/
pip install -r requirements.txt


cd /tmp/dynamic-hand-gestures-classification/utilities/
./conversion.py --filename /tmp/DatasetV1/Sequences/3.txt --csv-separator=,


If you get CUDA errors such:
ImportError: libcudart.so.9.0: cannot open shared object file: No such file or directory
then downgrade pytorch with:

pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html


cd /tmp/dynamic-hand-gestures-classification
unset LD_LIBRARY_PATH

./dynamic-hand-gestures.py ./utilities/unknown-3.csv.xz --dataset-path ./utilities/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-21_15.47.18-SFINGE3D-dataset-transfer-learning-from-our-dataset-data-augmentation-with-partial-gestures-and-noise.pkl --cuda-device cpu --inference-every-n-frames 20 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --fps 10 --save-image-only-when-prob-greater-than 0.98



