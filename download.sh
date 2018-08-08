# Download dataset splits
wget https://filebox.ece.vt.edu/~ram21/Projects/neuron-importance-zsl/data.zip
unzip data.zip

# Download dataset images
# AWA2
cd data/AWA2/
wget https://cvml.ist.ac.at/AwA2/AwA2-data.zip
unzip AwA2-data.zip
cd ../..
# CUB
cd data/CUB/
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar -xvzf images.tgz

# Download pretrained finetuned models and domain2alpha_checkpoints
wget https://filebox.ece.vt.edu/~ram21/Projects/neuron-importance-zsl/ckpt.zip
unzip ckpt.zip
