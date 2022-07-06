# ner_bio_phi_for_phrases
This is a tweaked version of self-supervised NER for tagging phrases

# Installations
pip install -r requirements.txt

apt-get udpate

apt-get install git-lfs

cd ner_bio_phi_for_phrases

git lfs install

git lfs pull --include bbc/bbc_labels.txt 

git lfs pull --include bio/a100_labels.txt 

# Test install
./test.sh

