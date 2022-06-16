python3 batch_main.py -input sentences.txt -output results.txt
apt-get install -y jq
cat results.txt | jq > readable_results.txt
