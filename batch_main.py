import time
import torch
import string
import pdb
import argparse

from transformers import BertTokenizer, BertForMaskedLM
import BatchInference as bd
import batched_main_NER as ner
import aggregate_server_json as aggr
import json


DEFAULT_TOP_K = 20
SPECIFIC_TAG=":__entity__"
DEFAULT_MODEL_PATH="ajitrajasekharan/biomedical"
DEFAULT_RESULTS="results.txt"


def perform_inference(text,bio_model,ner_bio,phi_model,ner_phi,aggr_ner):
    print("Getting predictions from BIO model...")
    bio_descs = bio_model.get_descriptors(text,None)
    print("Computing BIO results...")
    bio_ner = ner_bio.tag_sentence_service(text,bio_descs)
    print("Getting predictions from PHI model...")
    phi_results = phi_model.get_descriptors(text,None)
    print("Computing PHI results...")
    phi_ner = ner_phi.tag_sentence_service(text,phi_results)
    combined_arr = [json.loads(bio_ner),json.loads(phi_ner)]
    aggregate_results = aggr_ner.fetch_all(text,combined_arr)
    return aggregate_results
  
   
def process_input(results):
    try:
        input_file = results.input
        output_file = results.output
        print("Initializing BIO module...")
        bio_model = bd.BatchInference("bio/desc_a100_config.json",'ajitrajasekharan/biomedical',False,False,DEFAULT_TOP_K,True,True,       "bio/","bio/a100_labels.txt",False)
        ner_bio = ner.UnsupNER("bio/ner_a100_config.json")
        phi_model = bd.BatchInference("bbc/desc_bbc_config.json",'bert-base-cased',False,False,DEFAULT_TOP_K,True,True,       "bbc/","bbc/bbc_labels.txt",False)
        ner_phi = ner.UnsupNER("bbc/ner_bbc_config.json")
        print("Initializing Aggregation module...")
        aggr_ner = aggr.AggregateNER("./ensemble_config.json")
        wfp = open(output_file,"w")
        with open(input_file) as fp:
            for line in fp:
                text_input = line.strip().split()
                print(text_input)
                text_input = [t + ":__entity__" for t in text_input]  
                text_input = ' '.join(text_input)
                start = time.time()
                results = perform_inference(text_input,bio_model,ner_bio,phi_model,ner_phi,aggr_ner)
                print(json.dumps(results["ensembled_ner"]))
                #pdb.set_trace()
                print(f"prediction took {time.time() - start:.2f}s")
                wfp.write(json.dumps(results))
                wfp.write("\n\n")
        wfp.close()
    except Exception as e:
        print("Some error occurred in batch processing") 
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch handling of NER ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input", required=True,help='Input file with sentences')
    parser.add_argument('-output', action="store", dest="output", default=DEFAULT_RESULTS,help='Output file with sentences')
    results = parser.parse_args()
    process_input(results)
