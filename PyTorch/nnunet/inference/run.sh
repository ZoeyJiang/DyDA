export nnUNet_raw_data_base="/dataset/raw/raw_data/Task011_mixed_ACDC_myo_emidec/"
export nnUNet_preprocessed="/dataset/preprocessed/Task011_mixed_ACDC_myo_emidec/"
export RESULTS_FOLDER="/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/lambda07/"

python predict_simple.py -m 3d_fullres


