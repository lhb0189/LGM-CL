# MultiModal-Contrastive-GNN
This project is implemented in Python 3.8. All required libraries and their versions for running this code are specified in `requirements.txt` and `environment.yml`. The prompt templates used in the paper for large language model input construction are provided in `Prompt_template.txt`. We provided the LLM-augmented textual data derived from the pretraining dataset and the ten benchmark datasets in 'Pretrain_Text_Datasets' and `Process_Text_Datasets`, respectively. If you wish to use the provided templates to augment SMILES strings into corresponding textual descriptions, you can import the target SMILES into the `process` module under `Molecular_Language`. 
The files required for `Mistral-7B-Instruct-v0.3` and `deberta-v3-base` can be downloaded from their official websites or via the following Google Drive links:
- Mistral-7B-Instruct-v0.3: https://drive.google.com/drive/folders/1AABdP7gH2u9vLMlbqkp-bIkmTeJl0pZC?usp=drive_link
- DeBERTa-v3-base: https://drive.google.com/drive/folders/1KfnTP1WVnqNwhELc_qICTza5znLMzmIa?usp=drive_link
The pretrained model parameters used in the paper are stored in `Pretrain_model`.
The following model parameters were obtained from fine-tuning experiments with different random seeds and were used in the paper：
-Bace：https://drive.google.com/drive/folders/1urjw3ZaRlkY1yvEyo2EfdpqxFEoTzCJp?usp=drive_link
-BBBP：https://drive.google.com/drive/folders/1VDcvUdnogQaZrQDXiXAnWm7BrbV2fuPV?usp=drive_link
-Clintox：https://drive.google.com/drive/folders/19KJn4wjZYfU5P--gseQP4W0LrlmhFadg?usp=drive_link
-Tox21：
-ESOL：https://drive.google.com/drive/folders/1o9WI8OwUZJ7ydQpZAFI2oJ4zKdYfUU1C?usp=drive_link
-FreeSolv：https://drive.google.com/drive/folders/1pTkMC3rBaGaWS26CYpZsaNCFPnkrTEnU?usp=drive_link
-Lipophilicity：https://drive.google.com/drive/folders/1ipw-Wjr7QKiPuh0bZ5-CDH2q3dISOHeR?usp=drive_link
