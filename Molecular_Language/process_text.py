import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
import torch
import pandas as pd
import csv
from Mistral_model import LoraMistralModel
prompt_template = """You are an expert medicinal chemist and cheminformatics scientist.

Your goal:
Given information about ONE small molecule, produce a general, task-agnostic textual description that summarizes its key chemical features and physicochemical profile.

The description MUST:
- NOT assume or mention any specific prediction task or dataset.
- NOT mention any specific endpoint such as solubility value, toxicity label, binding affinity value, etc.
- ONLY use the provided numeric values as they are, do not invent new numbers.
- Use general medicinal chemistry and physical chemistry knowledge.

--------------------
[Molecule input]

SMILES: {SMILES_STRING}

Precomputed descriptors (from RDKit):
- MolecularWeight: {MolecularWeight}
- XLogP: {XLogP}
- TPSA: {TPSA}
- NumHBD: {NumHBD}
- NumHBA: {NumHBA}
- NumRotatableBonds: {NumRotatableBonds}
- FormalCharge: {FormalCharge}
- AromaticRingCount: {AromaticRingCount}

If some fields are not given, treat them as unknown and DO NOT fabricate numeric values.
You may still infer qualitative tendencies from the SMILES if they are obvious (e.g., "contains aromatic rings", "contains tertiary amine").
--------------------

Please analyze this molecule and output a JSON object with the following keys:

- "basic_identity":
    - "name_or_id": short text summarizing name/ID if available.
    - "core_scaffold": short description of the main scaffold (e.g., "benzene ring with substituted amide", "aliphatic amine").
- "physicochemical_profile":
    - "size": brief comment on size based on MolecularWeight and SMILES (e.g., "small", "medium").
    - "lipophilicity": qualitative comment based on XLogP and structural features.
    - "polarity": qualitative comment based on TPSA, heteroatoms, and functional groups.
    - "hydrogen_bonding": comment on HBD/HBA counts and key donor/acceptor groups.
    - "flexibility": comment on NumRotatableBonds and ring systems.
    - "charge_state": typical charge state at physiological pH, if inferrable (e.g., "mostly neutral", "likely positively charged amine", or "uncertain").
- "structural_features":
    - "rings_and_aromaticity": description of aromatic/non-aromatic rings and conjugation patterns.
    - "key_functional_groups": list-like text mentioning important functional groups (e.g., amide, ester, tertiary amine, halogens).
    - "substitution_pattern": comment on how substituents are arranged on the core scaffold.
    - "stereochemistry": description of chiral centers or geometrical isomerism if visible from SMILES; otherwise "not obvious".
- "general_medicinal_chemistry_notes":
    - 2–5 sentences of general, task-agnostic remarks such as:
      - balance between lipophilicity and polarity,
      - potential chemical stability or metabolic soft spots,
      - very rough comments on oral drug-likeness (without referring to any specific rule name),
      - any obviously extreme property (e.g., "very polar", "highly lipophilic", "very large for a typical small-molecule drug").
    - These notes MUST remain generic and MUST NOT refer to any specific biological target, disease, dataset, or endpoint.

Rules:
- Use only information that can be reasonably inferred from the given SMILES and descriptors.
- If something is uncertain, say "uncertain" or "not clear from the given information".
- Do NOT mention any dataset name or prediction task.
- Do NOT invent precise numeric values that are not provided.

Now output the JSON object.
"""
local_model_path="Mistral-7B-Instruct-v0.3"
use_lora=False
model=LoraMistralModel(local_model_path,use_lora=use_lora)
text_list=[]
datasets_path="Pretrain_Datasets\\Zinc15.csv"
df=pd.read_csv(datasets_path)
smiles=df["smiles"].tolist()

with open("text1.csv","w",newline="",encoding="utf-8") as f:
    writer=csv.writer(f)
    writer.writerow(["id","text"])
    for i in range(len(smiles)):
        print(i)
        smile=smiles[i]
        mol=Chem.MolFromSmiles(smile)
        props={}
        props["smiles"]=smile
        props["MolecularWeight"]=Descriptors.MolWt(mol)
        props["XLogP"]=Descriptors.MolLogP(mol)
        props["TPSA"]=rdMolDescriptors.CalcTPSA(mol)
        props["NumHBD"]=rdMolDescriptors.CalcNumHBD(mol)
        props["NumHBA"]=rdMolDescriptors.CalcNumHBA(mol)
        props["NumRotatableBonds"]=rdMolDescriptors.CalcNumRotatableBonds(mol)
        props["FormalCharge"]=Chem.GetFormalCharge(mol)
        props["AromaticRingCount"]=rdMolDescriptors.CalcNumAromaticRings(mol)
        prompt=prompt_template.format(
            SMILES_STRING=props["smiles"],
            MolecularWeight=props["MolecularWeight"],
            XLogP=props["XLogP"],
            TPSA=props["TPSA"],
            NumHBD=props["NumHBD"],
            NumHBA=props["NumHBA"],
            NumRotatableBonds=props["NumRotatableBonds"],
            FormalCharge=props["FormalCharge"],
            AromaticRingCount=props["AromaticRingCount"],
        )
        text=model.template_generate(prompt,max_new_tokens=1024,temperature=0.2,top_p=0.9)
        writer.writerow([i+1,text])