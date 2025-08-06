import torch
import esm
import pandas as pd
from tqdm import tqdm

def get_cls_embeddings(sequences, model_name="esm2_t33_650M_UR50D", batch_size=8):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.cuda()  # ✅ Use GPU
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]
        batch_data = [(f"protein{i+j}", seq) for j, seq in enumerate(batch_seqs)]
        labels, strs, toks = batch_converter(batch_data)
        toks = toks.cuda()

        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        token_reps = out["representations"][33]

        for j in range(len(batch_seqs)):
            cls_emb = token_reps[j, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    return all_embeddings

def save_embeddings(input_csv, output_path):
    df = pd.read_csv(input_csv)
    df.fillna('', inplace=True)

    sequences = df['Sequence'].tolist()
    ids = df['ID'].tolist()

    print(f"Extracting embeddings for {len(sequences)} proteins...")

    cls_embeddings = get_cls_embeddings(sequences)

    embed_df = pd.DataFrame(cls_embeddings)
    embed_df.insert(0, "ID", ids)  # ✅ Keep IDs as the first column

    embed_df.to_csv(output_path, index=False)
    print(f"✅ Saved CLS embeddings to: {output_path}")
