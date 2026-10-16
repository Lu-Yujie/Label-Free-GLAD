"""Code-problem evidence (no labels, no training): GLASS's TextEncoder.encode_texts uses last-position pooling
(outputs.last_hidden_state[:, -1, :]) with the tokenizer's default RIGHT padding. In a batch, every text shorter than
the longest one is then represented by the hidden state at a PADDING position. Check with two real MUTAG GraphDP
strings: embedding of the shorter text alone vs. inside a batch with the longer one."""
import json, os, os.path as osp, pickle, sys, time
import torch
HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import run_method as RM
G = RM.load_glass()
torch.set_num_threads(4)
c = pickle.load(open(RM.feature_cache_path("MUTAG"), "rb"))
texts = c["graph_dp"]
L = sorted(range(len(texts)), key=lambda i: len(texts[i]))
short, long_ = texts[L[0]], texts[L[-1]]
cfg = G.cfg.GLASSConfig(dataset_name="MUTAG", device="cpu"); cfg.mrl_mode = "native"; cfg.text_model_path = RM.TEXT_MODEL
te = G.te.TextEncoder(cfg, text_device="cpu")
te.model = te.model.float()   # CPU check only: fp32 (GLASS runs bf16 on GPU); the padding behaviour does not depend on dtype
inst = G.gdp.get_instruction(domain="molecular", mode="train")
t0 = time.time()
alone = te.encode_texts([short], [inst], device="cpu")[0]
batched = te.encode_texts([short, long_], [inst, inst], device="cpu")[0]
enc = te.tokenizer([f"Instruct: {inst}\nQuery: {t}" for t in (short, long_)], padding=True, truncation=True,
                   max_length=cfg.text_max_length, return_tensors="pt")
cos = torch.nn.functional.cosine_similarity(alone[None], batched[None]).item()
res = dict(padding_side=te.tokenizer.padding_side, n_tokens=[int(x) for x in enc["attention_mask"].sum(1)],
           last_token_ids=[int(x) for x in enc["input_ids"][:, -1]], pad_token_id=te.tokenizer.pad_token_id,
           cos_alone_vs_batched=cos, max_abs_diff=float((alone - batched).abs().max()),
           seconds=round(time.time() - t0, 1), note="CPU float32 copy of the bf16 model; same code path as GLASS (TextEncoder.encode_texts)")
print(json.dumps(res, indent=1))
json.dump(res, open(osp.join(HERE, "verify", "padding_check.json"), "w"), indent=1)
