"""Train the BertSumExt-style sentence scorer on a CNN/DM subset.

Steps:
 1. stream a train/validation subset of CNN/DM (cached to data_cache/)
 2. split sentences, run the frozen BERT encoder once, cache each sentence's
    [CLS] vector together with greedy-oracle labels (data_cache/emb_*.pt)
 3. train the 2-layer inter-sentence Transformer head with BCE loss
 4. after each epoch, select top-3 (trigram blocking) on validation, compute
    ROUGE vs the highlights, keep the best checkpoint -> checkpoints/ext_head.pt

Usage:  python -m src.train_extractive --train-n 3000 --val-n 300
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DATA_CACHE, ENCODER_NAME, EXT_CHECKPOINT, EXT_POOLING, RESULTS, get_device
from src.data import highlights_to_sents, load_cnndm, split_sentences
from src.extractive import BertSumEncoder, SentenceScorerHead, select_sentences
from src.metrics import rouge_scores
from src.oracle import oracle_labels

log = logging.getLogger("train")


def build_features(rows: list[dict], encoder: BertSumEncoder, tag: str, batch_size: int) -> list[dict]:
    safe = encoder.name.replace("/", "--") + "_" + encoder.pooling.replace("+", "-")
    path = DATA_CACHE / f"emb_{safe}_{tag}.pt"
    if path.exists():
        log.info("Loading cached embeddings %s", path.name)
        return torch.load(path, weights_only=False)
    docs = [split_sentences(r["article"]) for r in rows]
    vecs: list[torch.Tensor] = []
    step = 64
    for i in tqdm(range(0, len(docs), step), desc=f"Encoding {tag}", unit="x64 docs"):
        vecs += encoder.encode(docs[i:i + step], batch_size=batch_size)
    feats = []
    for r, sents, v in zip(rows, docs, vecs):
        n = v.shape[0]
        if n == 0:
            continue
        kept = sents[:n]
        feats.append({"id": r["id"], "emb": v.half(),
                      "labels": torch.tensor(oracle_labels(kept, r["highlights"]), dtype=torch.float),
                      "sents": kept, "ref": r["highlights"]})
    torch.save(feats, path)
    return feats


def collate(batch: list[dict]):
    S = max(f["emb"].shape[0] for f in batch)
    H = batch[0]["emb"].shape[1]
    x = torch.zeros(len(batch), S, H)
    y = torch.zeros(len(batch), S)
    m = torch.zeros(len(batch), S, dtype=torch.bool)
    for i, f in enumerate(batch):
        n = f["emb"].shape[0]
        x[i, :n] = f["emb"].float()
        y[i, :n] = f["labels"]
        m[i, :n] = True
    return x, y, m


@torch.no_grad()
def evaluate_head(head, feats, device, batch_size=64) -> dict:
    head.eval()
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for i in range(0, len(feats), batch_size):
        chunk = feats[i:i + batch_size]
        x, _, m = collate(chunk)
        probs = torch.sigmoid(head(x.to(device), m.to(device))).cpu()
        for j, f in enumerate(chunk):
            n = f["emb"].shape[0]
            sel, _ = select_sentences(f["sents"], probs[j, :n].tolist(), 3, True)
            r = rouge_scores([f["sents"][k] for k in sel], highlights_to_sents(f["ref"]))
            for k in agg:
                agg[k].append(r[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def baseline_scores(feats) -> dict:
    """Lead-3 and oracle ROUGE on the same (window-truncated) validation sentences."""
    out = {}
    for name, pick in [("lead3", lambda f: [0, 1, 2]),
                       ("oracle", lambda f: [i for i, l in enumerate(f["labels"].tolist()) if l > 0])]:
        rs = [rouge_scores([f["sents"][i] for i in pick(f) if i < len(f["sents"])],
                           highlights_to_sents(f["ref"])) for f in feats]
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in ("rouge1", "rouge2", "rougeL")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-n", type=int, default=3000)
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--encoder", default=ENCODER_NAME)
    ap.add_argument("--pooling", default=EXT_POOLING, choices=["cls", "cls+mean"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encode-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    t_start = time.time()
    log.info("Device: %s | encoder: %s", device, args.encoder)

    train_rows = load_cnndm("train", args.train_n)
    val_rows = load_cnndm("validation", args.val_n)
    t_data = time.time()
    log.info("Loaded %d train / %d val articles in %.1fs", len(train_rows), len(val_rows), t_data - t_start)

    encoder = BertSumEncoder(args.encoder, device=device, pooling=args.pooling)
    log.info("Segment embeddings (alternating 0/1): %s", encoder.use_segments)
    train_feats = build_features(train_rows, encoder, f"train_{args.train_n}", args.encode_batch_size)
    val_feats = build_features(val_rows, encoder, f"validation_{args.val_n}", args.encode_batch_size)
    hidden = encoder.out_dim
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    t_enc = time.time()
    log.info("Features ready in %.1fs", t_enc - t_data)

    base = baseline_scores(val_feats)
    log.info("Val Lead-3 %s | Oracle %s", base["lead3"], base["oracle"])

    head = SentenceScorerHead(in_dim=hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train_feats) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    history, best, best_epoch = [], -1.0, -1
    for epoch in range(1, args.epochs + 1):
        head.train()
        random.shuffle(train_feats)
        losses = []
        for i in range(0, len(train_feats), args.batch_size):
            x, y, m = collate(train_feats[i:i + args.batch_size])
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = (bce(head(x, m), y) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = evaluate_head(head, val_feats, device)
        avg = (val["rouge1"] + val["rouge2"] + val["rougeL"]) / 3
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val})
        log.info("epoch %d  loss %.4f  val R1 %.2f R2 %.2f RL %.2f", epoch, np.mean(losses),
                 val["rouge1"], val["rouge2"], val["rougeL"])
        if avg > best:
            best, best_epoch = avg, epoch
            torch.save({"encoder_name": args.encoder, "pooling": args.pooling, "head_config": head.config,
                        "state_dict": {k: v.detach().half().cpu() for k, v in head.state_dict().items()},
                        "epoch": epoch, "val": val, "train_n": len(train_feats)}, EXT_CHECKPOINT)

    summary = {"device": device, "encoder": args.encoder, "pooling": args.pooling, "lr": args.lr,
               "batch_size": args.batch_size, "train_n": len(train_feats),
               "val_n": len(val_feats), "epochs": args.epochs, "best_epoch": best_epoch,
               "best_val": history[best_epoch - 1], "val_lead3": base["lead3"],
               "val_oracle": base["oracle"], "history": history,
               "seconds": {"data": t_data - t_start, "encode": t_enc - t_data,
                           "train": time.time() - t_enc, "total": time.time() - t_start}}
    (RESULTS / "train_log.json").write_text(json.dumps(summary, indent=2))
    log.info("Best epoch %d -> %s (%.1f MB). Total %.1f min", best_epoch, EXT_CHECKPOINT,
             EXT_CHECKPOINT.stat().st_size / 1e6, (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
