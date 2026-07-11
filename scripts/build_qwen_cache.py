from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_comments_csv, read_text_file


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--index_csv', default='data/index.csv')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--output_index_csv', required=True)
    p.add_argument('--text_mode', choices=['asr', 'asr_comments', 'asr_comments_skg'], default='asr_comments_skg')
    p.add_argument('--target_column', choices=['text_feature_path', 'skg_cache_path'], default='skg_cache_path')
    p.add_argument('--model_path', default='Qwen/Qwen3-Embedding-0.6B')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--skg_json', default='data/skg/mined_skg_chinese_with_label.json')
    p.add_argument('--max_matches', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def load_skg(path: str, mode: str) -> dict:
    if mode != 'asr_comments_skg':
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('SKG JSON must map an attribute to sentiment/polarity entries')
    return value


def match_triples(text: str, skg: dict, limit: int) -> list[str]:
    matches = []
    for attribute, entries in skg.items():
        if str(attribute) not in text:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            if isinstance(entry, (list, tuple)):
                sentiment = entry[0] if entry else ''
                polarity = entry[1] if len(entry) > 1 else ''
            elif isinstance(entry, dict):
                sentiment = entry.get('sentiment', entry.get('value', ''))
                polarity = entry.get('polarity', entry.get('label', ''))
            else:
                sentiment, polarity = entry, ''
            matches.append(f'{attribute} | {sentiment} | {polarity}')
            if len(matches) >= limit:
                return matches
    return matches


def build_text(row: pd.Series, mode: str, skg: dict, max_matches: int) -> str:
    text = read_text_file(row.get('asr_text_path'))
    if mode in {'asr_comments', 'asr_comments_skg'}:
        text = (text + ' ' + ' '.join(read_comments_csv(row.get('comments_path')))).strip()
    if mode == 'asr_comments_skg':
        text = (text + '\n[SKG]\n' + '\n'.join(match_triples(text, skg, max_matches))).strip()
    return text


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.index_csv)
    skg = load_skg(args.skg_json, args.text_mode)
    out_dir = Path(args.output_dir)
    if args.dry_run:
        print(f'would encode {len(df)} rows to {out_dir}, mode={args.text_mode}, target={args.target_column}')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(args.device)
    model.eval()
    pending = []
    path_map = {}
    for _, row in df.iterrows():
        vid = str(row['video_id'])
        path = out_dir / f'{vid}.p'
        path_map[vid] = str(path)
        if not path.exists() or args.force:
            pending.append((vid, path, build_text(row, args.text_mode, skg, args.max_matches)))
    with torch.no_grad():
        for start in tqdm(range(0, len(pending), args.batch_size), desc='qwen_text_cache'):
            batch = pending[start:start + args.batch_size]
            texts = [x[2] or '' for x in batch]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            tokens = {k: v.to(args.device) for k, v in tokens.items()}
            out = model(**tokens)
            vecs = mean_pool(out.last_hidden_state, tokens['attention_mask']).detach().cpu().float().numpy()
            for (_, path, _), vec in zip(batch, vecs):
                with open(path, 'wb') as f:
                    pickle.dump(np.asarray(vec, dtype=np.float32), f)
    out_df = df.copy()
    out_df[args.target_column] = out_df['video_id'].astype(str).map(path_map)
    output_parent = os.path.dirname(args.output_index_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    out_df.to_csv(args.output_index_csv, index=False)
    print(f'saved {len(path_map)} cache paths to {args.output_index_csv}')


if __name__ == '__main__':
    main()
