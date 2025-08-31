#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:18:42 2025

@author: tomaslg
"""

import os, json, re, math, argparse, urllib.request, html, subprocess, shutil
from collections import Counter
from typing import List, Iterable, Dict, Tuple, Optional, Set

import torch
from torch import nn

# -----------------------------
# Tokenizer / Vocab
# -----------------------------
def tokenize(text: str) -> List[str]:
    TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return TOKEN_RE.findall(text.lower())

class Vocab:
    def __init__(self, tokens_iter: Iterable[List[str]], min_freq: int = 1, specials: Optional[List[str]] = None, max_size: Optional[int] = None):
        if specials is None:
            specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        counter = Counter()
        for toks in tokens_iter:
            counter.update(toks)
        words = [w for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]
        self.itos: List[str] = list(specials) + words
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @classmethod
    def from_itos(cls, itos: List[str]) -> "Vocab":
        # Rebuild a Vocab from an itos list saved in a checkpoint
        v = object.__new__(cls)
        v.itos = list(itos)
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        v.pad_id = v.stoi["<pad>"]; v.unk_id = v.stoi["<unk>"]; v.bos_id = v.stoi["<bos>"]; v.eos_id = v.stoi["<eos>"]
        return v

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"

def encode(vocab: Vocab, tokens: List[str], add_bos=False, add_eos=False) -> List[int]:
    ids = []
    if add_bos:
        ids.append(vocab.bos_id)
    ids.extend(vocab[t] for t in tokens)
    if add_eos:
        ids.append(vocab.eos_id)
    return ids

def decode(vocab: Vocab, ids: List[int]) -> str:
    toks = [vocab.lookup_token(i) for i in ids]
    s = " ".join(toks)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return s

# -----------------------------
# Transformer LM
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 2, num_layers: int = 2, d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.tok_emb.weight, -init_range, init_range)
        nn.init.zeros_(self.lm_head.bias)
        nn.init.uniform_(self.lm_head.weight, -init_range, init_range)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=src_mask)
        logits = self.lm_head(x)
        return logits

# -----------------------------
# Data building (URLs or text)
# -----------------------------
def _strip_html(html_text: str) -> str:
    # Very lightweight HTML -> text (no external deps). Good enough for plain content.
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def _split_into_sentences(text: str) -> List[str]:
    # Simple sentence split; keeps punctuation as its own token later
    # Break on . ! ? followed by space/newline
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def fetch_text_from_urls(urls: List[str], timeout: int = 20) -> str:
    chunks = []
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=timeout) as resp:
                raw = resp.read()
                # attempt to decode; fallback to utf-8
                try:
                    charset = resp.headers.get_content_charset() or "utf-8"
                except Exception:
                    charset = "utf-8"
                html_text = raw.decode(charset, errors="ignore")
                chunks.append(_strip_html(html_text))
        except Exception as e:
            print(f"[warn] failed to fetch {u}: {e}")
    return "\n".join(chunks)

def build_dataset(text: str) -> Tuple[Vocab, torch.Tensor]:
    lines = _split_into_sentences(text) if text.strip() else []
    tokenized = [tokenize(l) for l in lines]
    if not tokenized:
        tokenized = [tokenize("this is a tiny fallback corpus .")]
    vocab = Vocab(tokenized, min_freq=1, max_size=20000)
    ids: List[int] = []
    for toks in tokenized:
        ids.extend(encode(vocab, toks, add_bos=False, add_eos=True))
    data = torch.tensor(ids, dtype=torch.long)
    return vocab, data

def batchify(data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    nbatch = max(1, data.size(0) // max(1, batch_size))
    data = data[: nbatch * batch_size]
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(source: torch.Tensor, i: int, bptt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target

# -----------------------------
# Training + Checkpointing
# -----------------------------
def train_model(model: nn.Module, train_data: torch.Tensor, vocab_size: int, epochs: int = 5, bptt: int = 35, lr: float = 3e-3, clip: float = 1.0):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        i = 0
        step = 0
        while i < train_data.size(0) - 1:
            data, targets = get_batch(train_data, i, bptt)
            i += data.size(0)
            step += 1
            optimizer.zero_grad(set_to_none=True)
            mask = generate_causal_mask(data.size(0), device=device)
            logits = model(data, src_mask=mask)
            loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(1, step)
        print(f"Epoch {epoch:02d} | loss {avg:.3f} | ppl {math.exp(avg):.2f}")

def save_checkpoint(path: str, model: TransformerLM, vocab: Vocab, config: Dict):
    payload = {
        "state_dict": model.state_dict(),
        "vocab_itos": vocab.itos,
        "config": config,
    }
    torch.save(payload, path)
    print(f"[info] saved checkpoint to {path}")

def load_checkpoint(path: str, device: torch.device) -> Tuple[TransformerLM, Vocab, Dict]:
    payload = torch.load(path, map_location=device)
    vocab = Vocab.from_itos(payload["vocab_itos"])
    cfg = payload.get("config", {"vocab_size": len(vocab), "d_model":128, "nhead":2, "num_layers":2, "d_ff":256, "dropout":0.2})
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=cfg.get("d_model",128),
        nhead=cfg.get("nhead",2),
        num_layers=cfg.get("num_layers",2),
        d_ff=cfg.get("d_ff",256),
        dropout=cfg.get("dropout",0.2),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[info] loaded checkpoint from {path}")
    return model, vocab, cfg

# -----------------------------
# Generation with realistic stopwords
# -----------------------------
# def default_stopwords() -> Set[str]:
#     # A realistic English stopword set (common function words) + punctuation
#     words = {
#         "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
#         "be","because","been","before","being","below","between","both","but","by",
#         "can","can’t","cannot","could","couldn’t",
#         "did","didn’t","do","does","doesn’t","doing","don’t","down","during",
#         "each","few","for","from","further",
#         "had","hadn’t","has","hasn’t","have","haven’t","having","he","he’d","he’ll","he’s","her","here","here’s",
#         "hers","herself","him","himself","his","how","how’s",
#         "i","i’d","i’ll","i’m","i’ve","if","in","into","is","isn’t","it","it’s","its","itself",
#         "let’s",
#         "me","more","most","mustn’t","my","myself",
#         "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
#         "same","shan’t","she","she’d","she’ll","she’s","should","shouldn’t","so","some","such",
#         "than","that","that’s","the","their","theirs","them","themselves","then","there","there’s","these","they",
#         "they’d","they’ll","they’re","they’ve","this","those","through","to","too",
#         "under","until","up","very",
#         "was","wasn’t","we","we’d","we’ll","we’re","we’ve","were","weren’t","what","what’s","when","when’s",
#         "where","where’s","which","while","who","who’s","whom","why","why’s","with","won’t","would","wouldn’t",
#         "you","you’d","you’ll","you’re","you’ve","your","yours","yourself","yourselves",
#         # common symbols/punct tokens (tokenizer keeps them separate)
#         ",",".","!","?",";",":","—","-","(",")","[","]","{","}","…","'","\""
#     }
#     # Normalize ’ to ' just in case
#     normalized = set()
#     for w in words:
#         normalized.add(w.replace("’","'"))
#     # also include lowercase variants (already lowercase) and <eos> token
#     normalized.add("<eos>")
#     return normalized
def default_stopwords(mode: str = "punct_only") -> Set[str]:
    """Modes:
    - 'punct_only': stop only on punctuation & <eos>  (good default)
    - 'standard'  : common function words + punctuation + <eos>
    """
    punct = {",",".","!","?",";",":","—","-","(",")","[","]","{","}","…","'","\"", "<eos>"}
    if mode == "punct_only":
        return punct

    words = {
        "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
        "be","because","been","before","being","below","between","both","but","by",
        "can","can't","cannot","could","couldn't",
        "did","didn't","do","does","doesn't","doing","don't","down","during",
        "each","few","for","from","further",
        "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's",
        "hers","herself","him","himself","his","how","how's",
        "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
        "let's",
        "me","more","most","mustn't","my","myself",
        "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
        "same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
        "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they",
        "they'd","they'll","they're","they've","this","those","through","to","too",
        "under","until","up","very",
        "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's",
        "where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
        "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
    }
    # Normalize curly apostrophes
    words = {w.replace("’","'") for w in words}
    return words | punct

# def greedy_next_token(model: nn.Module, input_ids: torch.Tensor) -> int:
#     device = input_ids.device
#     mask = generate_causal_mask(input_ids.size(0), device=device)
#     with torch.no_grad():
#         logits = model(input_ids, src_mask=mask)
#     next_id = int(torch.argmax(logits[-1, 0]).item())
#     return next_id
def next_token_id(
    model: nn.Module,
    vocab: Vocab,
    input_ids: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 50,
    ban_specials: bool = True,
    ban_numbers: bool = False,
) -> int:
    device = input_ids.device
    mask = generate_causal_mask(input_ids.size(0), device=device)
    with torch.no_grad():
        logits = model(input_ids, src_mask=mask)  # [seq_len, 1, vocab]
    last_logits = logits[-1, 0]                  # [vocab]
    return sample_from_logits(
        last_logits, vocab,
        temperature=temperature,
        top_k=top_k,
        ban_specials=ban_specials,
        ban_numbers=ban_numbers,
    )


# def _ban_token_ids_inplace(logits: torch.Tensor, banned_ids):
#     # logits: [vocab]
#     if not banned_ids:
#         return
#     logits[banned_ids] = float("-inf")
def _ban_token_ids_inplace(logits: torch.Tensor, banned_ids):
    """
    Set logits for banned token ids to -inf (in place).
    `banned_ids` may be a Tensor or a Python iterable of ints.
    """
    if isinstance(banned_ids, torch.Tensor):
        if banned_ids.numel() == 0:
            return
        logits.index_fill_(0, banned_ids.to(dtype=torch.long), float("-inf"))
    elif isinstance(banned_ids, (list, tuple, set)):
        if len(banned_ids) == 0:
            return
        idx = torch.tensor(list(banned_ids), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, idx, float("-inf"))
    else:
        # Nothing to ban / unsupported type
        return
def extract_text_from_pdf(path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a local PDF.
    Tries: pdfminer.six -> PyPDF2 -> external 'pdftotext'.
    """
    # Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
        return _pdfminer_extract(path, maxpages=max_pages)
    except Exception:
        pass

    # Try PyPDF2
    try:
        import PyPDF2
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            end = total if max_pages is None else min(max_pages, total)
            for i in range(end):
                try:
                    text_parts.append(reader.pages[i].extract_text() or "")
                except Exception:
                    continue
        return "\n".join(text_parts)
    except Exception:
        pass

    # Try system 'pdftotext'
    if shutil.which("pdftotext"):
        tmp_txt = path + ".txt"
        try:
            # -layout preserves rough columns; drop it if you prefer plain flow
            cmd = ["pdftotext", "-layout", path, tmp_txt]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(tmp_txt, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_txt)
            except Exception:
                pass

    raise RuntimeError(
        "Could not extract text from PDF. Install 'pdfminer.six' or 'PyPDF2', "
        "or install the 'pdftotext' command-line tool."
    )

def fetch_text_from_pdfs(paths: List[str], max_pages: Optional[int] = None, recurse: bool = True) -> str:
    """
    Accepts a list of PDF file paths and/or directories.
    - For files: includes if they end with .pdf (case-insensitive).
    - For directories: recursively includes all *.pdf files inside.
    - Duplicates are ignored. Files are processed in sorted order for reproducibility.
    """
    def _is_pdf(p: str) -> bool:
        return p.lower().endswith(".pdf")

    # Gather candidate PDF files
    seen = set()
    files: List[str] = []

    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            # Walk the directory tree
            for root, _, filenames in os.walk(p, topdown=True, followlinks=False):
                for fname in filenames:
                    full = os.path.join(root, fname)
                    if _is_pdf(fname):
                        if full not in seen:
                            seen.add(full)
                            files.append(full)
        else:
            # Treat as a file path
            if os.path.exists(p) and _is_pdf(p):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
            else:
                print(f"[warn] Skipping non-PDF or missing path: {p}")

    if not files:
        print("[warn] No PDF files found from provided paths.")
        return ""

    files.sort()  # deterministic processing order
    texts = []
    print(f"[info] extracting text from {len(files)} PDF(s)…")
    for fp in files:
        try:
            print(f"[info] reading PDF: {fp}")
            t = extract_text_from_pdf(fp, max_pages=max_pages)
            if t and t.strip():
                texts.append(t)
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")

    return "\n\n".join(texts)


def _is_number_token(tok: str) -> bool:
    return tok.isdigit() or re.fullmatch(r"[+-]?\d+([.,]\d+)?", tok) is not None

def sample_from_logits(
    logits: torch.Tensor,
    vocab,
    temperature: float = 1.0,
    top_k: int = 0,
    ban_specials: bool = True,
    ban_numbers: bool = False,
):
    """Return a sampled token id with constraints."""
    # Ban undesirable tokens
    banned = []
    if ban_specials:
        for t in ("<pad>", "<unk>", "<bos>"):
            if t in vocab.stoi:
                banned.append(vocab.stoi[t])
    # Optionally ban pure-number tokens by scanning the vocab once
    if ban_numbers:
        for idx, tok in enumerate(vocab.itos):
            if _is_number_token(tok):
                banned.append(idx)
    if banned:
        logits = logits.clone()
        _ban_token_ids_inplace(logits, torch.tensor(banned, device=logits.device))

    # Temperature
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Top-k
    if top_k and top_k > 0:
        topk = torch.topk(logits, k=min(top_k, logits.size(-1)))
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk.indices, topk.values)
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

# def generate_until_stopword(
#     model: nn.Module,
#     vocab: Vocab,
#     prompt: str,
#     stopwords: Optional[Set[str]] = None,
#     max_new_tokens: int = 30
# ) -> str:
#     device = next(model.parameters()).device
#     stopwords = default_stopwords() if stopwords is None else stopwords
#     prompt_toks = tokenize(prompt) if prompt.strip() else ["<bos>"]
#     ids = encode(vocab, prompt_toks, add_bos=False, add_eos=False)
#     if not ids:
#         ids = [vocab.bos_id]
#     input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [seq_len, 1]
#     for _ in range(max_new_tokens):
#         next_id = greedy_next_token(model, input_ids)
#         next_tok = vocab.lookup_token(next_id)
#         if next_tok in stopwords or next_tok == "<eos>":
#             break
#         input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=0)
#     return decode(vocab, input_ids.squeeze(1).tolist())
def generate_until_stopword(
    model: nn.Module,
    vocab: Vocab,
    prompt: str,
    stopwords: Optional[Set[str]] = None,
    max_new_tokens: int = 50,
    min_new_tokens_before_stop: int = 5,
    stop_mode: str = "punct_only",
    temperature: float = 0.9,
    top_k: int = 50,
    ban_numbers: bool = False,
) -> str:
    device = next(model.parameters()).device
    sw = default_stopwords(stop_mode) if stopwords is None else stopwords

    prompt_toks = tokenize(prompt) if prompt.strip() else ["<bos>"]
    ids = encode(vocab, prompt_toks, add_bos=False, add_eos=False) or [vocab.bos_id]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [seq_len, 1]
    new_tokens = 0

    for _ in range(max_new_tokens):
        nxt = next_token_id(
            model, vocab, input_ids,
            temperature=temperature, top_k=top_k,
            ban_specials=True, ban_numbers=ban_numbers
        )
        nxt_tok = vocab.lookup_token(nxt)

        # allow a few tokens before we start honoring stopwords
        if new_tokens >= min_new_tokens_before_stop and (nxt_tok in sw or nxt_tok == "<eos>"):
            break

        # If it's a stopword but we're still in the grace window, try re-sampling once
        if new_tokens < min_new_tokens_before_stop and (nxt_tok in sw or nxt_tok == "<eos>"):
            alt = next_token_id(
                model, vocab, input_ids,
                temperature=temperature, top_k=top_k,
                ban_specials=True, ban_numbers=ban_numbers
            )
            nxt, nxt_tok = alt, vocab.lookup_token(alt)

        input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=device)], dim=0)
        new_tokens += 1

    return decode(vocab, input_ids.squeeze(1).tolist())


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny Transformer LM with realistic stopwords, web training, and checkpointing.")
    parser.add_argument("--urls", nargs="*", default=[], help="One or more URLs to fetch text from.")
    parser.add_argument("--text", type=str, default="", help="Raw text to train on (if provided, appended to fetched text).")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs (if training).")
    parser.add_argument("--bptt", type=int, default=35, help="Sequence length for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batchify.")
    parser.add_argument("--save_path", type=str, default="tiny_transformer_lm.pt", help="Checkpoint path.")
    parser.add_argument("--no_train", action="store_true", help="Do not train even if no checkpoint; exit if checkpoint missing.")
    parser.add_argument("--pdfs", nargs="*", default=[], help="Paths to local PDF files to include in training.")
    parser.add_argument("--max_pdf_pages", type=int, default=None, help="Limit pages per PDF (None = all).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Try to load checkpoint first
    if os.path.exists(args.save_path):
        model, vocab, cfg = load_checkpoint(args.save_path, device)
    else:
        if args.no_train:
            print("[error] checkpoint not found and --no_train set. Exiting.")
            return
        # Build corpus (URLs + text or fallback)
        raw_text = ""
        if args.urls:
            print(f"[info] fetching {len(args.urls)} URLs …")
            raw_text += fetch_text_from_urls(args.urls)
        if args.text:
            raw_text += ("\n" if raw_text else "") + args.text
        if args.pdfs:
            print(f"[info] extracting text from {len(args.pdfs)} PDF(s) …")
            pdf_text = fetch_text_from_pdfs(args.pdfs, max_pages=args.max_pdf_pages)
            if pdf_text.strip():
                raw_text += ("\n" if raw_text else "") + pdf_text


        if not raw_text.strip():
            # Fallback tiny corpus (portable demo)
            raw_text = """
            Once upon a time, a tiny model learned simple stories. The model loved small datasets.
            It practiced predicting the next word, one token at a time. Sometimes it stopped early.
            Sometimes it continued to explore. Transformers with attention can be compact and fun.
            Deep learning is powerful, but even small models can learn basic patterns.
            """

        vocab, stream = build_dataset(raw_text)
        print(f"[info] Vocab size: {len(vocab)} | Tokens: {len(stream)}")

        train_data = batchify(stream, batch_size=max(1, min(args.batch_size, len(stream))), device=device)
        cfg = {"d_model":128,"nhead":2,"num_layers":2,"d_ff":256,"dropout":0.2}

        model = TransformerLM(
            vocab_size=len(vocab),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        train_model(model, train_data, vocab_size=len(vocab), epochs=args.epochs, bptt=args.bptt, lr=3e-3)
        save_checkpoint(args.save_path, model, vocab, {"vocab_size":len(vocab), **cfg})

    # Demo generation with realistic stopwords
    prompts = [
        "once upon a time",
        "the model",
        "transformers",
        "deep learning"
    ]
    sw = default_stopwords()
    for p in prompts:
        # out = generate_until_stopword(model, vocab, p, stopwords=sw,
        #     max_new_tokens=300, temperature=.9)
        out = generate_until_stopword(
            model, vocab, p,
            stopwords=None,                 # use built-in set via stop_mode
            stop_mode="punct_only",         # only stop on punctuation (and <eos>)
            max_new_tokens=150,             # output size
            min_new_tokens_before_stop=20,  # ignore stops for the first N tokens
            temperature=0.95,               # the larger the temp the more exploratory
            top_k=80,                       # sample among top K
            ban_numbers=True,               # keeps it less list-like
        )

        print(f"\nPROMPT: {p}\nGEN   : {out}")

if __name__ == "__main__":
    # runfile('/home/tomaslg/Projects/LM/example_LM.py', args='--urls https://d2l.ai/ https://www.deeplearningbook.org/ https://www.deeplearningbook.org/contents/intro.html --epochs 6', wdir='/home/tomaslg/Projects/LM')
    # runfile('/home/tomaslg/Projects/LM/example_LM.py', args='--pdfs /home/tomaslg/Plambeck /home/tomaslg/Projects/fan2025optimization.pdf --max_pdf_pages 200 --urls https://d2l.ai/ https://www.deeplearningbook.org/ https://www.deeplearningbook.org/contents/intro.html --epochs 60', wdir='/home/tomaslg/Projects/LM')
    # runfile('/home/tomaslg/Projects/LM/example_LM.py', args='--pdfs /home/tomaslg/Plambeck /home/tomaslg/Projects/fan2025optimization.pdf --max_pdf_pages 200 --epochs 60', wdir='/home/tomaslg/Projects/LM')
    main()


# import math, re, random
# from collections import Counter
# from typing import List, Iterable, Dict, Tuple, Optional, Set

# import torch
# from torch import nn
# from torch.nn import functional as F

# def tokenize(text: str) -> List[str]:
#     # lowercase + keep punctuation as separate tokens
#     TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
#     return TOKEN_RE.findall(text.lower())

# class Vocab:
#     def __init__(self, tokens_iter: Iterable[List[str]], min_freq: int = 1, specials: Optional[List[str]] = None, max_size: Optional[int] = None):
#         if specials is None:
#             specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
#         counter = Counter()
#         for toks in tokens_iter:
#             counter.update(toks)

#         # sort by frequency then alphabetically for determinism
#         words = [w for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
#         if max_size is not None:
#             words = words[: max(0, max_size - len(specials))]

#         self.itos: List[str] = list(specials) + words
#         self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
#         self.pad_id = self.stoi["<pad>"]
#         self.unk_id = self.stoi["<unk>"]
#         self.bos_id = self.stoi["<bos>"]
#         self.eos_id = self.stoi["<eos>"]

#     def __len__(self) -> int:
#         return len(self.itos)

#     def __contains__(self, token: str) -> bool:
#         return token in self.stoi

#     def __getitem__(self, token: str) -> int:
#         return self.stoi.get(token, self.unk_id)

#     def lookup_token(self, idx: int) -> str:
#         if 0 <= idx < len(self.itos):
#             return self.itos[idx]
#         return "<unk>"

# def encode(vocab: Vocab, tokens: List[str], add_bos=False, add_eos=False) -> List[int]:
#     ids = []
#     if add_bos:
#         ids.append(vocab.bos_id)
#     ids.extend(vocab[t] for t in tokens)
#     if add_eos:
#         ids.append(vocab.eos_id)
#     return ids

# def decode(vocab: Vocab, ids: List[int]) -> str:
#     toks = [vocab.lookup_token(i) for i in ids]
#     # simple detokenizer: join with space, then fix spacing before punctuation
#     s = " ".join(toks)
#     s = re.sub(r"\s+([.,!?;:])", r"\1", s)
#     return s

# # -----------------------------
# # 2) Tiny Transformer language model
# # -----------------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [seq_len, batch, d_model]
#         x = x + self.pe[: x.size(0)]
#         return self.dropout(x)

# def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
#     # upper triangular filled with -inf to prevent attending to future tokens
#     return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

# class TransformerLM(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 2, num_layers: int = 2, d_ff: int = 256, dropout: float = 0.2):
#         super().__init__()
#         self.d_model = d_model
#         self.tok_emb = nn.Embedding(vocab_size, d_model)
#         self.pos_enc = PositionalEncoding(d_model, dropout)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=False)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.lm_head = nn.Linear(d_model, vocab_size)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         init_range = 0.1
#         nn.init.uniform_(self.tok_emb.weight, -init_range, init_range)
#         nn.init.zeros_(self.lm_head.bias)
#         nn.init.uniform_(self.lm_head.weight, -init_range, init_range)

#     def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # src: [seq_len, batch]
#         x = self.tok_emb(src) * math.sqrt(self.d_model)  # [seq_len, batch, d_model]
#         x = self.pos_enc(x)
#         x = self.transformer(x, mask=src_mask)           # [seq_len, batch, d_model]
#         logits = self.lm_head(x)                         # [seq_len, batch, vocab]
#         return logits



# def build_dataset(text: str) -> Tuple[Vocab, torch.Tensor]:
#     # prepare sentences -> tokens
#     lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
#     tokenized = [tokenize(l) for l in lines]
#     vocab = Vocab(tokenized, min_freq=1, max_size=5000)
#     # flatten all sentences into one long stream with <eos> markers
#     ids: List[int] = []
#     for toks in tokenized:
#         ids.extend(encode(vocab, toks, add_bos=False, add_eos=True))
#     data = torch.tensor(ids, dtype=torch.long)
#     return vocab, data

# def batchify(data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
#     # Work like the PyTorch LM tutorial: chop the 1D stream into batch_size columns
#     nbatch = data.size(0) // batch_size
#     data = data[: nbatch * batch_size]
#     data = data.view(batch_size, -1).t().contiguous()  # [seq_len, batch]
#     return data.to(device)

# def get_batch(source: torch.Tensor, i: int, bptt: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     # source: [full_seq_len, batch]
#     seq_len = min(bptt, source.size(0) - 1 - i)
#     data = source[i : i + seq_len]           # [seq_len, batch]
#     target = source[i + 1 : i + 1 + seq_len] # [seq_len, batch]
#     return data, target

# def train_model(model: nn.Module, train_data: torch.Tensor, vocab_size: int, epochs: int = 5, bptt: int = 35, lr: float = 3e-3, clip: float = 1.0):
#     device = next(model.parameters()).device
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     model.train()

#     for epoch in range(1, epochs + 1):
#         total_loss = 0.0
#         ntokens = vocab_size
#         i = 0
#         step = 0
#         while i < train_data.size(0) - 1:
#             data, targets = get_batch(train_data, i, bptt)   # [seq_len, batch], [seq_len, batch]
#             i += data.size(0)
#             step += 1

#             optimizer.zero_grad(set_to_none=True)
#             mask = generate_causal_mask(data.size(0), device=device)
#             logits = model(data, src_mask=mask)              # [seq_len, batch, ntokens]
#             loss = criterion(logits.view(-1, ntokens), targets.reshape(-1))
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()
#             total_loss += loss.item()

#         avg = total_loss / max(1, step)
#         print(f"Epoch {epoch:02d} | loss {avg:.3f} | ppl {math.exp(avg):.2f}")

# # -----------------------------
# # 5) Generation with stopword stop
# # -----------------------------
# def greedy_next_token(model: nn.Module, input_ids: torch.Tensor) -> int:
#     # input_ids: [seq_len, 1] on model device
#     device = input_ids.device
#     mask = generate_causal_mask(input_ids.size(0), device=device)
#     with torch.no_grad():
#         logits = model(input_ids, src_mask=mask)  # [seq_len, 1, vocab]
#     next_logits = logits[-1, 0]                  # [vocab]
#     next_id = int(torch.argmax(next_logits).item())
#     return next_id

# def generate_until_stopword(
#     model: nn.Module,
#     vocab: Vocab,
#     prompt: str,
#     stopwords: Set[str],
#     max_new_tokens: int = 30
# ) -> str:
#     device = next(model.parameters()).device
#     # tokenize prompt; if empty, start with <bos>
#     prompt_toks = tokenize(prompt) if prompt.strip() else ["<bos>"]
#     # ensure known tokens
#     ids = encode(vocab, prompt_toks, add_bos=False, add_eos=False)
#     if not ids:
#         ids = [vocab.bos_id]
#     input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [seq_len, 1]

#     for _ in range(max_new_tokens):
#         next_id = greedy_next_token(model, input_ids)
#         next_tok = vocab.lookup_token(next_id)
#         # termination on stopword or <eos>
#         if next_tok in stopwords or next_tok == "<eos>":
#             break
#         # append and continue
#         input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=0)

#     return decode(vocab, input_ids.squeeze(1).tolist())

# # -----------------------------
# # 6) Main demo
# # -----------------------------
# def main():
#     TOY_TEXT = """
#     Once upon a time, a tiny model learned simple stories. The model loved small datasets.
#     It practiced predicting the next word, one token at a time. Sometimes it stopped early.
#     Sometimes it continued to explore. Transformers with attention can be compact and fun.
#     Deep learning is powerful, but even small models can learn basic patterns.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     # build vocab + data
#     vocab, stream = build_dataset(TOY_TEXT)
#     print(f"Vocab size: {len(vocab)} | Data tokens: {len(stream)}")

#     # make batches (we keep batch_size small for portability)
#     batch_size = 8
#     train_data = batchify(stream, batch_size=batch_size, device=device)

#     # small model for portability
#     model = TransformerLM(
#         vocab_size=len(vocab),
#         d_model=128,
#         nhead=2,
#         num_layers=2,
#         d_ff=256,
#         dropout=0.2,
#     ).to(device)

#     # train briefly (toy)
#     train_model(model, train_data, vocab_size=len(vocab), epochs=6, bptt=35, lr=3e-3)

#     # generate with stopwords
#     stopwords = {"and", "the", "or", "but", ","}  # you can customize this set
#     prompts = [
#         "once upon a time",
#         "the model",
#         "transformers",
#         "deep learning"
#     ]
#     for p in prompts:
#         out = generate_until_stopword(model, vocab, p, stopwords=stopwords, max_new_tokens=20)
#         print(f"\nPROMPT: {p}\nGEN   : {out}")

# if __name__ == "__main__":
#     main()


# # import math
# # import torch
# # from torch import nn, optim

# # class PositionalEncoding(nn.Module):
# #     """Injects positional information using sine and cosine frequencies:contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}."""
# #     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
# #         super().__init__()
# #         self.dropout = nn.Dropout(p=dropout)
# #         # Create constant positional encoding matrix
# #         position = torch.arange(max_len).unsqueeze(1)                       # [max_len, 1]
# #         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
# #         pe = torch.zeros(max_len, 1, d_model)                               # [max_len, 1, d_model]
# #         pe[:, 0, 0::2] = torch.sin(position * div_term)                     # apply sin to even indices
# #         pe[:, 0, 1::2] = torch.cos(position * div_term)                     # apply cos to odd indices
# #         self.register_buffer('pe', pe)  # store pe as non-learnable buffer
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """Add positional encoding to input embeddings:contentReference[oaicite:14]{index=14}."""
# #         x = x + self.pe[:x.size(0)]  # add positional encodings up to sequence length
# #         return self.dropout(x)

# # class TransformerLM(nn.Module):
# #     """A small Transformer-based language model for next-word prediction."""
# #     def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 2, num_layers: int = 2, d_ff: int = 128, dropout: float = 0.2):
# #         super().__init__()
# #         # Layers: embedding, positional encoding, Transformer encoder, and output linear:contentReference[oaicite:15]{index=15} 
# #         self.embedding = nn.Embedding(vocab_size, d_model)
# #         self.pos_encoder = PositionalEncoding(d_model, dropout)
# #         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_ff, dropout=dropout)
# #         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
# #         self.output_fc = nn.Linear(d_model, vocab_size)
# #         self.d_model = d_model
# #         # Initialize weights for stability
# #         self._reset_parameters()
# #     def _reset_parameters(self):
# #         init_range = 0.1
# #         self.embedding.weight.data.uniform_(-init_range, init_range)
# #         self.output_fc.bias.data.zero_()
# #         self.output_fc.weight.data.uniform_(-init_range, init_range)
# #     def forward(self, src_tokens: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
# #         """
# #         src_tokens: LongTensor of shape [seq_len, batch_size] with token indices.
# #         src_mask: FloatTensor of shape [seq_len, seq_len] for masking (optional).
# #         Returns: Tensor of shape [seq_len, batch_size, vocab_size] with raw logits for each token position:contentReference[oaicite:16]{index=16}.
# #         """
# #         # Embed tokens and scale by sqrt(d_model):contentReference[oaicite:17]{index=17} 
# #         src = self.embedding(src_tokens) * math.sqrt(self.d_model) 
# #         src = self.pos_encoder(src)                      # add positional encoding:contentReference[oaicite:18]{index=18}
# #         memory = self.transformer(src, mask=src_mask)    # transformer encoder outputs
# #         output_logits = self.output_fc(memory)           # project to vocabulary logits
# #         return output_logits

# # def generate_text(model, start_text: str, vocab, stopwords=None, max_len=50):
# #     """Generate text from the model by iteratively predicting next words."""
# #     model.eval()  # set model to evaluation mode
# #     if stopwords is None:
# #         stopwords = set()
# #     # Tokenize the start_text into words and convert to indices
# #     words = start_text.split()  # simple whitespace tokenizer for demonstration
# #     # Ensure the model uses same device as inputs
# #     device = next(model.parameters()).device
# #     for _ in range(max_len):
# #         # Convert current word list to tensor of indices
# #         input_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in words]
# #         input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(1)  # shape: [seq_len, 1]
# #         # Create causal mask for current sequence length
# #         seq_len = input_tensor.size(0)
# #         mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
# #         # Get model predictions for the next token
# #         with torch.no_grad():
# #             output_logits = model(input_tensor, src_mask=mask)  # shape: [seq_len, 1, vocab_size]
# #         next_token_logits = output_logits[-1, 0, :]            # logits for the last time step
# #         # Pick the next token (greedy)
# #         next_token_id = int(torch.argmax(next_token_logits))
# #         next_word = vocab.lookup_token(next_token_id)  # invert index to word
# #         if next_word in stopwords or next_word is None:
# #             # If we've generated a stopword (or an unknown token), end the loop
# #             break
# #         words.append(next_word)  # append the predicted word and continue
# #     return " ".join(words)

# # if __name__ == '__main__':

# #     model = TransformerLM(vocab_size=len(vocab), d_model=128, nhead=2, num_layers=2, d_ff=128)
# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    
# #     # Assume `train_batches` is an iterable of (input_tensor, target_tensor) for each batch
# #     for epoch in range(5):  # train for 5 epochs
# #         model.train()
# #         total_loss = 0.0
# #         for inputs, targets in train_batches:
# #             optimizer.zero_grad()
# #             output_logits = model(inputs)               # output shape: [seq_len, batch, vocab_size]
# #             # Flatten outputs and targets for loss computation:contentReference[oaicite:28]{index=28}
# #             output_flat = output_logits.view(-1, len(vocab))
# #             loss = criterion(output_flat, targets.view(-1))
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #         print(f"Epoch {epoch+1}, average loss: {total_loss/len(train_batches):.3f}")
    
# #     stopwords = {"and", "the", "or", "a"}  # define some stopwords to stop on
# #     prompt = "Once upon a time"
# #     generated = generate_text(model, prompt, vocab, stopwords=stopwords, max_len=20)
# #     print("Generated text:", generated)

