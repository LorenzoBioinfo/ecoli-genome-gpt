# 🧬 EcoliGPT DNA language model: Teaching a GPT to Read the Genome

> *What does a transformer learn when trained on DNA — without being told about codons, ribosomes, or amino acids?*

This tutorial builds a minimal GPT-style language model from scratch and trains it on the complete *Escherichia coli* K-12 genome (~4.6 Mb). We then ask whether the model discovers biological structure entirely on its own, purely from statistical patterns in the sequence.

---

## 🔬 The Core Questions

After training, we run six experiments to probe what the model has learned:

| # | Question | What we measure |
|---|---|---|
| 1 | **Does the model recognise gene start signals?** | Attention weights on ATG start codons vs. internal ATGs — do they differ near the Shine-Dalgarno site? |
| 2 | **Do synonymous codons cluster together?** | Do codons encoding the same amino acid (e.g. all 6 Leucine codons) have similar internal representations? Representations extracted *in context* (from real CDS sequences, not isolated triplets).  |
| 3 | **Is the model more confident on coding regions?** | Is perplexity lower on CDS sequences than on intergenic regions? |
| 4 | **Has the model discovered codon periodicity?** | Does cross-entropy vary systematically across the 1st, 2nd, and 3rd position of a codon (period-3 structure)? |
| 5 | **Does the model reflect *E. coli* codon usage bias?** | Do the probabilities the model assigns to synonymous codons correlate with their real frequency in the *E. coli* genome? |
| 6 | **Can the model write plausible genes?** | Do sequences generated from `ATG` have the right GC content, few premature stop codons, and a codon usage similar to real *E. coli* CDS? Generation uses `model.generate()` built into the DNALM class. |

Nobody tells the model what a codon is. If it learns these patterns, it derives them from sequence statistics alone.

---

## 🔤 Background: DNA as Language

The key insight is that genomic sequences have rich statistical structure — much like natural language:

| Natural language | DNA |
|---|---|
| Characters (a, b, c, …) | Nucleotides (A, T, C, G) |
| Words | Codons (triplets, e.g. `ATG`) |
| Sentences | Genes |
| Book | Genome |

We tokenise **single nucleotides (k=1)** — not codons — vocabulary of just **8 tokens** (A, T, C, G + PAD, UNK, BOS, EOS). This is intentional: the model must rediscover that DNA is read in triplets without being given that information.

---

## 🏗️ Model Architecture

A decoder-only transformer (GPT-style), ~3M parameters, trained with next-nucleotide prediction.

| Component | Value |
|---|---|
| Tokenisation | k=1 (single nucleotides) |
| Vocabulary | 8 tokens |
| Context length | 512 nucleotides (1024 causes OOM on Colab) |
| Embedding dimension | 128 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Training objective | Next-token prediction (cross-entropy) |

We compare two variants of positional encoding:
- **Standard PE** — classic sinusoidal encoding
- **Bio PE** — sinusoidal + a period-3 periodic signal, aligned with codon structure

---

## 🚀 Quickstart (Google Colab)

The entire tutorial lives in a single notebook: **`dna_lm_tutorial.ipynb`**

1. Open the notebook in Colab (GPU recommended — training takes ~1 h on a A100)
2. Run all cells top to bottom
3. The notebook handles everything: genome download, tokenisation, training, and all three experiments

**Or run locally:**
```bash
pip install -r requirements.txt
jupyter notebook dna_lm_tutorial.ipynb
```

---

## 📓 Notebook Structure

```
dna_lm_tutorial.ipynb
│
├── Section 1 — Data
│   ├── Download E. coli K-12 genome from NCBI (accession U00096.3)
│   ├── Extract CDS and intergenic regions
│   │   └── Also saves ~50 bp upstream of each CDS (5'UTR with Shine-Dalgarno site)
│   ├── Build k=1 tokeniser
│   └── Exploratory k-mer frequency analysis
│
├── Section 2 — Model & Training
│   ├── Transformer architecture (Standard PE and Bio PE variants)
│   ├── Training loop with warmup + cosine decay
│   └── Standard PE vs Bio PE comparison
│
└── Section 3 — What Did the Model Learn?
    ├── Experiment 1: Shine-Dalgarno recognition (attention on ATG start vs internal)
    ├── Experiment 2: Synonymous codon clustering (UMAP + silhouette score, in-context reps + GC-bias control)
    ├── Experiment 3: CDS vs non-coding perplexity
    ├── Experiment 4: Codon periodicity (cross-entropy by position mod 3)
    ├── Experiment 5: Codon usage bias (Spearman correlation with real E. coli frequencies)
    └── Experiment 6: Sequence generation from ATG (GC content, stop codons, codon usage)
```

---

## 🧪 Key Biological Concepts

### Shine-Dalgarno Sequence
In bacteria like *E. coli*, translation doesn't start at any random ATG. The ribosome is recruited by a short sequence called the **Shine-Dalgarno** site — typically `AGGAGG` or similar — located ~5–10 nucleotides upstream of the start codon:

```
...AAGGAGGTT----ATG-GCT-AAA-CGT...
   Shine-Dal.  start
              ~8 nt
```

Because the model is trained on the full genome (including upstream regions), it has statistically *seen* this pattern. Experiment 1 tests whether attention weights reflect it.

### Codon Degeneracy
The genetic code maps 64 possible triplets to 20 amino acids + stop signals. This means the code is **redundant**: Leucine is encoded by 6 different codons (CTT, CTC, CTA, CTG, TTA, TTG). Experiment 2 tests whether the model's internal representations reflect this biological equivalence — without ever being told that CTT and TTG mean the same thing.

### Perplexity
Perplexity measures how "surprised" the model is by a sequence. A perplexity of K means the model is on average uncertain between K equally likely choices. With vocabulary size 8, a random baseline gives perplexity = 8. Experiment 3 tests whether coding sequences (more regular) have lower perplexity than intergenic regions.

### Codon Periodicity
In a coding sequence, each codon position has different statistical properties. The 3rd position (called the **wobble position**) is the most degenerate — multiple nucleotides often encode the same amino acid — and is subject to strong GC3 bias in *E. coli* (~51% GC at 3rd position). Experiment 4 tests whether the model has learned to treat the three positions differently.

### Codon Usage Bias
Not all synonymous codons are used equally. *E. coli* heavily favours certain codons — for example, CTG accounts for ~50% of all Leucine codons. This correlates with the abundance of corresponding tRNAs and directly affects translation speed. Experiment 5 tests whether the model's probability estimates reflect this organism-specific preference.

### Sequence Generation
A language model trained on DNA can generate new sequences by sampling from its learned distribution. Experiment 6 uses the model as a generator: starting from `ATG` (the universal start codon), it produces sequences and checks whether they resemble real *E. coli* coding sequences in terms of GC content, stop codon placement, and codon usage.

---

## 📚 References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Karpathy, A. (2023). *Let's build GPT: from scratch.* [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Ji et al. (2021). *DNABERT.* Bioinformatics.
- Dalla-Torre et al. (2023). *The Nucleotide Transformer.* bioRxiv.
- *E. coli* K-12 MG1655 genome: NCBI accession [U00096.3](https://www.ncbi.nlm.nih.gov/nuccore/U00096.3)
