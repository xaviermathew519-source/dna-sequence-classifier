# 🧬 DNA Sequence Classifier

A machine learning web application that classifies human DNA sequences into gene families using NLP techniques — treating DNA like a natural language.

## 🚀 Live Demo
👉 **[Try the app here](your-streamlit-link-here)**

## 🔬 Background
Every gene in the human body belongs to a specific gene family that determines its function. Identifying gene families manually in a lab is slow and expensive. This ML model classifies DNA sequences automatically with **97.95% accuracy**.

## 📊 Model Performance
| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | 97.95% |
| Random Forest | 91.21% |

## 🧬 7 Gene Families
| Class | Gene Family | Function |
|-------|-------------|----------|
| 0 | G Protein Coupled Receptors | Cell signaling — target for 35% of all drugs |
| 1 | Tyrosine Kinase | Controls cell growth — mutated in most cancers |
| 2 | Tyrosine Phosphatase | Balances kinase activity — linked to diabetes |
| 3 | Synthetase | Builds biological molecules — antibiotic targets |
| 4 | Synthase | Chemical synthesis in cells — metabolic research |
| 5 | Ion Channel | Controls electrical signals — heart disease drugs |
| 6 | Transcription Factor | Controls gene expression — cancer research |

## ⚙️ How It Works
1. Input raw DNA sequence (A, T, G, C bases)
2. Split into overlapping 6-letter k-mers (biological words)
3. Join k-mers into a sentence for NLP processing
4. Convert to numerical vector using CountVectorizer
5. Multinomial Naive Bayes predicts gene family

## 🛠️ Tech Stack
- **Data:** UCI Human DNA Dataset (4,380 sequences)
- **NLP:** K-mer tokenization, CountVectorizer
- **ML Model:** Multinomial Naive Bayes, Random Forest
- **Visualization:** Matplotlib, Seaborn
- **App:** Streamlit

## 📦 Installation
```bash
pip install streamlit scikit-learn numpy pillow
streamlit run app.py
```

## 🧪 Example Test Sequences
**G Protein Coupled Receptor (Class 0):**
ATGAGGCCCGAGCGTCCCCGGCCGCGCGGCAGCGCCCCCGGCCCGATGGAGACCCCGCCGTGGGACCCAGCCCGCAACGACTCGCTGCCGCCCACGCTGACCCCGGCCGTGCCCCCCTACGTGAAGCTTGGCCTCACCGTCGTCTACACCGTGTTCTACGCGCTGCTCTTCGTGTTCATCTACGTGCAGCTCTGGCTGGT
**Tyrosine Kinase (Class 1):**
ATGAAGACCATTACCGCCACTGGCGTCCTGTTTGTGCGGCTGGGTCCAACGCACAGCCCAAATCATAACTTTCAGGATGATTACCACGAGGATGGGTTCTGCCAGCCTTACCGGGGAATTGCCTGTGCACGCTTCATTGGCAACCGGACCATTTATGTGGACTCGCTTCAGATGCAGGGGGAGATTGAAAACCGAATCAC
## 👨‍💻 Author
Xavier Mathew — AI/ML Engineer & Computational Biologist