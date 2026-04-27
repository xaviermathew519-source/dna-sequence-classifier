import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load model and vectorizer
with open('dna_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dna_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Gene family labels
gene_families = {
    0: ('G Protein Coupled Receptors', 'Involved in cell signaling — target for 35% of all drugs'),
    1: ('Tyrosine Kinase', 'Controls cell growth — commonly mutated in cancer'),
    2: ('Tyrosine Phosphatase', 'Balances kinase activity — linked to diabetes and cancer'),
    3: ('Synthetase', 'Builds biological molecules — antibiotic drug targets'),
    4: ('Synthase', 'Chemical synthesis in cells — metabolic disease research'),
    5: ('Ion Channel', 'Controls electrical signals — heart disease and epilepsy drugs'),
    6: ('Transcription Factor', 'Controls gene expression — cancer and genetic disease research')
}

# K-mer function
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def predict_gene_family(sequence):
    kmers = getKmers(sequence)
    sentence = ' '.join(kmers)
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    return prediction, probability

# App config
st.set_page_config(page_title="DNA Sequence Classifier", page_icon="🧬", layout="wide")

# Header
st.title("🧬 DNA Sequence Classifier")
st.markdown("### Classify human DNA sequences into gene families using NLP & Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📖 About This Project")
    st.markdown("""
    **What is this?**
    A machine learning app that classifies human DNA sequences into one of 7 gene families using NLP techniques.
    
    **How does it work?**
    DNA sequences are treated like text — split into 6-letter k-mers (biological words) and classified using Multinomial Naive Bayes.
    
    **Dataset**
    - Source: UCI Human DNA Dataset
    - 4,380 human DNA sequences
    - 7 gene families
    
    **Model Performance**
    - Naive Bayes: **97.95% accuracy**
    - Random Forest: 91.21% accuracy
    
    **Tech Stack**
    Python, Scikit-learn, CountVectorizer, Naive Bayes, Streamlit
    
    **Why NLP for DNA?**
    DNA sequences follow patterns just like language — k-mers act as biological words that reveal gene function.
    """)

# Main content
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🔬 Classify DNA Sequence")
    st.markdown("Enter a human DNA sequence below to identify its gene family.")

    with st.expander("💡 Try these example sequences"):
        st.markdown("**Class 0 — G Protein Coupled Receptor:**")
        st.code("ATGAGGCCCGAGCGTCCCCGGCCGCGCGGCAGCGCCCCCGGCCCGATGGAGACCCCGCCGTGGGACCCAGCCCGCAACGACTCGCTGCCGCCCACGCTGACCCCGGCCGTGCCCCCCTACGTGAAGCTTGGCCTCACCGTCGTCTACACCGTGTTCTACGCGCTGCTCTTCGTGTTCATCTACGTGCAGCTCTGGCTGGT", language="text")
        st.markdown("**Class 1 — Tyrosine Kinase:**")
        st.code("ATGAAGACCATTACCGCCACTGGCGTCCTGTTTGTGCGGCTGGGTCCAACGCACAGCCCAAATCATAACTTTCAGGATGATTACCACGAGGATGGGTTCTGCCAGCCTTACCGGGGAATTGCCTGTGCACGCTTCATTGGCAACCGGACCATTTATGTGGACTCGCTTCAGATGCAGGGGGAGATTGAAAACCGAATCAC", language="text")
        st.markdown("**Class 2 — Tyrosine Phosphatase:**")
        st.code("ATGCCACTGCCATTTGGGTTGAAACTGAAACGCACCCGGCGCTACACGGTGTCCAGCAAGAGTTGCCTGGTTGCCCGGATCCAACTGCTTAATAACGAGTTTGTGGAGTTCACCCTGTCCGTGGAGAGCACTGGCCAGGAAAGCCTCGAGGCCGTGGCCCAGAGGCTGGAGCTGCGGGAGGTCACTTACTTCAGCCTCTG", language="text")
        st.markdown("**Class 3 — Synthetase:**")
        st.code("ATGTGTGGCATTTGGGCGCTGTTTGGCAGTGATGATTGCCTTTCTGTTCAGTGTCTGAGTGCTATGAAGATTGCACACAGAGGTCCAGATGCATTCCGTTTTGAGAATGTCAATGGATACACCAACTGCTGCTTTGGATTTCACCGGTTGGCGGTAGTTGACCCGCTGTTTGGAATGCAGCCAATTCGAGTGAAGAAATA", language="text")
        st.markdown("**Class 4 — Synthase:**")
        st.code("ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATTCCTCATCACCCAACTAAAAATATTAAACACAAACTACCACCTACCTCCCTCACCAAAGCCCATAAAAATAAAAAATTATAACAAACCCTGAGAACCAAAATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACA", language="text")
        st.markdown("**Class 5 — Ion Channel:**")
        st.code("ATGGCGGATTCCAGCGAAGGCCCCCGCGCGGGGCCCGGGGAGGTGGCTGAGCTCCCCGGGGATGAGAGTGGCACCCCAGGTGGGGAGGCTTTTCCTCTCTCCTCCCTGGCCAATCTGTTTGAGGGGGAGGATGGCTCCCTTTCGCCCTCACCGGCTGATGCCAGTCGCCCTGCTGGCCCAGGCGATGGGCGACCAAATCT", language="text")
        st.markdown("**Class 6 — Transcription Factor:**")
        st.code("ATGGCCTCAAATGAGAGAGATGCTATATCGTGGTACCAAAAGAAGATTGGAGCCTACGATCAGCAGATATGGGAAAAGTCAATCGAACAGACTCAGATTAAGGGTTTGAAAAACAAACCGAAAAAGATGGGCCACATAAAGCCAGACTTGATTGACGTTGACTTAATCAGAGGGTCAACATTTGCCAAAGCAAAACCTGA", language="text")

    dna_input = st.text_area("Enter DNA sequence (A, T, G, C only):",
        height=150,
        placeholder="e.g. ATGCCCCAACTAAATACTACCGTATGGCCCACC...")

    if st.button("🔍 Classify Sequence", use_container_width=True):
        if dna_input:
            # Clean input
            dna_clean = dna_input.strip().upper()
            invalid = set(dna_clean) - set('ATGC')
            if invalid:
                st.warning(f"⚠️ Invalid characters found: {invalid}. Please use only A, T, G, C.")
            elif len(dna_clean) < 6:
                st.warning("⚠️ Sequence too short. Please enter at least 6 characters.")
            else:
                prediction, probability = predict_gene_family(dna_clean)
                family_name, family_desc = gene_families[prediction]

                st.markdown("---")
                st.success(f"✅ Predicted Gene Family: **{family_name}**")
                st.info(f"📌 Function: {family_desc}")

                st.markdown("### Prediction Confidence")
                top3 = sorted(enumerate(probability), key=lambda x: x[1], reverse=True)[:3]
                for idx, prob in top3:
                    st.progress(float(prob))
                    st.caption(f"{gene_families[idx][0]}: {prob*100:.1f}%")

        else:
            st.warning("⚠️ Please enter a DNA sequence.")

with col2:
    st.subheader("📊 How It Works")
    st.markdown("""
    **Step 1 — Input DNA Sequence**
    Raw DNA sequence of A, T, G, C bases.
    
    **Step 2 — K-mer Splitting (k=6)**
    Sequence is split into overlapping 6-letter chunks:
    `ATGCCC → ATGCCC, TGCCCC, GCCCCA...`
    
    **Step 3 — Sentence Formation**
    K-mers joined as a sentence:
    `"atgccc tgcccc gcccca ccccaa..."`
    
    **Step 4 — CountVectorizer**
    Sentence converted to numerical vector using NLP bag-of-words technique.
    
    **Step 5 — Classification**
    Multinomial Naive Bayes predicts gene family with 97.95% accuracy.
    
    **7 Gene Families:**
    - 0: G Protein Coupled Receptors
    - 1: Tyrosine Kinase
    - 2: Tyrosine Phosphatase
    - 3: Synthetase
    - 4: Synthase
    - 5: Ion Channel
    - 6: Transcription Factor
    """)

# Model performance
st.markdown("---")
st.subheader("📈 Model Performance")
img = Image.open('dna_classification_analysis.png')
st.image(img, caption='Model Analysis — Naive Bayes vs Random Forest')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 13px;'>
    Dataset: UCI Human DNA Dataset | Model: Multinomial Naive Bayes (97.95% Accuracy) | 
    Built by Xavier Mathew
</div>
""", unsafe_allow_html=True)