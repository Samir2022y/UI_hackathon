import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Classificateur de Propositions d'Emploi", layout="wide")

# Chargement du modèle et du tokenizer
@st.cache_resource
def charger_modele():
    CHEMIN_MODELE = "saved_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHEMIN_MODELE)
        model = AutoModelForSequenceClassification.from_pretrained(CHEMIN_MODELE)
        return tokenizer, model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

# Fonction de classification
def classifier_textes(liste_textes, tokenizer, model, batch_size=16):
    if not liste_textes:
        return []
    
    predictions = []
    
    for i in range(0, len(liste_textes), batch_size):
        batch = liste_textes[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
            
            # S'assurer que preds est bien une liste plate
            if preds.ndim == 0:
                preds = [float(preds)]
            elif preds.ndim == 1:
                preds = preds.tolist()
            else:
                preds = [float(p[0]) for p in preds]
            
            predictions.extend([int(p >= 0.5) for p in preds])
    
    return predictions

# Interface utilisateur principale
st.title("📌 Classificateur de Propositions d'Emploi")
st.markdown("Ce programme permet de classifier les propositions d'emploi en fonction de leur acceptation potentielle.")

# Message sur le format requis
tips = """
📌 **Format du fichier CSV requis :**
- **Colonne 'activity'** : Contenu de la proposition d'emploi
- **Colonne 'proposition_accepted'** : (sera générée automatiquement après classification)
"""
st.info(tips)

# Chargement du modèle
tokenizer, model = charger_modele()
if tokenizer is None or model is None:
    st.stop()

# Barre latérale pour instructions
st.sidebar.header("📋 Instructions")
st.sidebar.write("1️⃣ Téléchargez un fichier CSV contenant une colonne 'activity'.")
st.sidebar.write("2️⃣ Cliquez sur **Classer les Propositions**.")
st.sidebar.write("3️⃣ Téléchargez le fichier de résultats.")

# Téléchargement du fichier
uploaded_file = st.file_uploader("📤 Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        if "activity" not in df.columns:
            st.error("Le fichier CSV doit contenir une colonne 'activity'.")
            st.stop()
        
        st.subheader("🔍 Aperçu du fichier chargé")
        st.write(df.head())
        
        # Bouton de classification
        if st.button("🚀 Classer les Propositions"):
            with st.spinner("Classification en cours..."):
                predictions = classifier_textes(df["activity"].astype(str).tolist(), tokenizer, model)
                df["proposition_accepted"] = predictions
                
                # Séparation des décisions
                df_accepted = df[df["proposition_accepted"] == 1]
                df_rejected = df[df["proposition_accepted"] == 0]
                
                # Résultats
                st.success("✅ Classification terminée !")
                st.subheader("📊 Résultats de classification")
                st.write(df)
                
                # Statistiques
                total = len(predictions)
                accepted = sum(predictions)
                rejected = total - accepted
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Propositions", total)
                with col2:
                    st.metric("Acceptées", accepted)
                with col3:
                    st.metric("Rejetées", rejected)
                
                # Affichage des différences entre acceptés et rejetés
                st.subheader("📌 Différence entre propositions acceptées et rejetées")
                st.write("### Propositions Acceptées")
                st.write(df_accepted)
                st.write("### Propositions Rejetées")
                st.write(df_rejected)
                
                # Bouton de téléchargement
                st.download_button(
                    "📥 Télécharger les résultats",
                    df.to_csv(index=False).encode('utf-8'),
                    "resultats_classification.csv",
                    "text/csv",
                    key='download-csv'
                )
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
