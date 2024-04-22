import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import contractions
import plotly.graph_objects as go
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from io import BytesIO
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pyvis.network import Network
import streamlit.components.v1 as components
import spacy

st.set_page_config(page_title="Dataviz", page_icon="📈")

st.markdown("# Data Visualization")
st.sidebar.header("Data Visualization")
st.sidebar.write("This section focuses on interpreting the results of the tiktok/youtube videos")

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    # Get spaCy's set of English stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    # Tokenize the text using spaCy
    doc = nlp(text)
    # Filter out stopwords
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words]
    return " ".join(filtered_tokens)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

data= st.session_state.get("data", None)
chart_data= st.session_state.get("chart_data", None)
for i in range(0,len(data)):
    contracted_data=contractions.fix(data[i]["transcript"])
    text_data = remove_stopwords(contracted_data)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text_data])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords = [feature_names[idx] for idx in tfidf_matrix.toarray()[0].argsort()[-10:][::-1]]
    relationships = defaultdict(int)
    tokens = text_data.lower().split()
    for i in range(len(tokens)):
        if tokens[i] in top_keywords:
            for j in range(i + 1, len(tokens)):
                if tokens[j] in top_keywords:
                    relationships[(tokens[i], tokens[j])] += 1
    #Pyvis plot
    nodes = set()
    edges = []
    for relationship, weight in relationships.items():
        source, target = relationship
        if source != target:  
            nodes.add(source)
            nodes.add(target)
            edges.append((source, target, weight))
    net = Network(height="500px", width="100%")
    for node in nodes:
        net.add_node(node, label=node, shape="dot")
    max_weight = max([weight for _, _, weight in edges])
    for source, target, weight in edges:
        scaled_weight = 1 + 2 * (weight / max_weight)
        net.add_edge(source, target, value=scaled_weight, width=scaled_weight*0.2)
    net.barnes_hut(gravity=-1000, central_gravity=0.3, spring_length=200)
    path = '/tmp'
    net.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
    components.html(HtmlFile.read(), height=435)
    #Adding bar chart to show number of videos containing misinformation
    st.subheader("Misinformation Status Frequency for Videos Uploaded")
    bar_chart=st.bar_chart(chart_data, x="Misinformation Status", y="Frequency")
  
