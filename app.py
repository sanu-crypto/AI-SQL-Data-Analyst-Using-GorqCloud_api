import streamlit as st
import pandas as pd
import sqlite3
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AI SQL Data Analyst", layout="wide")
st.title("📊 AI SQL Data Analyst Agent")

# ------------------ API KEY INPUT ------------------ #
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# ------------------ LLM ------------------ #
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

# ------------------ FILE UPLOAD ------------------ #
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ------------------ CREATE DATABASE ------------------ #
    conn = sqlite3.connect("data.db")
    df.to_sql("data_table", conn, if_exists="replace", index=False)

    st.success("✅ Data stored in SQLite database!")

    # ------------------ USER QUERY ------------------ #
    question = st.text_input("💬 Ask a question about your data")

    if question:
        prompt = f"""
You are a SQL expert.

Table name: data_table
Columns: {list(df.columns)}

Write ONLY SQLite query.

Rules:
- No explanation
- No markdown
- Only SQL query
- Use only the table name data_table

Question: {question}
"""

        try:
            response = llm.invoke(prompt)
            query = response.content.strip()

            query = query.replace("```sql", "").replace("```", "").strip()

            st.subheader("🧠 Generated SQL Query")
            st.code(query, language="sql")

            result_df = pd.read_sql_query(query, conn)

            st.subheader("📊 Result")
            st.dataframe(result_df)

            if not result_df.empty and len(result_df.columns) >= 2:
                st.subheader("📈 Visualization")

                x = result_df.iloc[:, 0]
                y = result_df.iloc[:, 1]

                fig, ax = plt.subplots()
                ax.bar(x.astype(str), y)
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a CSV file.")