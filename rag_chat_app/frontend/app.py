import streamlit as st
import requests

st.title("Chat with RAG System")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "bot":
        st.markdown(f"**Bot:** {message['content']}")
        if "sources" in message:
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']}) (Relevance: {source['relevance_score']})")


# Temporary variable to hold user input
if "input_value" not in st.session_state:
    st.session_state["input_value"] = ""

# Text input linked to `input_value`
user_input = st.text_input("Type your message here:", value=st.session_state["input_value"])

# Send button
if st.button("Send"):
    if user_input:
        # result = {
        #     'answer': 'dfrrdg',
        #     'sources': [{
        #         'title': 'title',
        #         'url': 'url',
        #         'relevance_score': '0.1'
        #     }]
        # }
        response = requests.post("http://localhost:8000/chat", json={"user_input": user_input})
        if response.status_code == 200:
            result = response.json()
        # if result:
            bot_response = result.get("answer", "No answer")
            sources = result.get("sources", [])

            # Append to chat history
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "bot", "content": bot_response, "sources": sources})

            # Reset input_value safely
            st.session_state["input_value"] = ""  # This will reset next rerun

            st.rerun()
        else:
            st.error("Error: Unable to get response from the server.")
    else:
        st.warning("Please enter a message.")
