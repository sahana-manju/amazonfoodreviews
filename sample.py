import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from textblob import TextBlob
import pickle
import re
from bs4 import BeautifulSoup


st.set_page_config(page_title="Sentiment Analysis", layout="wide")
#https://images.unsplash.com/photo-1512388908228-685f99a66ac2?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTY4NTA5MjU0NQ&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080
# CSS styles
bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1490818387583-1baba5e638af?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTY4MjYyNDY4NQ&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080');
background-size: cover;
background-repeat: no-repeat;
}
</style>
'''
#https://images.unsplash.com/photo-1528459801416-a9e53bbf4e17
#https://www.gettyimages.ca/detail/photo/splashed-with-fresh-air-royalty-free-image/1127069296?adppopup=true
#WQD6TCLOozg
st.markdown(bg_img, unsafe_allow_html=True)

col1,col2=st.columns([2,4])
with col2:
    st.markdown('<h2 style="color:black";font-size:20px;">Amazon Food review Sentiment Analysis</h3>', unsafe_allow_html=True)





    st.markdown(f'<h1 style="color:red;font-size:25px;">{"Unveiling the Flavor of Feedback!"}</h1>', unsafe_allow_html=True)

    tabs=st.tabs(['About','Sentiment Analysis'])
    with tabs[0]:
    #st.image('/Users/sahanamanjunath/Desktop/food.jpeg')
        st.markdown("<p style='text-align:justify;'>In a world where culinary experiences intertwine with digital expression, Our website emerges as the ultimate destination for unraveling the sentiments behind Amazon Food reviews. As passionate food enthusiasts, we understand the importance of savoring every bite, and we believe that every review tells a unique story.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify;'>We've harnessed the power of sentiment analysis to decode the emotions embedded within Amazon Food reviews. Our platform is a gateway to a deeper understanding of customer experiences, allowing you to navigate the diverse landscape of culinary delights with confidence.</p>", unsafe_allow_html=True)

    with tabs[1]:          
        with open('random_forest_model_SA.pkl', 'rb') as file:
            rf_model = pickle.load(file)
            
        with open('LR_model_SA.pkl', 'rb') as file:
            lr_model = pickle.load(file)
            
        with open('vectorizer_SA.pkl', 'rb') as file:
            tf_idf_vect = pickle.load(file)

        stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
                    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
                    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
                    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
                    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
                    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
                    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
                    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
                    'won', "won't", 'wouldn', "wouldn't"])

        def decontracted(phrase):
            # specific
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)

            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            return phrase

        def predict_sentiment(model, sentence):
            sentence = re.sub(r"http\S+", "", sentence)
            sentence = BeautifulSoup(sentence, 'lxml').get_text()
            sentence = decontracted(sentence)
            sentence = re.sub("\S*\d\S*", "", sentence).strip()
            sentence = re.sub('[^A-Za-z]+', ' ', sentence)
            sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

            text_vector = tf_idf_vect.transform([sentence])
            
            if model == 'LR':
                prob = lr_model.predict_proba(text_vector)
                predicted_class = lr_model.predict(text_vector)
            elif model== 'RF':
                prob = rf_model.predict_proba(text_vector)
                predicted_class = rf_model.predict(text_vector)
            
            
            class_label = "Positive" if predicted_class[0] == 1 else "Negative"
            
            return class_label, prob.max()+0.2

        input_text = st.text_input("Please type in your feedback here",placeholder="Type your feedback here...")
        arrow_clicked = st.button("Submit")

        # Check if the arrow button is clicked
        if arrow_clicked and input_text :
            success_placeholder = st.empty()
            success_placeholder.success(f"Sentiment analysis started")
            

            # Perform actions based on the input text
            i = 0
            chart=st.empty()
            while i < 10:
                
                min_value = 0
                max_value = 1
                random_array = min_value + (max_value - min_value) * np.random.rand(2, 1)
                chart_data = pd.DataFrame(random_array, columns=["Positive"])
                colors=['grey','red']
                fig = go.Figure()
                
                x_t=['Positive','Negative']
                fig.add_trace(go.Bar(x=x_t, y=chart_data["Positive"], width=0.25,marker=dict(color=colors)))
                fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=['Positive', 'Negative']))
                fig.update_layout(
                    xaxis=dict(tickvals=[0, 1], ticktext=['Positive', 'Negative']),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper
                    )
                fig.update_yaxes(showgrid=False)
                chart.plotly_chart(fig, use_container_width=True)
                time.sleep(0.3)
                i += 1
            

            
            model='RF'
            class_label,sentiment_strength=predict_sentiment(model,input_text)
            testimonial = TextBlob(input_text)
            polarity = testimonial.sentiment.polarity
            sentiment_strength = abs(polarity)
            
            if sentiment_strength == 0:
                sentiment_strength = 0.67
            if polarity >= 0:
            #if class_label=='Positive':
                y_t=[sentiment_strength,abs(1-sentiment_strength)] 
                st.success('Congratulations! The customer gave a Positive feedback!')
            else:
                y_t=[abs(1-sentiment_strength), sentiment_strength]
                st.error('Oops.. Sorry the customer gave a Negative feedback')
            
                
            fig.update_traces(go.Bar(x=x_t, y=y_t, width=0.25,marker=dict(color=colors)))
            fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=['Positive', 'Negative']))
            fig.update_yaxes(showgrid=False)
            chart.plotly_chart(fig, use_container_width=True)
            success_placeholder.success(f"Sentiment analysis done")
            
            
        if arrow_clicked and not input_text:
            st.error(f"Please enter your feedback")


