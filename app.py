import json
import re
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request



def find_related_words(input_list, model, related_keywords, similarity_threshold=0.5):
	related_words = set()
    
	for keyword in related_keywords:
		if keyword not in model:
			continue
		
		for word in input_list:
			if word not in model:
				continue

			similarity = model.similarity(keyword, word)
			if similarity >= similarity_threshold:
				related_words.add(word)

	return list(related_words)



def get_related_words(df, model, input_):
	df_DT = df[['DAO', 'Description_Tokens']]
	vectorizer = CountVectorizer()
	vectorizer.fit(df['Description_Tokens'].apply(' '.join))
	X = vectorizer.transform(df['Description_Tokens'].apply(' '.join))
	
	word_counts = X.toarray().sum(axis=0)
	vocab = vectorizer.get_feature_names_out()
	sorted_vocab = [x for _,x in sorted(zip(word_counts, vocab), reverse=True)]

	input_list = sorted_vocab

	keywords_input = input_
    
	related_keywords = [keyword.strip() for keyword in keywords_input.split(',')]

	similarity_threshold = 0.7
	related_words = find_related_words(input_list, model, related_keywords, similarity_threshold)

	return related_words




# Function to query proposals
def query_proposals(lim, off):
	query = f"""query Governors {{
		governors (
        		chainIds: ["eip155:1"],
        		pagination: {{
        			limit:{lim},
				offset:{off}
 	    		}}
      )
      {{
        name
        proposals {{
		id,
          	title,
          	description
        	}}
      	}}
    	}}"""
	return query




# Function to preprocess text
def preprocess_text(text):
	if not isinstance(text, str):
		return []

	text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
	text = text.lower()  # Convert to lowercase
	words = nltk.word_tokenize(text)  # Tokenize words

	# Define negations
	negations = {'not', 'no'}

	# Remove stopwords, but keep negations
	words = [word for word in words if word not in stopwords.words('english') or word in negations]

	# Lemmatize words
	lemmatizer = WordNetLemmatizer()
	words = [lemmatizer.lemmatize(word) for word in words]

	return words




def fetch_and_process_data(related_words):
	url = 'https://api.tally.xyz/query'
	headers = {"Api-key": 'a6a335a19f8b493c976e78566f7e1795dee68409b0e7dbd2ef98247edaafea93'}

	df = pd.DataFrame()
	for x in range(0, 200, 5):
		r = requests.post(url, json={'query': query_proposals(5, x)}, headers=headers)
		json_data = json.loads(r.text)
		space_lst = []
		proposal_id_lst = []
		title_lst = []
		description_lst = []
		for i in range(len(json_data['data']['governors'])):
			for j in range(len(json_data['data']['governors'][i]['proposals'])):
				space_lst.append(json_data['data']['governors'][i]['name'])
				proposal_id_lst.append(json_data['data']['governors'][i]['proposals'][j]['id'])
				title_lst.append(json_data['data']['governors'][i]['proposals'][j]['title'])        
				description_lst.append(json_data['data']['governors'][i]['proposals'][j]['description'])
		df_temp = pd.DataFrame({'DAO': space_lst, 'Proposal_ID': proposal_id_lst, 'Title': title_lst, 'Description': description_lst})
		df = pd.concat([df, df_temp])
	df.reset_index(drop=True, inplace=True)
	df['Description_Tokens'] = df['Description'].apply(preprocess_text)
	df['Word_Count'] = df['Description_Tokens'].apply(lambda x: sum(1 for word in x if word in related_words))
	sorted_df = df.sort_values(by='Word_Count', ascending=False)

	return sorted_df



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
        	words = request.form['words']
        	df = fetch_and_process_data([])
        	model = api.load('fasttext-wiki-news-subwords-300')
        	related_words = get_related_words(df, model, words)
        	sorted_df = fetch_and_process_data(related_words)
        	table = sorted_df.to_html(classes="data", header="true", table_id="dataframe")
        	return render_template('index.html', table=table)
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
