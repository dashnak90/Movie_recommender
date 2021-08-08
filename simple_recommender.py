import random
import pickle
import numpy as np
import re
import pandas as pd
import requests 
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import time

class Recommender:
    
    def __init__(self):
        self.dic= pickle.load(open("dic.pkl", "rb"))
        self.dic_imdb= pickle.load(open("dic_imdb.pkl", "rb"))

    def get_link(self, x):
        """returns poster link"""
        x=str(x)
        if len(x)==7:
            url=f"https://www.omdbapi.com/?i=tt{x}&apikey=e815a9b1"
        if len(x)==6:
            url=f"https://www.omdbapi.com/?i=tt0{x}&apikey=e815a9b1"
        else:
            url=f"https://www.omdbapi.com/?i=tt00{x}&apikey=e815a9b1"
        page = requests.get(url).json()
#            link=re.findall('"Poster":"([^"]+)',page)
        link=page.get("Poster")
        return link
    
    def get_pics(self, movies):
        "returns list of posters of recommended movies "
        pics=[]
        for i in movies:
            ID=self.dic_imdb[i]
            link=self.get_link(ID)
            if link is None:
                link = 'http://www.absolutesardinia.it/images/not_available.png'
            pics.append(link)
        return pics    
    
    def year(self,y):
        """extracts year from movie title"""
        year=re.findall("\(([0-9]+)\)",y)
        try:
            year=year[0]
        except:
            year=0
        return int(year)    
    
    def nmf_recommendations(self,movie_list, rating_list):
        """Non-negative matrix factorization"""

        mx = pickle.load(open("mx.pkl", "rb"))
        nmf_model = pickle.load(open("nmf.pkl", "rb"))
        nmf_Q = pickle.load(open("nmf_Q.pkl", "rb"))

        
        new_user = np.full(shape=(1,mx.shape[1]), fill_value=mx.mean().mean())
        for i,m in enumerate(movie_list):    
            try:
                new_user[0][self.dic[m]] = rating_list[i]
            except:
                continue
        user_P = nmf_model.transform(new_user) 
        actual_recommendations = np.dot(user_P, nmf_Q)

        recommended_movies=pd.Series(actual_recommendations[0], index=mx.columns)
        user_df=recommended_movies.to_frame('predictions').reset_index()

        user_df['year']=user_df['movie_title'].apply(self.year) #extract year
        user_df=user_df.set_index("movie_title")
        user_df=user_df.loc[~user_df.index.isin(movie_list)] #remove seen movies
        choice=user_df.reset_index().sort_values(by=['predictions']).groupby('year').max()[-10:]['movie_title'].values         #10 best movies in the last 10 years
        new=list(np.random.choice(choice,5, replace=False)) #randomly pick 5
#        new=list(user_df.sort_values(by=['year','predictions'], ascending=False)[:5].index) #return recent unseen movies
        all=list(user_df.sort_values(by='predictions', ascending=False)[:5].index) #return unseen movies
        new_pics=self.get_pics(new)
        all_pics=self.get_pics(all)
        print(new,all,new_pics,all_pics)
        return new,new_pics,all,all_pics



    def nb_recommendations(self,movie_list, rating_list):

        """Neighborhood-based Collaborative Filtering"""

        mx = pickle.load(open("mx_nb.pkl", "rb"))
        
        new_user = np.full(shape=(1,mx.shape[1]), fill_value=0)
        for i,m in enumerate(movie_list):    
            try:  # in case user doesnt provide data
                new_user[0][self.dic[m]] = rating_list[i]
                print('test')
            except:
                continue
        
        target_user = "new_user"
        mx.loc['new_user']=new_user[0]
        cs=cosine_similarity(mx)
        cs = pd.DataFrame(cs, index=mx.index, columns=mx.index)
        related_users = cs.loc[target_user]
        related_users.pop(target_user)
        user_df=mx.loc[related_users.index[np.argmax(related_users)]].to_frame('predictions')
        user_df=user_df.loc[~user_df.index.isin(movie_list)] #remove seen movies
        all=list(user_df.sort_values(by='predictions', ascending=False)[:10].index)
        all_pics=self.get_pics(all)
        print(all,all_pics)
        return all,all_pics



get_rec=Recommender()

if __name__ == "__main__":
    # this block of code is only executed when we exeute this file explicitly.
    # not when importing it   
    # test code
    pass