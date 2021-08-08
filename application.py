from flask import Flask, render_template, request,jsonify
from simple_recommender import Recommender, get_rec

import pickle
main_moive_list = pickle.load(open("main_moive_list.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    names = main_moive_list
    return render_template('index.html', title='Welcome to The Best Movie Recommender',movienames=names)

@app.route('/nmf_recommender',methods=["POST"])
def nmfrecommender():
    input = dict(request.form)

    print(input)

    movies_list = list(input.values())[::2] #list of movie names
    ratings_list = list(input.values())[1::2] #list of ratings    
   
    print(movies_list,ratings_list)

    recs = get_rec.nmf_recommendations(movies_list, ratings_list)
        # at this point, we would then pass this
    #information as an argument into our recommender function.
    
    return render_template('recommendations.html',
                            movies_new=recs[0], pics_new=recs[1], movies_all=recs[2] ,pics_all=recs[3],title='Your recommendations:')


@app.route('/nb_recommender',methods=["POST"])
def nbrecommender():
    input = dict(request.form)

    print(input)

    movies_list = list(input.values())[::2] #list of movie names
    ratings_list = list(input.values())[1::2] #list of ratings   
    
    print(movies_list,ratings_list) 

    recs = get_rec.nb_recommendations(movies_list, ratings_list)
        # at this point, we would then pass this
    #information as an argument into our recommender function.
    
    return render_template('nb_recommendations.html',
                            movies=recs[0],pics=recs[1] ,title='Your recommendations:')




if __name__ == "__main__":
    app.run(debug=True, port=5000)