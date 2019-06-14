from flask import request
from flask_app import predictor_api
from flask_app import forms
from flask_app import app
import flask
# Initialize the app


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
# @app.route("/")
# def hello():
#     return flask.send_file("static/html/index.html")


@app.route("/", methods=["GET", "POST"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)

    print(request.method)
    print(request.form.get('submit_button'))
    if request.method == "POST":
        print('POST', '::'*50)
        if request.form['submit_button'] == "Generate":
            print('HERE' + '#' * 60)
            dish1 = predictor_api.main(temp = 0.15)
            dish2 = predictor_api.main(temp = 0.3)
            dish3 = predictor_api.main(temp = 0.45)
            dish4 = predictor_api.main(temp = 0.6)
            dish5 = predictor_api.main(temp = 0.75)
    else:
        dish1 = None
        dish2 = None
        dish3 = None
        dish4 = None
        dish5 = None

    # print("DISH", dish, '=='*50)
    # form = forms.DishForm()
    # if form.validate_on_submit():
    #     dish = predictor_api.main()
    # else: 
    #     dish = None
    return flask.render_template('bootstrap.html', dish1=dish1, dish2=dish2, dish3=dish3, dish4=dish4, dish5=dish5)

@app.route('/about')
def about() -> str:
    return flask.render_template("about.html")

# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()
