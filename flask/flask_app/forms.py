from flask_wtf import FlaskForm
from wtforms import SubmitField

class DishForm(FlaskForm):
    submit = SubmitField('Generate a dish!')