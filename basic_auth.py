from flask import Response, Flask

app = Flask(__name__)

# @requires_auth
@app.route('/secret-page')
def secret_page():
#    return render_template('secret_page.html')
    return Response(
    'Could not verify your access level for that URL.\nYou have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})
