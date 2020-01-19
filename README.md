# Pet ID
Pet ID is a dog breed classification model and web application

## Running the Backend Server

This project uses [pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management.  
Please install Python and pip on your machine. 

```bash
git clone https://github.com/irvinodjuana/pet-id.git
cd pet-id/
pip install pipenv
pipenv install --dev
```

Pipenv will now manage a virtual environment for the project.
To run the flask server, we can open a new shell in the virtual environment and run the python script:

```bash
pipenv shell
cd src/server/
python app.py
python app.py
```

The server should now be running, and can be accessed through the frontend webpage or any REST client.


## Running the Frontend App

In the frontend web application, we use [npm](https://docs.npmjs.com/) for our JS dependency management.  
Install the node modules, build the app and start the webpage with the following commands from the pet-id/ directory:

```bash
cd src/frontend
npm install
npm run build
npm run start
```

The React webpage should now be running in your browser on http://localhost:3000/!  
If the server is set up and running, you can upload pictures and test the predictions.


