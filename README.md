MockSet
=======

MockSet is an NLP project that allows a user to input long and short description and provides type, size, length and material as the output.

Installation
------------
Start project by creating a virtual environment

`pip install pip`

`pip install virtualvenv`

`virtualvenv env_name`

Navigate into the virtual environment:

Linux: `source env_name/bin/activate`
Windows: `\pathto\env\Scripts\activate`

Install the libraries as indicated in `requirements.txt` using:

`pip install -r requirements.txt`

Run Server
-----------
* Navigate into folder with `manage.py`
* Open folder in terminal and run server through `python manage.py runserver`

Retrain Model
-------------
1. Ensure the csv is on same file as Mock_Set.py
2. Open shell/terminal and run python3 Mock_Set.py
3. Enter filename and sheet name (both mandatory)
4. Model is trained, saved and accuracy displayed.

Run your predictions on saved trained model.

## Author
Carol Muchemi

### References:
https://pypi.org/project/pip/
https://note.nkmk.me/en/python-pip-install-requirements/
https://www.liquidweb.com/kb/how-to-setup-a-python-virtual-environment-on-windows-10/
