# -*- coding: utf-8 -*-

from clarifai.rest import ClarifaiApp

# Create your API key in your account's `Manage your API keys` page:
# https://clarifai.com/developer/account/keys

app = ClarifaiApp(api_key='b65f51845a944310b713c9faf20ce8f1')

# You can also create an environment variable called `CLARIFAI_API_KEY` 
# and set its value to your API key.
# In this case, the construction of the object requires no `api_key` argument.

app = ClarifaiApp()

