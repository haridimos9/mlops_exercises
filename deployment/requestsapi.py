import requests
# response = requests.get(
#    'http://127.0.0.1:8000/query_items',
#    params={'item_id': '564',},
# )

# response = requests.get('https://api.github.com/this-api-should-not-exist')
# if response.status_code == 200:
#    print('Success!')
# elif response.status_code == 404:
#    print('Not Found.')
# print(response.json())


# response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')


# with open(r'img.png','wb') as f:
#    f.write(response.content)

# pload = {'username':'Olivia','password':'123',}
# response = requests.post('http://127.0.0.1:8000/login/', params = pload)
# print(response.json())


# headers = {
#     'accept': 'application/json',
#     'Content-Type': 'application/json',
# }
# pload = {'email':'mlops@gmail.com','domain':'gmail'}
# response = requests.post(
#    'http://127.0.0.1:8000/text_model',headers=headers,
#    json=pload
# )
# print(response.json())

headers = {
    'accept': 'application/json',
    # requests won't add a boundary if this header is set when you pass files=
    # 'Content-Type': 'multipart/form-data',
}

files = {
    'data': open('me.jpg', 'rb'),
}

response = requests.post('http://localhost:8000/transformer/', headers=headers, files=files)

print(response.json())