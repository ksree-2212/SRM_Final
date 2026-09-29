import requests, time
url = 'http://localhost:8000/api/v1/super-resolution'
files = {'file': open('data/raw/sentinel2/srm_agriculture.tif', 'rb')}
print('Uploading srm_agriculture.tif to backend...')
response = requests.post(url, files=files)
if response.status_code == 200:
    job_id = response.json()['jobId']
    print(f'Job ID: {job_id}')
    while True:
        st = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}').json()
        print(f'{st["status"]}: {st["message"]}')
        if st['status'] in ['completed', 'failed']:
            print(st.get('outputs', 'No outputs'))
            break
        time.sleep(2)
else:
    print(response.text)
