import requests
import time

url = "http://localhost:8000/api/v1/super-resolution"
files = {'file': open('test.tif', 'rb')}
print("Uploading test.tif to backend...")
response = requests.post(url, files=files)
if response.status_code == 200:
    job_data = response.json()
    job_id = job_data["jobId"]
    print(f"Job scheduled! ID: {job_id}")
    
    # Poll for completion
    while True:
        status_url = f"http://localhost:8000/api/v1/jobs/{job_id}"
        st_res = requests.get(status_url)
        st_data = st_res.json()
        print(f"Status: {st_data['status']} - {st_data['message']}")
        if st_data['status'] in ['completed', 'failed']:
            print("Finished!")
            if 'outputs' in st_data:
                print("Outputs generated:", st_data['outputs'])
            break
        time.sleep(2)
else:
    print(f"Failed to submit: {response.status_code}")
    print(response.text)
