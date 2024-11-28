import requests

def upload_image_to_imgbb(image_path, api_key="0e7fb6d183b8db925812dee230f71079"):
    """
    Uploads an image to ImgBB and returns the URL.

    :param image_path: Path to the local image
    :param api_key: ImgBB API key
    :return: URL of the uploaded image
    """
    try:
        # API endpoint for ImgBB
        url = "https://api.imgbb.com/1/upload"
        
        # Open the image in binary mode
        with open(image_path, "rb") as image_file:
            # Send POST request to upload the image
            response = requests.post(
                url,
                data={"key": api_key},
                files={"image": image_file}
            )
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print(f'Uploaded to {data["data"]["url"]}')
            return data["data"]["url"]
        else:
            raise Exception(f"Error uploading image: {response.status_code}, {response.text}")
    except Exception as e:
        return str(e)

# # Replace with your local image path and ImgBB API key
# image_path = "fridge.JPG"  # Replace this with your local image path
# api_key = "0e7fb6d183b8db925812dee230f71079"         # Get your API key from https://api.imgbb.com/

# uploaded_url = upload_image_to_imgbb(image_path, api_key)
# print(f"Uploaded image URL: {uploaded_url}")
