# Style_Transfer
A simple implementation of [style transfer](https://arxiv.org/abs/1508.06576) which works for both images and videos.

## Dependencies
Install all the required python dependencies using the following code :

`pip3 install -r requirements.txt`

## Run
### For Images

`python3 transfer_style.py -c_p <content_image_path> -s_p <style_image_path>`

### For Videos

`python3 transfer_style.py -c_p <content_image_path> -s_p <style_image_path> -v`

*Note:* Code is tested on python3 and is not guaranteed to work on python2.
