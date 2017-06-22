
import requests
import hmac
import hashlib
import time
import struct
import json
import base64
import httplib2

root = "http://hdechallenge-solve.appspot.com/challenge/003/endpoint"
host = "http://hdegip.appspot.com/challenge/003/endpoint"
url = "challenge/003/endpoint"
content_type = "application/json"
userid = "saopayne@gmail.com"
secret_suffix = "HDECHALLENGE003"
shared_secret = userid+secret_suffix

timestep = 30
T0 = 0


def HOTP(K, C, digits=10):
    """HTOP:
    K is the shared key
    C is the counter value
    digits control the response length
    """
    K_bytes = str.encode(K)
    C_bytes = struct.pack(">Q", C)
    hmac_sha512 = hmac.new(key = K_bytes, msg=C_bytes, digestmod=hashlib.sha512).hexdigest()
    return Truncate(hmac_sha512)[-digits:]


def Truncate(hmac_sha512):
    """truncate sha512 value"""
    offset = int(hmac_sha512[-1], 16)
    binary = int(hmac_sha512[(offset *2):((offset*2)+8)], 16) & 0x7FFFFFFF
    return str(binary)


def TOTP(K, digits=10, timeref = 0, timestep = 30):
    """TOTP, time-based variant of HOTP
    digits control the response length
    the C in HOTP is replaced by ( (currentTime - timeref) / timestep )
    """
    C = int ( time.time() - timeref ) / timestep
    # C = int ( time.time() - timeref ) / timestep
    return HOTP(K, C, digits = digits)

data = {
   "github_url": "https://gist.github.com/saopayne/ef3a68967d1a5a60a8d911b54fbf68a1",
   "contact_email": "saopayne@gmail.com"
}

passwd = TOTP(shared_secret, 10, T0, timestep).zfill(10)

# base64 encode the username and password
auth = base64.encodestring('%s:%s' % (userid, passwd)).replace('\n', '')
# print("Basic %s" % auth)
# headers = {
#     "Host": "admissionchallenge.example.com",
#     "Connection": "keep-alive",
#     "Content-Length": "104",
#     "Content-Type": "application/json",
#     "Accept": "*/*",
#     "Authorization": "Basic %s" % auth
# }
# # Adding empty header as parameters are being sent in payload
# # headers = {"Authorization": "Basic %s" % auth}
# print(headers)
# r = requests.post(root, data=json.dumps(data), headers=headers)
#
# print(r.content)


h = httplib2.Http()
h.add_credentials( userid, passwd )
header = {"content-type": "application/json"}
resp, content = h.request(root, "POST", headers = header, body = json.dumps(data))
print(resp)
print(content)

