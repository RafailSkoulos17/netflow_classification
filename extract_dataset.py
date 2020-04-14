# import json
# import urllib
#
# response = urllib2.urlopen('https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow/')
# data = json.loads(response.read())
#
# print(type(data))  # prints <type 'dict'>
#
# print(data['tribe'])  # prints "Three Feathers"

import requests

link = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54/detailed-bidirectional-flow-labels/capture20110815-3.binetflow"
# f = requests.get(link,verify=False)
# print(f.text)

response = requests.get(link, stream=True, verify=False)

# Throw an error for bad status codes
response.raise_for_status()

with open('pypackage/flexfringe/datasets/bidirectional/CTU-Malware-Capture-Botnet-54', 'wb') as handle:
    for block in response.iter_content(1024):
        handle.write(block)
