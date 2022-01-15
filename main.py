# import train
# import test
from model_training.test import get_model
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs
import torch
from pprint import pprint
data = {'result': 'this is a test'}
host = ('localhost', 8888)

loaded_model, tokenizer, label_keys = get_model()
print(label_keys)


class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')

        query = parse_qs(urlparse(self.path).query)
        self.end_headers()
        # for k, v in query.items():
        #     if k == 'command':
        if query["command"]:
            cmd = query['command']
            model_inputs = tokenizer(cmd, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")
            pt_outputs = loaded_model(**model_inputs)
            pt_outputs_digits = pt_outputs.logits.softmax(dim=1)
            pt_outputs = pt_outputs_digits.argmax(dim=1)
            results = {t: [label_keys[label], digit[label].item()] for t, label,
                       digit in zip(cmd, pt_outputs, pt_outputs_digits)}

        self.wfile.write(json.dumps(
            {'path': self.path, 'query': query, 'results': results}).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
    print('bye')
