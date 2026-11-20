"""Mock HTTP responses to check direct HTML/PDF fetching and optional Jina."""
import argparse
import ast
from io import BytesIO
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import requests
    import pdfplumber
    from bs4 import BeautifulSoup
except ImportError:
    requests = None


def minimal_pdf():
    stream = b'BT /F1 12 Tf 72 720 Td (Legal rules.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>',
               b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    document = b'%PDF-1.4\n'
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document += f'{number} 0 obj\n'.encode() + obj + b'\nendobj\n'
    start = len(document)
    document += b'xref\n0 6\n0000000000 65535 f \n'
    for offset in offsets:
        document += f'{offset:010d} 00000 n \n'.encode()
    document += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF'.encode()
    return document


@unittest.skipIf(requests is None, 'Install requests, beautifulsoup4, and pdfplumber to test fetching')
class WebFetchTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'glare/serper.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_text_from_url')
        ns = dict(requests=requests, BeautifulSoup=BeautifulSoup, BytesIO=BytesIO,
                  pdfplumber=pdfplumber, re=re)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'serper', 'exec'), ns)
        self.fetch = ns['extract_text_from_url']

    def response(self, content, content_type, status=200):
        response = requests.Response()
        response.status_code = status
        response.url = 'https://example.com/legal'
        response._content = content
        response.headers['Content-Type'] = content_type
        return response

    def test_direct_html_without_jina(self):
        response = self.response('<html><style>hidden</style><script>bad()</script><p>法律规则</p></html>'.encode(), 'text/html')
        with patch.object(requests, 'get', return_value=response) as get:
            self.assertEqual(self.fetch(response.url), '法律规则')
            self.assertEqual(get.call_args.args[0], response.url)
            self.assertEqual(get.call_args.kwargs['timeout'], 20)

    def test_direct_pdf_by_content_type_or_signature(self):
        for content_type in ['application/pdf', 'application/octet-stream']:
            response = self.response(minimal_pdf(), content_type)
            with patch.object(requests, 'get', return_value=response):
                self.assertIn('Legal rules.', self.fetch(response.url))

    def test_jina_and_optional_credentials(self):
        response = self.response(b'Legal --- rules', 'text/plain; charset=utf-8')
        for key in ['secret-test', None, 'None']:
            with patch.object(requests, 'get', return_value=response) as get:
                self.assertEqual(self.fetch(response.url, True, key), 'Legal - rules')
                self.assertEqual(get.call_args.args[0], 'https://r.jina.ai/' + response.url)
                self.assertEqual('Authorization' in get.call_args.kwargs['headers'], key == 'secret-test')

    def test_http_and_timeout_errors(self):
        with patch.object(requests, 'get', return_value=self.response(b'failed', 'text/html', 403)):
            self.assertTrue(self.fetch('https://example.com').startswith('HTTP error occurred:'))
        with patch.object(requests, 'get', side_effect=requests.exceptions.Timeout):
            self.assertIn('timed out', self.fetch('https://example.com'))

    def test_length_limit(self):
        with patch.object(requests, 'get', return_value=self.response(b'<p>' + b'a' * 9000 + b'</p>', 'text/html')):
            self.assertEqual(len(self.fetch('https://example.com')), 8000)


class JinaArgumentTests(unittest.TestCase):
    def test_explicit_booleans(self):
        tree = ast.parse((ROOT / 'glare/glare.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['parse_bool', 'parse_args']]
        ns = {'argparse': argparse}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'glare', 'exec'), ns)
        base = ['glare.py', '--model_name', 'qwq', '--model_path', 'model', '--multiljp_file_path', 'corpus']
        for value, expected in [('False', False), ('True', True), ('0', False), ('1', True)]:
            with patch.object(sys, 'argv', base + ['--use_jina', value]):
                self.assertEqual(ns['parse_args']().use_jina, expected)
        with self.assertRaises(argparse.ArgumentTypeError):
            ns['parse_bool']('flase')


if __name__ == '__main__':
    unittest.main()
