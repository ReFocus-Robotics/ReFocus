"""Provider contract tests use mock HTTP; they do not call external models."""
import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from structrl.rsi.advisor import _provider_settings, _schema, _structured_call
from structrl.rsi.compare_advisors import main as compare_main


class AdvisorProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'frame.jpg'
        self.image.write_bytes(b'\xff\xd8\xff\xd9')
        self.schema = _schema('result', {'ok': {'type': 'boolean'}}, ['ok'])

    def fake_response(self, envelope):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(envelope).encode()
        opener = Mock()
        opener.open.return_value = response
        return opener

    def test_qwen_compatible_prompt_mode_sends_image_without_astra_options(self):
        opener = self.fake_response({'model': 'actual-qwen', 'choices': [{'message': {'content': '{"ok":true}'}}]})
        with patch.dict(os.environ, {'ADVISOR_API_KEY': 'sensitive'}, clear=True), patch(
                'structrl.rsi.advisor.urllib.request.build_opener', return_value=opener):
            result = _structured_call(self.root, 'request.json', 'Inspect', base='https://host/v1',
                model='Qwen/Qwen3.8-27B', reasoning_effort=None, response_schema=self.schema,
                images=[('frame', self.image)], json_mode='prompt')
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(result, {'ok': True})
        self.assertNotIn('reasoning_effort', payload)
        self.assertNotIn('response_format', payload)
        self.assertEqual(payload['messages'][0]['content'][1]['type'], 'image_url')
        self.assertNotIn('sensitive', (self.root/'request.json').read_text())
        attempts=json.loads((self.root/'request.json.attempts.json').read_text())
        self.assertEqual(attempts['attempts'][0]['served_model'], 'actual-qwen')

    def test_anthropic_native_request_and_image(self):
        opener = self.fake_response({'model': 'claude-test', 'content': [{'type': 'text', 'text': '{"ok":true}'}]})
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'secret'}, clear=True), patch(
                'structrl.rsi.advisor.urllib.request.build_opener', return_value=opener):
            result = _structured_call(self.root, 'anthropic.json', 'Inspect', base='https://api.anthropic.com/v1',
                model='claude-test', reasoning_effort=None, response_schema=self.schema,
                images=[('frame', self.image)], provider='anthropic', max_tokens=500)
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(result, {'ok': True})
        self.assertEqual(request.full_url, 'https://api.anthropic.com/v1/messages')
        self.assertEqual(payload['max_tokens'], 500)
        self.assertEqual(payload['messages'][0]['content'][1]['source']['media_type'], 'image/jpeg')
        self.assertNotIn('secret', (self.root/'anthropic.json').read_text())

    def test_astra_low_and_generic_model_have_explicit_separate_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_provider_settings('https://host/v1', 'gpt-6-astra')[2], 'low')
            self.assertIsNone(_provider_settings('https://host/v1', 'Qwen/Qwen3.8-27B')[2])
            with self.assertRaises(ValueError):
                _provider_settings('https://host/v1', 'gpt-6-astra', 'high')

    def test_comparison_keeps_source_untouched_and_uses_committed_attempt(self):
        source = self.root/'source'
        source.mkdir()
        (source/'r0_global').mkdir()
        (source/'r0_global'/'complete.json').write_text(json.dumps({'phase_attempt_id': 'committed'}))
        output = self.root/'result'
        argv = ['compare', '--source-root', str(source), '--output-root', str(output),
                '--round', '0', '--checkpoints', '5', '10', '--advisor-model', 'Qwen/model',
                '--advisor-base-url', 'https://host/v1', '--advisor-json-mode', 'prompt']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'ADVISOR_API_KEY': 'sensitive'}, clear=True), \
                patch('structrl.rsi.compare_advisors.select', return_value={}) as call:
            compare_main()
        self.assertEqual(call.call_args.kwargs['phase_attempt_id'], 'committed')
        self.assertEqual(call.call_args.kwargs['output_root'], output.resolve())
        self.assertFalse((source/'windows_round0.json').exists())
        self.assertNotIn('sensitive', (output/'comparison_config.json').read_text())


if __name__ == '__main__':
    unittest.main()
