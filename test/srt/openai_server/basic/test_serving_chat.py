"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    # ------------- sampling-params -------------
    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            stop=["</s>"],
        )
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1], None, None, [], ["</s>"], None),
        ):
            params = self.chat._build_sampling_params(req, ["</s>"], None)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["max_new_tokens"], 150)
            self.assertEqual(params["min_new_tokens"], 5)
            self.assertEqual(params["stop"], ["</s>"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
